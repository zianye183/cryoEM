"""Generate offline CryoSPARC particle-location picks from vesicle masks."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from scipy import ndimage

from segmentation_integration.cryosparc_io import (
    decode_path,
    patch_csg_cs_path,
    write_cs,
)
from segmentation_integration.vesicle_masks import load_vesicle_masks, mask_path_for_uid


@dataclass(frozen=True)
class VesicleMembranePickOutputs:
    picks_cs: Path
    picks_csg: Path
    summary_json: Path


@dataclass(frozen=True)
class MicrographMetadata:
    uid: int
    path: Any
    shape: Any
    exp_group_id: int
    psize_a: float
    path_dtype: np.dtype
    shape_dtype: np.dtype


PARTICLE_LOCATION_FIELDS = {
    "location/micrograph_uid",
    "location/micrograph_path",
    "location/micrograph_shape",
}

EXPOSURE_PATH_FIELDS = ("micrograph_blob/path", "blob/path")
EXPOSURE_SHAPE_FIELDS = ("micrograph_blob/shape", "blob/shape")
EXPOSURE_PSIZE_FIELDS = ("micrograph_blob/psize_A", "blob/psize_A")
EXPOSURE_EXP_GROUP_FIELDS = (
    "ctf/exp_group_id",
    "micrograph_blob/exp_group_id",
    "mscope_params/exp_group_id",
)


def generate_pick_indices(
    masks: list[dict[str, Any]],
    *,
    psize_a: float,
    downsample: int,
    box_size_a: float,
    dilation_radius_a: float = 0.0,
    mode: str = "edge",
) -> np.ndarray:
    """Return full-resolution ``(y, x)`` pick indices from downsampled masks."""
    validate_pick_parameters(psize_a=psize_a, downsample=downsample, box_size_a=box_size_a, mode=mode)
    if not masks:
        return np.zeros((0, 2), dtype=np.uint32)

    segmentations = transformed_segmentations(
        masks,
        psize_a=psize_a,
        downsample=downsample,
        dilation_radius_a=dilation_radius_a,
    )
    pick_mask = pick_mask_for_mode(segmentations, mode=mode)
    spacing_px = int(np.round(box_size_a / (psize_a * downsample)))
    spacing_px = max(1, spacing_px)
    downsampled_indices = pick_one_pixel_per_grid_cell(pick_mask, spacing_px)
    if len(downsampled_indices) == 0:
        return np.zeros((0, 2), dtype=np.uint32)
    return (downsampled_indices * downsample).astype(np.uint32)


def validate_pick_parameters(*, psize_a: float, downsample: int, box_size_a: float, mode: str) -> None:
    if psize_a <= 0:
        raise ValueError("--psize-a must be positive")
    if downsample <= 0:
        raise ValueError("--downsample must be positive")
    if box_size_a <= 0:
        raise ValueError("--box-size-a must be positive")
    if mode not in {"edge", "surface"}:
        raise ValueError("--mode must be either 'edge' or 'surface'")


def transformed_segmentations(
    masks: list[dict[str, Any]],
    *,
    psize_a: float,
    downsample: int,
    dilation_radius_a: float,
) -> list[np.ndarray]:
    iterations = int(np.round(abs(dilation_radius_a) / (psize_a * downsample)))
    out = []
    first_shape = np.asarray(masks[0]["segmentation"], dtype=bool).shape
    for mask in masks:
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        if segmentation.shape != first_shape:
            raise ValueError("All masks for a micrograph must have the same shape")
        if iterations > 0 and dilation_radius_a > 0:
            segmentation = ndimage.binary_dilation(segmentation, iterations=iterations)
        elif iterations > 0 and dilation_radius_a < 0:
            segmentation = ndimage.binary_erosion(segmentation, iterations=iterations)
        out.append(segmentation)
    return out


def pick_mask_for_mode(segmentations: list[np.ndarray], *, mode: str) -> np.ndarray:
    first_shape = segmentations[0].shape
    pick_mask = np.zeros(first_shape, dtype=bool)
    for segmentation in segmentations:
        if mode == "surface":
            pick_mask |= segmentation
        else:
            edge = segmentation & ~ndimage.binary_erosion(segmentation)
            pick_mask |= edge
    return pick_mask


def pick_one_pixel_per_grid_cell(pick_mask: np.ndarray, spacing_px: int) -> np.ndarray:
    """Select the occupied pixel nearest each grid-cell center."""
    picks: list[tuple[int, int]] = []
    height, width = pick_mask.shape
    for y0 in range(0, height, spacing_px):
        y1 = min(y0 + spacing_px, height)
        for x0 in range(0, width, spacing_px):
            x1 = min(x0 + spacing_px, width)
            patch = pick_mask[y0:y1, x0:x1]
            indices = np.argwhere(patch)
            if len(indices) == 0:
                continue
            center = np.array([spacing_px / 2, spacing_px / 2])
            distances = np.sum((indices - center) ** 2, axis=1)
            best_y, best_x = indices[int(np.argmin(distances))]
            picks.append((int(y0 + best_y), int(x0 + best_x)))
    return np.asarray(picks, dtype=np.uint32)


def pick_fractional_coordinates(
    *,
    y_mask: int,
    x_mask: int,
    micrograph_shape: tuple[int, int] | np.ndarray,
    downsample: int,
) -> tuple[float, float]:
    """Convert a downsampled mask coordinate to CryoSPARC ``(x_frac, y_frac)``."""
    shape = np.asarray(micrograph_shape, dtype=float)
    if shape.shape[0] < 2:
        raise ValueError("location/micrograph_shape must contain at least two dimensions")
    ny, nx = shape[:2]
    return (float(x_mask * downsample) / float(nx), float(y_mask * downsample) / float(ny))


def export_vesicle_membrane_picks(
    *,
    metadata_cs: Path | None = None,
    metadata_csg: Path | None = None,
    particles_cs: Path | None = None,
    particles_csg: Path | None = None,
    vesicle_mask_dir: Path,
    output_dir: Path,
    psize_a: float = 1.068,
    downsample: int = 4,
    box_size_a: float = 100.0,
    dilation_radius_a: float = 0.0,
    mode: str = "edge",
    progress: bool = False,
    progress_stream: TextIO = sys.stderr,
) -> VesicleMembranePickOutputs:
    """Write offline vesicle membrane picks as CryoSPARC-compatible files."""
    validate_pick_parameters(psize_a=psize_a, downsample=downsample, box_size_a=box_size_a, mode=mode)
    metadata_cs, metadata_csg = resolve_metadata_paths(
        metadata_cs=metadata_cs,
        metadata_csg=metadata_csg,
        particles_cs=particles_cs,
        particles_csg=particles_csg,
    )
    metadata_kind, representatives = load_micrograph_metadata(metadata_cs, default_psize_a=psize_a)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    micrograph_summaries = []
    total_micrographs = len(representatives)
    for index, (uid, metadata) in enumerate(representatives.items(), start=1):
        mask_path = mask_path_for_uid(vesicle_mask_dir, uid)
        if not mask_path.exists():
            micrograph_summaries.append(
                micrograph_summary(metadata, status="missing_mask", pick_count=0)
            )
            emit_progress(
                progress_stream,
                enabled=progress,
                index=index,
                total=total_micrographs,
                uid=uid,
                status="missing_mask",
                pick_count=0,
            )
            continue
        masks = load_vesicle_masks(mask_path)
        pick_indices = generate_pick_indices(
            masks,
            psize_a=psize_a,
            downsample=downsample,
            box_size_a=box_size_a,
            dilation_radius_a=dilation_radius_a,
            mode=mode,
        )
        for y_full, x_full in pick_indices:
            rows.append({"metadata": metadata, "y_full": int(y_full), "x_full": int(x_full)})
        micrograph_summaries.append(
            micrograph_summary(metadata, status="picked", pick_count=len(pick_indices))
        )
        emit_progress(
            progress_stream,
            enabled=progress,
            index=index,
            total=total_micrographs,
            uid=uid,
            status="picked",
            pick_count=len(pick_indices),
        )

    picks_cs = output_dir / "vesicle_membrane_picks.cs"
    picks_csg = output_dir / "vesicle_membrane_picks.csg"
    summary_json = output_dir / "vesicle_membrane_picks_summary.json"
    picks = build_output_particles(list(representatives.values()), rows)
    write_cs(picks_cs, picks)
    patch_csg_cs_path(metadata_csg, picks_csg, metadata_cs.name, picks_cs)
    write_pick_summary(
        summary_json,
        metadata_cs=metadata_cs,
        metadata_csg=metadata_csg,
        metadata_kind=metadata_kind,
        vesicle_mask_dir=vesicle_mask_dir,
        outputs=VesicleMembranePickOutputs(picks_cs=picks_cs, picks_csg=picks_csg, summary_json=summary_json),
        parameters={
            "psize_a": psize_a,
            "downsample": downsample,
            "box_size_a": box_size_a,
            "dilation_radius_a": dilation_radius_a,
            "mode": mode,
        },
        micrographs=micrograph_summaries,
        total_picks=len(picks),
    )
    return VesicleMembranePickOutputs(picks_cs=picks_cs, picks_csg=picks_csg, summary_json=summary_json)


def emit_progress(
    stream: TextIO,
    *,
    enabled: bool,
    index: int,
    total: int,
    uid: int,
    status: str,
    pick_count: int,
) -> None:
    if not enabled:
        return
    print(
        f"[vesicle membrane picks] {index}/{total} done uid={uid} status={status} picks={pick_count}",
        file=stream,
        flush=True,
    )


def resolve_metadata_paths(
    *,
    metadata_cs: Path | None,
    metadata_csg: Path | None,
    particles_cs: Path | None,
    particles_csg: Path | None,
) -> tuple[Path, Path]:
    resolved_cs = metadata_cs if metadata_cs is not None else particles_cs
    resolved_csg = metadata_csg if metadata_csg is not None else particles_csg
    if resolved_cs is None or resolved_csg is None:
        raise ValueError("Provide either metadata_cs/metadata_csg or particles_cs/particles_csg")
    return Path(resolved_cs), Path(resolved_csg)


def load_micrograph_metadata(path: Path, *, default_psize_a: float) -> tuple[str, dict[int, MicrographMetadata]]:
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise ValueError(f"{path} is not a structured CryoSPARC .cs array")
    names = set(arr.dtype.names)
    if PARTICLE_LOCATION_FIELDS <= names:
        return "particle", metadata_from_particle_rows(arr, default_psize_a=default_psize_a)
    if "uid" in names and any(field in names for field in EXPOSURE_PATH_FIELDS):
        return "exposure", metadata_from_exposure_rows(arr, default_psize_a=default_psize_a)
    raise ValueError(
        f"{path} does not look like a particle or exposure metadata .cs file; "
        "expected location/* fields or micrograph_blob/* fields"
    )


def metadata_from_particle_rows(particles: np.ndarray, *, default_psize_a: float) -> dict[int, MicrographMetadata]:
    representatives: dict[int, MicrographMetadata] = {}
    names = particles.dtype.names or ()
    for particle in particles:
        uid = int(particle["location/micrograph_uid"])
        representatives.setdefault(
            uid,
            MicrographMetadata(
                uid=uid,
                path=particle["location/micrograph_path"],
                shape=particle["location/micrograph_shape"],
                exp_group_id=int(particle["location/exp_group_id"]) if "location/exp_group_id" in names else 0,
                psize_a=float(particle["location/micrograph_psize_A"])
                if "location/micrograph_psize_A" in names
                else default_psize_a,
                path_dtype=particles.dtype["location/micrograph_path"],
                shape_dtype=particles.dtype["location/micrograph_shape"],
            ),
        )
    return dict(sorted(representatives.items()))


def metadata_from_exposure_rows(exposures: np.ndarray, *, default_psize_a: float) -> dict[int, MicrographMetadata]:
    names = exposures.dtype.names or ()
    path_field = first_existing_field(names, EXPOSURE_PATH_FIELDS, "micrograph path")
    shape_field = first_existing_field(names, EXPOSURE_SHAPE_FIELDS, "micrograph shape")
    psize_field = first_optional_field(names, EXPOSURE_PSIZE_FIELDS)
    exp_group_field = first_optional_field(names, EXPOSURE_EXP_GROUP_FIELDS)
    representatives: dict[int, MicrographMetadata] = {}
    for exposure in exposures:
        uid = int(exposure["uid"])
        representatives.setdefault(
            uid,
            MicrographMetadata(
                uid=uid,
                path=exposure[path_field],
                shape=exposure[shape_field],
                exp_group_id=int(exposure[exp_group_field]) if exp_group_field is not None else 0,
                psize_a=float(exposure[psize_field]) if psize_field is not None else default_psize_a,
                path_dtype=exposures.dtype[path_field],
                shape_dtype=exposures.dtype[shape_field],
            ),
        )
    return dict(sorted(representatives.items()))


def first_existing_field(names: tuple[str, ...], candidates: tuple[str, ...], label: str) -> str:
    field = first_optional_field(names, candidates)
    if field is None:
        choices = ", ".join(candidates)
        raise ValueError(f"Exposure metadata is missing {label}; expected one of: {choices}")
    return field


def first_optional_field(names: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    for field in candidates:
        if field in names:
            return field
    return None


def build_output_particles(metadata_rows: list[MicrographMetadata], rows: list[dict[str, Any]]) -> np.ndarray:
    dtype = output_dtype(metadata_rows)
    output = np.zeros(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        output["uid"][index] = index
        output["location/micrograph_uid"][index] = metadata.uid
        output["location/exp_group_id"][index] = metadata.exp_group_id
        output["location/micrograph_path"][index] = metadata.path
        output["location/micrograph_shape"][index] = metadata.shape
        shape = metadata.shape
        output["location/center_x_frac"][index] = float(row["x_full"]) / float(shape[1])
        output["location/center_y_frac"][index] = float(row["y_full"]) / float(shape[0])
        output["location/micrograph_psize_A"][index] = metadata.psize_a
    return output


def output_dtype(metadata_rows: list[MicrographMetadata]) -> np.dtype:
    if metadata_rows:
        path_dtype = metadata_rows[0].path_dtype
        shape_dtype = metadata_rows[0].shape_dtype
    else:
        path_dtype = np.dtype("S1024")
        shape_dtype = np.dtype(("<u4", (2,)))
    fields: list[tuple] = [("uid", "<u8")]
    fields.append(("location/micrograph_uid", "<u8"))
    fields.append(("location/exp_group_id", "<u4"))
    fields.append(("location/micrograph_path", path_dtype))
    fields.append(("location/micrograph_shape", shape_dtype))
    fields.append(("location/center_x_frac", "<f4"))
    fields.append(("location/center_y_frac", "<f4"))
    fields.append(("location/micrograph_psize_A", "<f4"))
    return np.dtype(fields)


def micrograph_summary(metadata: MicrographMetadata, *, status: str, pick_count: int) -> dict[str, Any]:
    return {
        "micrograph_uid": metadata.uid,
        "micrograph_path": decode_path(metadata.path),
        "status": status,
        "pick_count": pick_count,
    }


def write_pick_summary(
    path: Path,
    *,
    metadata_cs: Path,
    metadata_csg: Path,
    metadata_kind: str,
    vesicle_mask_dir: Path,
    outputs: VesicleMembranePickOutputs,
    parameters: dict[str, Any],
    micrographs: list[dict[str, Any]],
    total_picks: int,
) -> None:
    missing_mask_count = sum(1 for row in micrographs if row["status"] == "missing_mask")
    payload = {
        "inputs": {
            "metadata_cs": str(metadata_cs),
            "metadata_csg": str(metadata_csg),
            "metadata_kind": metadata_kind,
            "vesicle_mask_dir": str(vesicle_mask_dir),
        },
        "outputs": {
            "vesicle_membrane_picks_cs": str(outputs.picks_cs),
            "vesicle_membrane_picks_csg": str(outputs.picks_csg),
        },
        "parameters": parameters,
        "total_picks": total_picks,
        "missing_mask_count": missing_mask_count,
        "skipped_micrograph_count": missing_mask_count,
        "micrographs": micrographs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
