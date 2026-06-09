"""Generate offline CryoSPARC particle-location picks from vesicle masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from segmentation_integration.cryosparc_io import (
    decode_path,
    load_particles_cs,
    patch_csg_cs_path,
    write_cs,
)
from segmentation_integration.vesicle_masks import load_vesicle_masks, mask_path_for_uid


@dataclass(frozen=True)
class VesicleMembranePickOutputs:
    picks_cs: Path
    picks_csg: Path
    summary_json: Path


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
    particles_cs: Path,
    particles_csg: Path,
    vesicle_mask_dir: Path,
    output_dir: Path,
    psize_a: float = 1.068,
    downsample: int = 4,
    box_size_a: float = 100.0,
    dilation_radius_a: float = 0.0,
    mode: str = "edge",
) -> VesicleMembranePickOutputs:
    """Write offline vesicle membrane picks as CryoSPARC-compatible files."""
    validate_pick_parameters(psize_a=psize_a, downsample=downsample, box_size_a=box_size_a, mode=mode)
    source_particles = load_particles_cs(particles_cs)
    representatives = representative_rows_by_micrograph(source_particles)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    micrograph_summaries = []
    for uid, source_row in representatives.items():
        mask_path = mask_path_for_uid(vesicle_mask_dir, uid)
        if not mask_path.exists():
            micrograph_summaries.append(
                micrograph_summary(source_row, micrograph_uid=uid, status="missing_mask", pick_count=0)
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
            rows.append({"source": source_row, "y_full": int(y_full), "x_full": int(x_full)})
        micrograph_summaries.append(
            micrograph_summary(source_row, micrograph_uid=uid, status="picked", pick_count=len(pick_indices))
        )

    picks_cs = output_dir / "vesicle_membrane_picks.cs"
    picks_csg = output_dir / "vesicle_membrane_picks.csg"
    summary_json = output_dir / "vesicle_membrane_picks_summary.json"
    picks = build_output_particles(source_particles, rows)
    write_cs(picks_cs, picks)
    patch_csg_cs_path(particles_csg, picks_csg, particles_cs.name, picks_cs)
    write_pick_summary(
        summary_json,
        particles_cs=particles_cs,
        particles_csg=particles_csg,
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


def representative_rows_by_micrograph(particles: np.ndarray) -> dict[int, np.void]:
    representatives: dict[int, np.void] = {}
    for particle in particles:
        uid = int(particle["location/micrograph_uid"])
        representatives.setdefault(uid, particle)
    return dict(sorted(representatives.items()))


def build_output_particles(source_particles: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    dtype = output_dtype(source_particles.dtype)
    output = np.zeros(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        source = row["source"]
        output["uid"][index] = index
        output["location/micrograph_uid"][index] = source["location/micrograph_uid"]
        if "location/exp_group_id" in output.dtype.names:
            output["location/exp_group_id"][index] = source["location/exp_group_id"] if "location/exp_group_id" in source.dtype.names else 0
        output["location/micrograph_path"][index] = source["location/micrograph_path"]
        output["location/micrograph_shape"][index] = source["location/micrograph_shape"]
        shape = source["location/micrograph_shape"]
        output["location/center_x_frac"][index] = float(row["x_full"]) / float(shape[1])
        output["location/center_y_frac"][index] = float(row["y_full"]) / float(shape[0])
        if "location/micrograph_psize_A" in output.dtype.names:
            output["location/micrograph_psize_A"][index] = (
                source["location/micrograph_psize_A"] if "location/micrograph_psize_A" in source.dtype.names else 0
            )
    return output


def output_dtype(source_dtype: np.dtype) -> np.dtype:
    names = source_dtype.names or ()
    fields: list[tuple] = [("uid", "<u8")]
    fields.append(("location/micrograph_uid", source_dtype["location/micrograph_uid"]))
    fields.append(
        (
            "location/exp_group_id",
            source_dtype["location/exp_group_id"] if "location/exp_group_id" in names else np.dtype("<u4"),
        )
    )
    fields.append(("location/micrograph_path", source_dtype["location/micrograph_path"]))
    fields.append(("location/micrograph_shape", source_dtype["location/micrograph_shape"]))
    fields.append(("location/center_x_frac", "<f4"))
    fields.append(("location/center_y_frac", "<f4"))
    fields.append(
        (
            "location/micrograph_psize_A",
            source_dtype["location/micrograph_psize_A"] if "location/micrograph_psize_A" in names else np.dtype("<f4"),
        )
    )
    return np.dtype(fields)


def micrograph_summary(source_row: np.void, *, micrograph_uid: int, status: str, pick_count: int) -> dict[str, Any]:
    return {
        "micrograph_uid": micrograph_uid,
        "micrograph_path": decode_path(source_row["location/micrograph_path"]),
        "status": status,
        "pick_count": pick_count,
    }


def write_pick_summary(
    path: Path,
    *,
    particles_cs: Path,
    particles_csg: Path,
    vesicle_mask_dir: Path,
    outputs: VesicleMembranePickOutputs,
    parameters: dict[str, Any],
    micrographs: list[dict[str, Any]],
    total_picks: int,
) -> None:
    missing_mask_count = sum(1 for row in micrographs if row["status"] == "missing_mask")
    payload = {
        "inputs": {
            "particles_cs": str(particles_cs),
            "particles_csg": str(particles_csg),
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
