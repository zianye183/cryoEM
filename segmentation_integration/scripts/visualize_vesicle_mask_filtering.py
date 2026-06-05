#!/usr/bin/env python3
"""Render upstream Vesicle Picker QC: found SAM masks vs filtered masks."""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path
from time import monotonic
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage

from segmentation_integration.cryosparc_io import (
    build_micrograph_index,
    decode_path,
    load_particles_cs,
    resolve_micrograph_path,
)
from segmentation_integration.mrc_io import read_mode2_mrc
from segmentation_integration.vesicle_masks import load_vesicle_masks
from segmentation_integration.visualization import (
    render_mask_comparison_png,
    render_mask_filter_debug_png,
    render_mask_metrics_png,
)


STATS_FIELDS = (
    "area",
    "roundness",
    "average_radius_A",
    "semi_minor_A",
    "semi_major_A",
    "interior_mean",
    "membrane_mean",
    "local_bg_mean",
    "ring_contrast",
    "interior_contrast",
    "interior_std",
    "membrane_std",
    "local_bg_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--found-mask-dir", type=Path, required=True)
    parser.add_argument("--filtered-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles-cs", type=Path, help="Optional particle .cs for resolving micrograph underlays.")
    parser.add_argument("--micrographs-dir", type=Path, help="Optional micrograph directory for overlay renderings.")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--display-downsample", type=int, default=4)
    parser.add_argument("--contrast-percentiles", default="1,99")
    parser.add_argument("--filter-parameters", type=Path, help="Optional filter_vesicles.ini for found-mask reject reasons.")
    parser.add_argument("--max-micrographs", type=int, default=0, help="Maximum UIDs to render; 0 means all.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write progress every N micrographs; 0 disables progress lines.",
    )
    args = parser.parse_args()
    if (args.particles_cs is None) != (args.micrographs_dir is None):
        parser.error("--particles-cs and --micrographs-dir must be provided together")
    return args


def render_mask_filter_qc(
    *,
    found_mask_dir: Path,
    filtered_mask_dir: Path,
    output_dir: Path,
    max_micrographs: int,
    particles_cs: Path | None = None,
    micrographs_dir: Path | None = None,
    downsample: int = 4,
    display_downsample: int = 4,
    contrast_percentiles: tuple[float, float] = (1, 99),
    filter_parameters: Path | None = None,
    progress_every: int = 1,
    progress_stream=sys.stderr,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    uid_to_micrograph = micrograph_paths_by_uid(particles_cs, micrographs_dir) if particles_cs and micrographs_dir else {}
    rules = load_filter_rules(filter_parameters) if filter_parameters else {}
    stats_rows = []
    rendered = 0
    missing_filtered = 0
    found_kept = 0
    found_rejected = 0
    found_paths = sorted(found_mask_dir.glob("*_vesicles.pkl"))
    if max_micrographs > 0:
        found_paths = found_paths[:max_micrographs]
    total = len(found_paths)
    for index, found_path in enumerate(found_paths, start=1):
        item_started_at = monotonic()
        uid = uid_from_found_mask_path(found_path)
        if should_emit_progress(index, progress_every):
            emit_progress(progress_stream, index=index, total=total, uid=uid, status="start")
        filtered_path = filtered_mask_dir / f"{uid}_vesicles_filtered.pkl"
        found_masks = load_vesicle_masks(found_path)
        filtered_masks = load_vesicle_masks(filtered_path) if filtered_path.exists() else []
        if not filtered_path.exists():
            missing_filtered += 1
        micrograph_path = uid_to_micrograph.get(uid)
        if micrograph_path is not None:
            add_density_metrics(found_masks, micrograph_path, downsample)
            add_density_metrics(filtered_masks, micrograph_path, downsample)
        classified = classify_masks(found_masks, rules)
        found_kept += sum(1 for item in classified if not item["reject_reasons"])
        found_rejected += sum(1 for item in classified if item["reject_reasons"])
        stats_rows.extend(mask_stats_rows(uid, "found", found_masks, classified))
        stats_rows.extend(mask_stats_rows(uid, "filtered", filtered_masks))
        if not found_masks:
            if should_emit_progress(index, progress_every):
                emit_progress(
                    progress_stream,
                    index=index,
                    total=total,
                    uid=uid,
                    status="empty",
                    found_masks=0,
                    filtered_masks=len(filtered_masks),
                    elapsed_seconds=monotonic() - item_started_at,
                )
            continue
        render_mask_filter_debug_png(
            micrograph_path=micrograph_path,
            output_path=output_dir / f"{uid}_found_filter_debug.png",
            found_kept_masks=[np.asarray(item["mask"]["segmentation"], dtype=bool) for item in classified if not item["reject_reasons"]],
            found_rejected_masks=[np.asarray(item["mask"]["segmentation"], dtype=bool) for item in classified if item["reject_reasons"]],
            filtered_masks=[np.asarray(mask["segmentation"], dtype=bool) for mask in filtered_masks],
            downsample=downsample,
            title=f"uid {uid}" + (f" {micrograph_path.name}" if micrograph_path is not None else ""),
            contrast_percentiles=contrast_percentiles,
            display_downsample=display_downsample,
        )
        render_mask_metrics_png(
            micrograph_path=micrograph_path,
            output_path=output_dir / f"{uid}_found_mask_metrics.png",
            masks=[item["mask"] for item in classified if not item["reject_reasons"]],
            rejected_masks=[item["mask"] for item in classified if item["reject_reasons"]],
            downsample=downsample,
            title=f"uid {uid}" + (f" {micrograph_path.name}" if micrograph_path is not None else ""),
            contrast_percentiles=contrast_percentiles,
            display_downsample=display_downsample,
        )
        if micrograph_path is None:
            found_union = union_mask(found_masks)
            filtered_union = union_mask(filtered_masks, shape=found_union.shape)
            render_mask_comparison_png(
                output_path=output_dir / f"{uid}_found_vs_filtered_masks.png",
                found_mask=found_union,
                filtered_mask=filtered_union,
                title=f"uid {uid}",
                found_count=len(found_masks),
                filtered_count=len(filtered_masks),
            )
        rendered += 1
        if should_emit_progress(index, progress_every):
            emit_progress(
                progress_stream,
                index=index,
                total=total,
                uid=uid,
                status="done",
                found_masks=len(found_masks),
                filtered_masks=len(filtered_masks),
                elapsed_seconds=monotonic() - item_started_at,
            )
    write_stats_tsv(output_dir / "mask_filter_stats.tsv", stats_rows)
    return {
        "rendered_comparisons": rendered,
        "missing_filtered_masks": missing_filtered,
        "stats_rows": len(stats_rows),
        "found_masks_kept": found_kept,
        "found_masks_rejected": found_rejected,
    }


def should_emit_progress(index: int, progress_every: int) -> bool:
    return progress_every > 0 and index % progress_every == 0


def emit_progress(
    stream,
    *,
    index: int,
    total: int,
    uid: int,
    status: str,
    found_masks: int | None = None,
    filtered_masks: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    parts = [
        "[mask filter qc]",
        f"{index}/{total}",
        f"uid={uid}",
        f"status={status}",
    ]
    if found_masks is not None:
        parts.append(f"found={found_masks}")
    if filtered_masks is not None:
        parts.append(f"filtered={filtered_masks}")
    if elapsed_seconds is not None:
        parts.append(f"elapsed={elapsed_seconds:.2f}s")
    print(" ".join(parts), file=stream, flush=True)


def uid_from_found_mask_path(path: Path) -> int:
    return int(path.name.removesuffix("_vesicles.pkl"))


def micrograph_paths_by_uid(particles_cs: Path, micrographs_dir: Path) -> dict[int, Path]:
    particles = load_particles_cs(particles_cs)
    index = build_micrograph_index(micrographs_dir)
    out: dict[int, Path] = {}
    for particle in particles:
        uid = int(particle["location/micrograph_uid"])
        if uid in out:
            continue
        out[uid] = resolve_micrograph_path(decode_path(particle["location/micrograph_path"]), micrographs_dir, index)
    return out


def union_mask(masks: list[dict[str, Any]], shape: tuple[int, int] | None = None) -> np.ndarray:
    if not masks:
        if shape is None:
            raise ValueError("Cannot infer union mask shape from empty masks")
        return np.zeros(shape, dtype=bool)
    out = np.zeros(np.asarray(masks[0]["segmentation"]).shape, dtype=bool)
    for mask in masks:
        out |= np.asarray(mask["segmentation"], dtype=bool)
    return out


def load_filter_rules(path: Path) -> dict[str, tuple[float | None, float | None]]:
    config = configparser.ConfigParser()
    config.read(path)
    rules = {}
    for section in config.sections():
        if not config.has_option(section, "min") and not config.has_option(section, "max"):
            continue
        min_value = config.getfloat(section, "min", fallback=None)
        max_value = config.getfloat(section, "max", fallback=None)
        rules[section] = (min_value, max_value)
    return rules


def classify_masks(masks: list[dict[str, Any]], rules: dict[str, tuple[float | None, float | None]]) -> list[dict[str, Any]]:
    classified = []
    for mask in masks:
        reject_reasons = []
        for field, (min_value, max_value) in rules.items():
            value = mask.get(field)
            if value in (None, ""):
                continue
            value = float(value)
            if min_value is not None and value < min_value:
                reject_reasons.append(f"{field}_min")
            if max_value is not None and value > max_value:
                reject_reasons.append(f"{field}_max")
        classified.append({"mask": mask, "reject_reasons": reject_reasons})
    return classified


def mask_stats_rows(
    uid: int,
    stage: str,
    masks: list[dict[str, Any]],
    classified: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, mask in enumerate(masks):
        reject_reasons = classified[index]["reject_reasons"] if classified is not None else []
        status = "rejected" if reject_reasons else "kept" if classified is not None else "output"
        row = {
            "uid": uid,
            "stage": stage,
            "mask_index": index,
            "status": status,
            "reject_reasons": ",".join(reject_reasons),
        }
        for field in STATS_FIELDS:
            row[field] = mask.get(field, "")
        rows.append(row)
    return rows


def add_density_metrics(masks: list[dict[str, Any]], micrograph_path: Path, downsample: int) -> None:
    if not masks:
        return
    image = normalize_micrograph(read_mode2_mrc(micrograph_path))
    first_shape = np.asarray(masks[0]["segmentation"], dtype=bool).shape
    grid = downsample_image_to_mask_grid(image, first_shape, downsample)
    for mask in masks:
        metrics = density_metrics_for_mask(np.asarray(mask["segmentation"], dtype=bool), grid)
        mask.update(metrics)


def normalize_micrograph(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Micrograph has no finite pixels")
    low, high = np.percentile(finite, (1, 99))
    scale = high - low
    if scale <= 0:
        scale = float(np.std(finite)) or 1.0
    return (image - float(np.median(finite))) / scale


def downsample_image_to_mask_grid(image: np.ndarray, shape: tuple[int, int], downsample: int) -> np.ndarray:
    if downsample <= 1:
        return image[: shape[0], : shape[1]]
    rows = min(shape[0], image.shape[0] // downsample)
    cols = min(shape[1], image.shape[1] // downsample)
    cropped = image[: rows * downsample, : cols * downsample]
    grid = cropped.reshape(rows, downsample, cols, downsample).mean(axis=(1, 3))
    if (rows, cols) == shape:
        return grid
    out = np.full(shape, np.nan, dtype=float)
    out[:rows, :cols] = grid
    return out


def density_metrics_for_mask(segmentation: np.ndarray, image: np.ndarray) -> dict[str, float | str]:
    if not segmentation.any():
        return empty_density_metrics()
    segmentation = segmentation & np.isfinite(image)
    if not segmentation.any():
        return empty_density_metrics()
    eroded = ndimage.binary_erosion(segmentation, iterations=2)
    interior = eroded if eroded.any() else segmentation
    membrane = segmentation & ~eroded
    if not membrane.any():
        membrane = segmentation
    near = ndimage.binary_dilation(segmentation, iterations=4)
    gap = ndimage.binary_dilation(segmentation, iterations=1)
    local_bg = near & ~gap & np.isfinite(image)
    if not local_bg.any():
        local_bg = ~segmentation & np.isfinite(image)
    if not local_bg.any():
        return empty_density_metrics()
    interior_values = image[interior]
    membrane_values = image[membrane]
    bg_values = image[local_bg]
    interior_mean = float(np.mean(interior_values))
    membrane_mean = float(np.mean(membrane_values))
    bg_mean = float(np.mean(bg_values))
    return {
        "interior_mean": interior_mean,
        "membrane_mean": membrane_mean,
        "local_bg_mean": bg_mean,
        "ring_contrast": abs(membrane_mean - bg_mean),
        "interior_contrast": abs(interior_mean - bg_mean),
        "interior_std": float(np.std(interior_values)),
        "membrane_std": float(np.std(membrane_values)),
        "local_bg_std": float(np.std(bg_values)),
    }


def empty_density_metrics() -> dict[str, str]:
    return {
        "interior_mean": "",
        "membrane_mean": "",
        "local_bg_mean": "",
        "ring_contrast": "",
        "interior_contrast": "",
        "interior_std": "",
        "membrane_std": "",
        "local_bg_std": "",
    }


def write_stats_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("uid", "stage", "mask_index", "status", "reject_reasons", *STATS_FIELDS)
    with path.open("w") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")


def main() -> None:
    args = parse_args()
    contrast = parse_percentiles(args.contrast_percentiles)
    summary = render_mask_filter_qc(
        found_mask_dir=args.found_mask_dir,
        filtered_mask_dir=args.filtered_mask_dir,
        output_dir=args.output_dir,
        max_micrographs=args.max_micrographs,
        particles_cs=args.particles_cs,
        micrographs_dir=args.micrographs_dir,
        downsample=args.downsample,
        display_downsample=args.display_downsample,
        contrast_percentiles=contrast,
        filter_parameters=args.filter_parameters,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


def parse_percentiles(text: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Expected low,high percentiles such as 1,99")
    return parts[0], parts[1]


if __name__ == "__main__":
    main()
