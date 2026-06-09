"""Particle filtering against Vesicle Picker membrane bands."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from segmentation_integration.cryosparc_io import decode_path
from segmentation_integration.vesicle_masks import (
    build_membrane_band,
    load_vesicle_masks,
    mask_path_for_uid,
    membrane_radius_px,
)


@dataclass(frozen=True)
class FilterResult:
    kept: np.ndarray
    rejected: np.ndarray
    particle_rows: list[dict[str, object]]
    micrograph_rows: list[dict[str, object]]
    membrane_bands: dict[int, np.ndarray]
    reason_counts: Counter


def particle_fullres_xy(particle: np.void) -> tuple[float, float]:
    """Return full-resolution particle center coordinates as `(x, y)`."""
    shape = particle["location/micrograph_shape"]
    ny = int(shape[0])
    nx = int(shape[1])
    return (
        float(particle["location/center_x_frac"]) * nx,
        float(particle["location/center_y_frac"]) * ny,
    )


def particle_mask_coordinate(particle: np.void, downsample: int) -> tuple[int, int]:
    """Map a CryoSPARC particle row to `(y, x)` coordinates on a mask grid."""
    x_full, y_full = particle_fullres_xy(particle)
    return int(np.round(y_full / downsample)), int(np.round(x_full / downsample))


def filter_particles_by_masks(
    particles: np.ndarray,
    *,
    vesicle_mask_dir: Path,
    psize_a: float,
    downsample: int,
    membrane_band_a: float,
    jobs: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FilterResult:
    """Split particles into kept/rejected arrays using Vesicle Picker masks."""
    if jobs <= 0:
        raise ValueError("--jobs must be positive")
    radius_px = membrane_radius_px(membrane_band_a, psize_a, downsample)
    micrograph_indices: dict[int, list[int]] = {}
    for index, particle in enumerate(particles):
        uid = int(particle["location/micrograph_uid"])
        micrograph_indices.setdefault(uid, []).append(index)
    keep_array = np.zeros(len(particles), dtype=bool)
    particle_rows: list[dict[str, object]] = []
    bands: dict[int, np.ndarray] = {}
    reason_counts: Counter = Counter()
    groups = [
        (uid, indices, particles[indices], vesicle_mask_dir, radius_px, downsample)
        for uid, indices in micrograph_indices.items()
    ]
    total_micrographs = len(groups)

    completed = 0
    for group_result in iter_micrograph_filter_results(groups, jobs=jobs):
        uid, group_keep_flags, group_rows, band, group_reason_counts = group_result
        indices = micrograph_indices[uid]
        keep_array[indices] = group_keep_flags
        particle_rows.extend(group_rows)
        if band is not None:
            bands[uid] = band
        reason_counts.update(group_reason_counts)
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total_micrographs)

    particle_rows.sort(key=lambda row: int(row["row_index"]))
    micrograph_rows = summarize_by_micrograph(particle_rows)
    return FilterResult(
        kept=particles[keep_array],
        rejected=particles[~keep_array],
        particle_rows=particle_rows,
        micrograph_rows=micrograph_rows,
        membrane_bands=bands,
        reason_counts=reason_counts,
    )


def iter_micrograph_filter_results(
    groups: list[tuple[int, list[int], np.ndarray, Path, int, int]],
    *,
    jobs: int,
):
    if jobs == 1 or len(groups) <= 1:
        for group in groups:
            yield filter_micrograph_group(group)
        return

    pending_limit = max(1, jobs * 2)
    group_iter = iter(groups)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        pending = set()
        for _ in range(min(pending_limit, len(groups))):
            pending.add(executor.submit(filter_micrograph_group, next(group_iter)))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    group = next(group_iter)
                except StopIteration:
                    continue
                pending.add(executor.submit(filter_micrograph_group, group))


def filter_micrograph_group(group: tuple[int, list[int], np.ndarray, Path, int, int]):
    uid, indices, particles, vesicle_mask_dir, radius_px, downsample = group
    band = None
    missing_mask = False
    mask_path = mask_path_for_uid(vesicle_mask_dir, uid)
    if mask_path.exists():
        masks = load_vesicle_masks(mask_path)
        band = build_membrane_band(masks, radius_px) if masks else np.zeros((0, 0), dtype=bool)
    else:
        missing_mask = True

    keep_flags = np.zeros(len(particles), dtype=bool)
    particle_rows = []
    reason_counts: Counter = Counter()
    for local_index, (row_index, particle) in enumerate(zip(indices, particles)):
        x_full, y_full = particle_fullres_xy(particle)
        y_mask, x_mask = particle_mask_coordinate(particle, downsample)
        reason = "kept"
        keep = False
        if missing_mask:
            reason = "missing_mask"
        else:
            if band is None:
                raise RuntimeError("internal error: band is missing for non-missing mask")
            shape = particle["location/micrograph_shape"]
            ny = int(shape[0])
            nx = int(shape[1])
            if band.size == 0:
                reason = "empty_mask"
            elif not (0 <= x_full < nx and 0 <= y_full < ny):
                reason = "out_of_bounds"
            elif not (0 <= y_mask < band.shape[0] and 0 <= x_mask < band.shape[1]):
                reason = "out_of_bounds"
            elif bool(band[y_mask, x_mask]):
                keep = True
            else:
                reason = "outside_membrane_band"

        keep_flags[local_index] = keep
        reason_counts[reason] += 1
        particle_rows.append(
            {
                "row_index": row_index,
                "micrograph_uid": uid,
                "micrograph_path": decode_path(particle["location/micrograph_path"]),
                "x_full": x_full,
                "y_full": y_full,
                "x_mask": x_mask,
                "y_mask": y_mask,
                "status": "kept" if keep else "rejected",
                "reason": reason,
            }
        )
    return uid, keep_flags, particle_rows, band, reason_counts


def summarize_by_micrograph(particle_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, Counter] = {}
    paths: dict[int, str] = {}
    for row in particle_rows:
        uid = int(row["micrograph_uid"])
        grouped.setdefault(uid, Counter())[str(row["reason"])] += 1
        paths.setdefault(uid, str(row["micrograph_path"]))
    output = []
    for uid, counts in sorted(grouped.items()):
        total = sum(counts.values())
        kept = counts.get("kept", 0)
        output.append(
            {
                "micrograph_uid": uid,
                "micrograph_path": paths[uid],
                "total_particles": total,
                "kept_particles": kept,
                "rejected_particles": total - kept,
                "missing_mask": counts.get("missing_mask", 0),
                "empty_mask": counts.get("empty_mask", 0),
                "out_of_bounds": counts.get("out_of_bounds", 0),
                "outside_membrane_band": counts.get("outside_membrane_band", 0),
            }
        )
    return output
