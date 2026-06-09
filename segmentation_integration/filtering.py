"""Particle filtering against Vesicle Picker membrane bands."""

from __future__ import annotations

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
    progress_callback: Callable[[int, int], None] | None = None,
) -> FilterResult:
    """Split particles into kept/rejected arrays using Vesicle Picker masks."""
    radius_px = membrane_radius_px(membrane_band_a, psize_a, downsample)
    bands: dict[int, np.ndarray] = {}
    missing_uids: set[int] = set()
    micrograph_indices: dict[int, list[int]] = {}
    for index, particle in enumerate(particles):
        uid = int(particle["location/micrograph_uid"])
        micrograph_indices.setdefault(uid, []).append(index)
    keep_array = np.zeros(len(particles), dtype=bool)
    particle_rows_by_index: list[dict[str, object] | None] = [None] * len(particles)
    reason_counts: Counter = Counter()
    total_micrographs = len(micrograph_indices)

    for completed, (uid, indices) in enumerate(micrograph_indices.items(), start=1):
        if uid not in bands and uid not in missing_uids:
            mask_path = mask_path_for_uid(vesicle_mask_dir, uid)
            if mask_path.exists():
                masks = load_vesicle_masks(mask_path)
                bands[uid] = build_membrane_band(masks, radius_px) if masks else np.zeros((0, 0), dtype=bool)
            else:
                missing_uids.add(uid)

        for index in indices:
            particle = particles[index]
            x_full, y_full = particle_fullres_xy(particle)
            y_mask, x_mask = particle_mask_coordinate(particle, downsample)
            reason = "kept"
            keep = False
            if uid in missing_uids:
                reason = "missing_mask"
            else:
                band = bands[uid]
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

            keep_array[index] = keep
            reason_counts[reason] += 1
            particle_rows_by_index[index] = {
                "row_index": index,
                "micrograph_uid": uid,
                "micrograph_path": decode_path(particle["location/micrograph_path"]),
                "x_full": x_full,
                "y_full": y_full,
                "x_mask": x_mask,
                "y_mask": y_mask,
                "status": "kept" if keep else "rejected",
                "reason": reason,
            }
        if progress_callback is not None:
            progress_callback(completed, total_micrographs)

    particle_rows = [row for row in particle_rows_by_index if row is not None]
    micrograph_rows = summarize_by_micrograph(particle_rows)
    return FilterResult(
        kept=particles[keep_array],
        rejected=particles[~keep_array],
        particle_rows=particle_rows,
        micrograph_rows=micrograph_rows,
        membrane_bands=bands,
        reason_counts=reason_counts,
    )


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
