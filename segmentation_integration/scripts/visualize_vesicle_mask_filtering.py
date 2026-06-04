#!/usr/bin/env python3
"""Render upstream Vesicle Picker QC: found SAM masks vs filtered masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from segmentation_integration.vesicle_masks import load_vesicle_masks
from segmentation_integration.visualization import render_mask_comparison_png


STATS_FIELDS = ("area", "roundness", "average_radius_A", "semi_minor_A", "semi_major_A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--found-mask-dir", type=Path, required=True)
    parser.add_argument("--filtered-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-micrographs", type=int, default=0, help="Maximum UIDs to render; 0 means all.")
    return parser.parse_args()


def render_mask_filter_qc(
    *,
    found_mask_dir: Path,
    filtered_mask_dir: Path,
    output_dir: Path,
    max_micrographs: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    rendered = 0
    missing_filtered = 0
    for found_path in sorted(found_mask_dir.glob("*_vesicles.pkl")):
        if max_micrographs > 0 and rendered >= max_micrographs:
            break
        uid = uid_from_found_mask_path(found_path)
        filtered_path = filtered_mask_dir / f"{uid}_vesicles_filtered.pkl"
        found_masks = load_vesicle_masks(found_path)
        filtered_masks = load_vesicle_masks(filtered_path) if filtered_path.exists() else []
        if not filtered_path.exists():
            missing_filtered += 1
        stats_rows.extend(mask_stats_rows(uid, "found", found_masks))
        stats_rows.extend(mask_stats_rows(uid, "filtered", filtered_masks))
        if not found_masks:
            continue
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
    write_stats_tsv(output_dir / "mask_filter_stats.tsv", stats_rows)
    return {
        "rendered_comparisons": rendered,
        "missing_filtered_masks": missing_filtered,
        "stats_rows": len(stats_rows),
    }


def uid_from_found_mask_path(path: Path) -> int:
    return int(path.name.removesuffix("_vesicles.pkl"))


def union_mask(masks: list[dict[str, Any]], shape: tuple[int, int] | None = None) -> np.ndarray:
    if not masks:
        if shape is None:
            raise ValueError("Cannot infer union mask shape from empty masks")
        return np.zeros(shape, dtype=bool)
    out = np.zeros(np.asarray(masks[0]["segmentation"]).shape, dtype=bool)
    for mask in masks:
        out |= np.asarray(mask["segmentation"], dtype=bool)
    return out


def mask_stats_rows(uid: int, stage: str, masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, mask in enumerate(masks):
        row = {"uid": uid, "stage": stage, "mask_index": index}
        for field in STATS_FIELDS:
            row[field] = mask.get(field, "")
        rows.append(row)
    return rows


def write_stats_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("uid", "stage", "mask_index", *STATS_FIELDS)
    with path.open("w") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")


def main() -> None:
    args = parse_args()
    summary = render_mask_filter_qc(
        found_mask_dir=args.found_mask_dir,
        filtered_mask_dir=args.filtered_mask_dir,
        output_dir=args.output_dir,
        max_micrographs=args.max_micrographs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
