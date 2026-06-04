#!/usr/bin/env python3
"""Render overlays for original, filtered, and rejected CryoSPARC particle sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from segmentation_integration.cryosparc_io import build_micrograph_index, load_particles_cs, resolve_micrograph_path
from segmentation_integration.filtering import filter_particles_by_masks, particle_fullres_xy
from segmentation_integration.visualization import render_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-cs", type=Path, required=True)
    parser.add_argument("--filtered-particles-cs", type=Path, required=True)
    parser.add_argument("--rejected-particles-cs", type=Path, required=True)
    parser.add_argument("--vesicle-mask-dir", type=Path, required=True)
    parser.add_argument("--micrographs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psize-a", type=float, default=1.068)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--membrane-band-a", type=float, default=40.0)
    parser.add_argument("--max-overlays", type=int, default=10)
    parser.add_argument("--display-downsample", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    particles = load_particles_cs(args.particles_cs)
    filtered_particles = load_particles_cs(args.filtered_particles_cs)
    rejected_particles = load_particles_cs(args.rejected_particles_cs)
    result = filter_particles_by_masks(
        particles,
        vesicle_mask_dir=args.vesicle_mask_dir,
        psize_a=args.psize_a,
        downsample=args.downsample,
        membrane_band_a=args.membrane_band_a,
    )
    micrograph_index = build_micrograph_index(args.micrographs_dir)
    rendered = 0
    for row in result.micrograph_rows:
        if rendered >= args.max_overlays:
            break
        uid = int(row["micrograph_uid"])
        band = result.membrane_bands.get(uid)
        if band is None:
            continue
        micrograph_path = resolve_micrograph_path(str(row["micrograph_path"]), args.micrographs_dir, micrograph_index)
        render_overlay(
            micrograph_path=micrograph_path,
            output_path=args.output_dir / f"{micrograph_path.stem}_vesicle_filter_overlay.png",
            membrane_band=band,
            downsample=args.downsample,
            original_xy=coords_for_uid(particles, uid),
            kept_xy=coords_for_uid(filtered_particles, uid),
            rejected_xy=coords_for_uid(rejected_particles, uid),
            display_downsample=args.display_downsample,
        )
        rendered += 1
    print(json.dumps({"rendered_overlays": rendered, "output_dir": str(args.output_dir)}, indent=2))


def coords_for_uid(particles: np.ndarray, uid: int) -> np.ndarray:
    rows = [particle_fullres_xy(row) for row in particles if int(row["location/micrograph_uid"]) == uid]
    return np.asarray(rows, dtype=float)


if __name__ == "__main__":
    main()
