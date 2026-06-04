#!/usr/bin/env python3
"""Render standalone membrane-band mask previews from Vesicle Picker outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation_integration.cryosparc_io import decode_path, load_particles_cs
from segmentation_integration.dataset_presets import dataset_choices, resolve_particle_cs_path
from segmentation_integration.vesicle_masks import (
    build_membrane_band,
    load_vesicle_masks,
    mask_path_for_uid,
    membrane_radius_px,
)
from segmentation_integration.visualization import render_mask_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), help="Use standard Prestin paths for raw or draco.")
    parser.add_argument("--particle-export-dir", type=Path, help="Override the preset particle export directory.")
    parser.add_argument("--particles-cs", type=Path, help="Override the preset particle .cs file.")
    parser.add_argument("--vesicle-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psize-a", type=float, default=1.068)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--membrane-band-a", type=float, default=40.0)
    parser.add_argument(
        "--max-masks",
        type=int,
        default=0,
        help="Maximum mask previews to render; 0 means all micrographs with masks.",
    )
    args = parser.parse_args()
    if args.dataset is None and args.particles_cs is None:
        parser.error("--dataset is required unless --particles-cs is provided")
    return args


def render_mask_previews(
    *,
    particles_cs: Path,
    vesicle_mask_dir: Path,
    output_dir: Path,
    psize_a: float,
    downsample: int,
    membrane_band_a: float,
    max_masks: int,
) -> dict[str, int]:
    """Render one standalone membrane-band PNG per UID available in particles."""
    particles = load_particles_cs(particles_cs)
    radius_px = membrane_radius_px(membrane_band_a, psize_a, downsample)
    uid_to_path: dict[int, str] = {}
    for particle in particles:
        uid_to_path.setdefault(
            int(particle["location/micrograph_uid"]),
            decode_path(particle["location/micrograph_path"]),
        )

    rendered = 0
    missing = 0
    for uid, micrograph_path in sorted(uid_to_path.items()):
        if max_masks > 0 and rendered >= max_masks:
            break
        mask_path = mask_path_for_uid(vesicle_mask_dir, uid)
        if not mask_path.exists():
            missing += 1
            continue
        masks = load_vesicle_masks(mask_path)
        if not masks:
            missing += 1
            continue
        band = build_membrane_band(masks, radius_px)
        stem = Path(micrograph_path).stem
        render_mask_png(
            output_path=output_dir / f"{stem}_membrane_band_mask.png",
            membrane_band=band,
            title=f"{stem} membrane band",
        )
        rendered += 1
    return {"rendered_masks": rendered, "missing_masks": missing}


def main() -> None:
    args = parse_args()
    particles_cs = args.particles_cs
    if args.dataset is not None:
        particles_cs = resolve_particle_cs_path(
            args.dataset,
            particle_export_dir=args.particle_export_dir,
            particles_cs=args.particles_cs,
        )
    summary = render_mask_previews(
        particles_cs=particles_cs,
        vesicle_mask_dir=args.vesicle_mask_dir,
        output_dir=args.output_dir,
        psize_a=args.psize_a,
        downsample=args.downsample,
        membrane_band_a=args.membrane_band_a,
        max_masks=args.max_masks,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
