#!/usr/bin/env python3
"""Filter CryoSPARC particle locations by Vesicle Picker membrane masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from segmentation_integration.cryosparc_io import (
    build_micrograph_index,
    decode_path,
    load_particles_cs,
    patch_csg_cs_path,
    resolve_micrograph_path,
    write_cs,
)
from segmentation_integration.dataset_presets import dataset_choices, resolve_standard_paths
from segmentation_integration.filtering import filter_particles_by_masks, particle_fullres_xy
from segmentation_integration.reporting import write_summary_json, write_tsv
from segmentation_integration.visualization import render_mask_png, render_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), help="Use standard Prestin paths for raw or draco.")
    parser.add_argument("--particle-export-dir", type=Path, help="Override the preset particle export directory.")
    parser.add_argument("--particles-cs", type=Path, help="Override the preset particle .cs file.")
    parser.add_argument("--particles-csg", type=Path, help="Override the preset particle .csg file.")
    parser.add_argument("--vesicle-mask-dir", type=Path, required=True)
    parser.add_argument("--micrographs-dir", type=Path, help="Override the preset micrograph directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psize-a", type=float, default=1.068)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--membrane-band-a", type=float, default=40.0)
    parser.add_argument("--max-overlays", type=int, default=10)
    parser.add_argument("--display-downsample", type=int, default=4)
    parser.add_argument("--max-picks-per-class", type=int, default=500)
    parser.add_argument("--contrast-percentiles", default="1,99")
    parser.add_argument(
        "--skip-mask-pngs",
        action="store_true",
        help="Do not render standalone membrane-band mask PNGs during filtering.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write progress every N micrographs; 0 disables progress lines.",
    )
    args = parser.parse_args()
    if args.dataset is None:
        missing = [name for name in ("particles_cs", "particles_csg", "micrographs_dir") if getattr(args, name) is None]
        if missing:
            parser.error("--dataset is required unless --particles-cs, --particles-csg, and --micrographs-dir are all provided")
    return args


def parse_percentiles(text: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Expected low,high percentiles such as 1,99")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    if args.dataset is not None:
        paths = resolve_standard_paths(
            args.dataset,
            particle_export_dir=args.particle_export_dir,
            particles_cs=args.particles_cs,
            particles_csg=args.particles_csg,
            micrographs_dir=args.micrographs_dir,
        )
        args.particles_cs = paths.particles_cs
        args.particles_csg = paths.particles_csg
        args.micrographs_dir = paths.micrographs_dir
    low_high = parse_percentiles(args.contrast_percentiles)
    particles = load_particles_cs(args.particles_cs)
    result = filter_particles_by_masks(
        particles,
        vesicle_mask_dir=args.vesicle_mask_dir,
        psize_a=args.psize_a,
        downsample=args.downsample,
        membrane_band_a=args.membrane_band_a,
        progress_every=args.progress_every,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_cs = args.output_dir / "filtered_particles.cs"
    rejected_cs = args.output_dir / "rejected_particles.cs"
    filtered_csg = args.output_dir / "filtered_particles.csg"
    rejected_csg = args.output_dir / "rejected_particles.csg"
    write_cs(filtered_cs, result.kept)
    write_cs(rejected_cs, result.rejected)
    patch_csg_cs_path(args.particles_csg, filtered_csg, args.particles_cs.name, csg_cs_reference_path(filtered_cs, args.particles_csg))
    patch_csg_cs_path(args.particles_csg, rejected_csg, args.particles_cs.name, csg_cs_reference_path(rejected_cs, args.particles_csg))
    write_tsv(args.output_dir / "filter_summary.tsv", result.micrograph_rows)
    write_summary_json(
        args.output_dir / "filter_summary.json",
        inputs={
            "particles_cs": str(args.particles_cs),
            "particles_csg": str(args.particles_csg),
            "vesicle_mask_dir": str(args.vesicle_mask_dir),
            "micrographs_dir": str(args.micrographs_dir),
            "psize_a": args.psize_a,
            "downsample": args.downsample,
            "membrane_band_a": args.membrane_band_a,
        },
        outputs={
            "filtered_particles_cs": str(filtered_cs),
            "filtered_particles_csg": str(filtered_csg),
            "rejected_particles_cs": str(rejected_cs),
            "rejected_particles_csg": str(rejected_csg),
        },
        reason_counts=result.reason_counts,
        micrographs=result.micrograph_rows,
    )

    if args.max_overlays > 0:
        render_overlays(args, particles, result, low_high)

    print(f"Kept {len(result.kept)} particles; rejected {len(result.rejected)} particles.")
    print(f"Import filtered result group in CryoSPARC with: {filtered_csg}")


def csg_cs_reference_path(cs_path: Path, particles_csg: Path) -> Path:
    """Return the `.cs` reference path CryoSPARC should see inside an imported `.csg`."""
    project_dir = cryosparc_project_dir_from_export_path(particles_csg)
    if project_dir is None:
        return cs_path
    try:
        return cs_path.relative_to(project_dir)
    except ValueError:
        return cs_path


def cryosparc_project_dir_from_export_path(path: Path) -> Path | None:
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("exports", "groups"):
            return Path(*parts[:index])
    return None


def render_overlays(args: argparse.Namespace, particles: np.ndarray, result, low_high: tuple[float, float]) -> None:
    micrograph_index = build_micrograph_index(args.micrographs_dir)
    rows_by_uid: dict[int, list[dict[str, object]]] = {}
    for row in result.particle_rows:
        rows_by_uid.setdefault(int(row["micrograph_uid"]), []).append(row)
    written = 0
    overlays_dir = args.output_dir / "overlays"
    masks_dir = args.output_dir / "masks"
    for uid in sorted(rows_by_uid):
        if written >= args.max_overlays:
            break
        rows = rows_by_uid[uid]
        band = result.membrane_bands.get(uid)
        if band is None:
            continue
        stem = Path(decode_path(rows[0]["micrograph_path"])).stem
        if not args.skip_mask_pngs:
            render_mask_png(
                output_path=masks_dir / f"{stem}_membrane_band_mask.png",
                membrane_band=band,
                title=f"{stem} membrane band",
            )
        micrograph_path = resolve_micrograph_path(
            str(rows[0]["micrograph_path"]),
            args.micrographs_dir,
            micrograph_index,
        )
        particle_indices = [int(row["row_index"]) for row in rows]
        original_xy = np.asarray([particle_fullres_xy(particles[index]) for index in particle_indices], dtype=float)
        kept_indices = [int(row["row_index"]) for row in rows if row["status"] == "kept"]
        rejected_indices = [int(row["row_index"]) for row in rows if row["status"] == "rejected"]
        kept_xy = np.asarray([particle_fullres_xy(particles[index]) for index in kept_indices], dtype=float)
        rejected_xy = np.asarray([particle_fullres_xy(particles[index]) for index in rejected_indices], dtype=float)
        render_overlay(
            micrograph_path=micrograph_path,
            output_path=overlays_dir / f"{stem}_vesicle_filter_overlay.png",
            membrane_band=band,
            downsample=args.downsample,
            original_xy=original_xy,
            kept_xy=kept_xy,
            rejected_xy=rejected_xy,
            contrast_percentiles=low_high,
            display_downsample=args.display_downsample,
            max_picks_per_class=args.max_picks_per_class,
        )
        written += 1


if __name__ == "__main__":
    main()
