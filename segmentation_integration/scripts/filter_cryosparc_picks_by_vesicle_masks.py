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
    load_particles_cs_many,
    patch_csg_cs_path,
    resolve_micrograph_path,
    write_cs,
)
from segmentation_integration.dataset_presets import dataset_choices, find_unique_file, get_dataset_preset, resolve_standard_paths
from segmentation_integration.filtering import filter_particles_by_masks, particle_fullres_xy
from segmentation_integration.reporting import write_summary_json, write_tsv
from segmentation_integration.visualization import render_mask_png, render_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), help="Use standard Prestin paths for raw or draco.")
    parser.add_argument(
        "--particle-export-dir",
        type=Path,
        nargs="+",
        help="One or more CryoSPARC particle result-group directories. Each directory must contain exactly one .cs and one .csg file.",
    )
    parser.add_argument("--particles-cs", type=Path, nargs="+", help="One or more particle .cs files.")
    parser.add_argument("--particles-csg", type=Path, nargs="+", help="One or more particle .csg files matching --particles-cs.")
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
        "--no-progress",
        action="store_true",
        help="Disable the one-line micrograph progress bar.",
    )
    args = parser.parse_args()
    particle_source_supplied = args.dataset is not None or args.particle_export_dir is not None or args.particles_cs is not None
    if not particle_source_supplied:
        parser.error("--dataset, --particle-export-dir, or --particles-cs is required")
    if args.dataset is None and args.micrographs_dir is None:
        parser.error("--dataset is required unless --micrographs-dir is provided")
    if args.particles_cs is not None and args.particles_csg is None:
        parser.error("--particles-csg is required when --particles-cs is provided")
    if args.particles_csg is not None and args.particles_cs is None:
        parser.error("--particles-cs is required when --particles-csg is provided")
    if args.particle_export_dir is not None and (args.particles_cs is not None or args.particles_csg is not None):
        parser.error("--particle-export-dir cannot be combined with --particles-cs or --particles-csg")
    return args


def parse_percentiles(text: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Expected low,high percentiles such as 1,99")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    particle_export_dirs = list(args.particle_export_dir or [])
    particle_cs_paths = list(args.particles_cs or [])
    particle_csg_paths = list(args.particles_csg or [])
    if args.dataset is not None:
        if particle_export_dirs or particle_cs_paths:
            preset = get_dataset_preset(args.dataset)
            args.micrographs_dir = args.micrographs_dir or preset.micrographs_dir
        else:
            paths = resolve_standard_paths(
                args.dataset,
                micrographs_dir=args.micrographs_dir,
            )
            particle_cs_paths = [paths.particles_cs]
            particle_csg_paths = [paths.particles_csg]
            args.micrographs_dir = paths.micrographs_dir
    if particle_export_dirs:
        particle_cs_paths, particle_csg_paths = resolve_particle_export_dirs(particle_export_dirs)
    validate_particle_sources(particle_cs_paths, particle_csg_paths)
    low_high = parse_percentiles(args.contrast_percentiles)
    particles = load_particles_cs_many(particle_cs_paths)
    progress_callback = None if args.no_progress else print_micrograph_progress
    result = filter_particles_by_masks(
        particles,
        vesicle_mask_dir=args.vesicle_mask_dir,
        psize_a=args.psize_a,
        downsample=args.downsample,
        membrane_band_a=args.membrane_band_a,
        progress_callback=progress_callback,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_cs = args.output_dir / "filtered_particles.cs"
    rejected_cs = args.output_dir / "rejected_particles.cs"
    filtered_csg = args.output_dir / "filtered_particles.csg"
    rejected_csg = args.output_dir / "rejected_particles.csg"
    write_cs(filtered_cs, result.kept)
    write_cs(rejected_cs, result.rejected)
    template_particles_cs = particle_cs_paths[0]
    template_particles_csg = particle_csg_paths[0]
    patch_csg_cs_path(template_particles_csg, filtered_csg, template_particles_cs.name, filtered_cs)
    patch_csg_cs_path(template_particles_csg, rejected_csg, template_particles_cs.name, rejected_cs)
    write_tsv(args.output_dir / "filter_summary.tsv", result.micrograph_rows)
    write_summary_json(
        args.output_dir / "filter_summary.json",
        inputs={
            "particles_cs": [str(path) for path in particle_cs_paths],
            "particles_csg": [str(path) for path in particle_csg_paths],
            "template_particles_cs": str(template_particles_cs),
            "template_particles_csg": str(template_particles_csg),
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


def resolve_particle_export_dirs(directories: list[Path]) -> tuple[list[Path], list[Path]]:
    particle_cs_paths = []
    particle_csg_paths = []
    for directory in directories:
        particle_cs_paths.append(find_unique_file(directory, "*.cs"))
        particle_csg_paths.append(find_unique_file(directory, "*.csg"))
    return particle_cs_paths, particle_csg_paths


def validate_particle_sources(particle_cs_paths: list[Path], particle_csg_paths: list[Path]) -> None:
    if not particle_cs_paths:
        raise ValueError("At least one particle source is required")
    if len(particle_cs_paths) != len(particle_csg_paths):
        raise ValueError(
            f"Expected the same number of .cs and .csg inputs, got {len(particle_cs_paths)} .cs and "
            f"{len(particle_csg_paths)} .csg"
        )


def render_progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}] 0/0"
    completed = max(0, min(completed, total))
    filled = int(round(width * completed / total))
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total}"


def print_micrograph_progress(completed: int, total: int) -> None:
    sys.stderr.write("\rFiltering micrographs " + render_progress_bar(completed, total))
    if completed >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


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
