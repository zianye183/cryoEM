#!/usr/bin/env python3
"""Convert CryoSPARC particle .cs exports into Topaz training TSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_PARTICLE_FIELDS = (
    "location/micrograph_path",
    "location/micrograph_shape",
    "location/center_x_frac",
    "location/center_y_frac",
)
IMAGE_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CryoSPARC particle .cs files to Topaz image/coordinate TSVs.")
    parser.add_argument("--micrographs-dir", required=True, type=Path)
    parser.add_argument("--particles-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--particle-cs-name", default=None, help="Particle .cs filename/path. Required if multiple .cs files are present.")
    parser.add_argument("--downsample-factor", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_directory(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} does not exist: {path}")
    if not path.is_dir():
        fail(f"{label} is not a directory: {path}")


def find_particle_cs(particles_dir: Path, particle_cs_name: str | None) -> Path:
    if particle_cs_name:
        candidate = Path(particle_cs_name)
        if not candidate.is_absolute():
            candidate = particles_dir / candidate
        if not candidate.exists():
            fail(f"Requested particle .cs file does not exist: {candidate}")
        if candidate.suffix != ".cs":
            fail(f"Requested particle file is not a .cs file: {candidate}")
        return candidate

    candidates = sorted(particles_dir.rglob("*.cs"))
    if not candidates:
        fail(f"No .cs files found under particles directory: {particles_dir}")
    if len(candidates) > 1:
        rels = "\n  ".join(str(path.relative_to(particles_dir)) for path in candidates)
        fail("Multiple .cs files found under particles directory. Re-run with --particle-cs-name. Candidates:\n  " + rels)
    return candidates[0]


def load_particles(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        fail(f"Failed to load particle .cs file {path}: {exc}")
    if arr.dtype.names is None:
        fail(f"Particle .cs file is not a NumPy structured array: {path}")
    missing = [field for field in REQUIRED_PARTICLE_FIELDS if field not in arr.dtype.names]
    if missing:
        available = ", ".join(arr.dtype.names)
        fail(f"Particle .cs is missing required fields {missing}. Available fields: {available}")
    if len(arr) == 0:
        fail(f"Particle .cs file contains no particles: {path}")
    return arr


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return value


def micrograph_rel_path(row: np.void) -> str:
    return str(decode_value(row["location/micrograph_path"]))


def build_micrograph_index(micrographs_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in micrographs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name].append(path.resolve())
    if not index:
        fail(f"No supported image files found under {micrographs_dir}")
    return dict(index)


def resolve_micrograph_paths(arr: np.ndarray, micrographs_dir: Path) -> tuple[dict[str, Path], list[str]]:
    index = build_micrograph_index(micrographs_dir)
    unique_rel_paths = sorted({micrograph_rel_path(row) for row in arr})
    resolved: dict[str, Path] = {}
    errors: list[str] = []
    for rel_path in unique_rel_paths:
        basename = Path(rel_path).name
        matches = index.get(basename, [])
        if len(matches) == 1:
            resolved[rel_path] = matches[0]
        elif len(matches) == 0:
            direct = micrographs_dir / rel_path
            if direct.exists() and direct.is_file():
                resolved[rel_path] = direct.resolve()
            else:
                errors.append(f"missing micrograph for {rel_path!r} (basename {basename!r})")
        else:
            rendered = ", ".join(str(path) for path in matches[:5])
            suffix = "" if len(matches) <= 5 else f", ... {len(matches) - 5} more"
            errors.append(f"ambiguous micrograph basename {basename!r}: {rendered}{suffix}")
    return resolved, errors


def coordinate_from_row(row: np.void, downsample_factor: float) -> tuple[float, float, float, float, int, int]:
    shape = row["location/micrograph_shape"]
    if len(shape) < 2:
        raise ValueError(f"invalid micrograph shape {shape!r}")
    ny = int(shape[0])
    nx = int(shape[1])
    x_frac = float(row["location/center_x_frac"])
    y_frac = float(row["location/center_y_frac"])
    if not math.isfinite(x_frac) or not math.isfinite(y_frac):
        raise ValueError(f"non-finite fractional coordinate ({x_frac}, {y_frac})")
    if nx <= 0 or ny <= 0:
        raise ValueError(f"invalid micrograph shape [ny={ny}, nx={nx}]")
    x_raw = x_frac * nx
    y_raw = y_frac * ny
    return x_raw, y_raw, x_raw / downsample_factor, y_raw / downsample_factor, nx, ny


def split_micrographs(image_names: list[str], test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0 < test_fraction < 1:
        fail(f"--test-fraction must be between 0 and 1, got {test_fraction}")
    if len(image_names) < 2:
        fail("Need at least two unique micrographs to create train/test split")
    names = list(image_names)
    rng = random.Random(seed)
    rng.shuffle(names)
    test_count = max(1, round(len(names) * test_fraction))
    test_count = min(test_count, len(names) - 1)
    return set(names[test_count:]), set(names[:test_count])


def write_tsv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_counts(counts: Counter[str]) -> dict[str, float | int]:
    values = list(counts.values())
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def main() -> None:
    args = parse_args()
    if args.downsample_factor <= 0 or not math.isfinite(args.downsample_factor):
        fail(f"--downsample-factor must be positive and finite, got {args.downsample_factor}")

    micrographs_dir = args.micrographs_dir.resolve()
    particles_dir = args.particles_dir.resolve()
    output_dir = args.output_dir
    require_directory(micrographs_dir, "--micrographs-dir")
    require_directory(particles_dir, "--particles-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    particle_cs = find_particle_cs(particles_dir, args.particle_cs_name).resolve()
    arr = load_particles(particle_cs)
    resolved_paths, path_errors = resolve_micrograph_paths(arr, micrographs_dir)
    if path_errors:
        fail("Could not resolve all referenced micrographs:\n  " + "\n  ".join(path_errors))

    image_rows_by_name: dict[str, dict[str, str]] = {}
    particle_rows: list[dict[str, Any]] = []
    particles_per_image: Counter[str] = Counter()
    shape_counter: Counter[str] = Counter()
    psize_counter: Counter[str] = Counter()
    coordinate_errors: list[str] = []

    for idx, row in enumerate(arr):
        rel_path = micrograph_rel_path(row)
        resolved = resolved_paths[rel_path]
        image_name = resolved.name
        try:
            x_raw, y_raw, x_topaz, y_topaz, nx, ny = coordinate_from_row(row, args.downsample_factor)
        except ValueError as exc:
            coordinate_errors.append(f"particle {idx}: {exc}")
            continue
        if not (0 <= x_raw <= nx and 0 <= y_raw <= ny):
            coordinate_errors.append(f"particle {idx}: raw coordinate outside image bounds: x={x_raw}, y={y_raw}, nx={nx}, ny={ny}")
            continue
        nx_topaz = nx / args.downsample_factor
        ny_topaz = ny / args.downsample_factor
        if not (0 <= x_topaz <= nx_topaz and 0 <= y_topaz <= ny_topaz):
            coordinate_errors.append(f"particle {idx}: Topaz coordinate outside bounds: x={x_topaz}, y={y_topaz}, nx={nx_topaz}, ny={ny_topaz}")
            continue

        image_row = {"image_name": image_name, "path": str(resolved)}
        existing = image_rows_by_name.get(image_name)
        if existing is not None and existing != image_row:
            fail(f"Image name collision for {image_name}: {existing['path']} vs {resolved}")
        image_rows_by_name[image_name] = image_row
        particle_rows.append({"image_name": image_name, "x_coord": f"{x_topaz:.6f}", "y_coord": f"{y_topaz:.6f}"})
        particles_per_image[image_name] += 1
        shape_counter[f"{ny}x{nx}"] += 1
        if "location/micrograph_psize_A" in arr.dtype.names:
            psize = float(row["location/micrograph_psize_A"])
            if math.isfinite(psize):
                psize_counter[f"{psize:.6g}"] += 1

    if coordinate_errors:
        preview = "\n  ".join(coordinate_errors[:20])
        suffix = "" if len(coordinate_errors) <= 20 else f"\n  ... {len(coordinate_errors) - 20} more"
        fail(f"Coordinate validation failed:\n  {preview}{suffix}")
    if not particle_rows:
        fail("No valid particle rows were produced")

    image_rows = [image_rows_by_name[name] for name in sorted(image_rows_by_name)]
    train_images, test_images = split_micrographs([row["image_name"] for row in image_rows], args.test_fraction, args.seed)
    train_image_rows = [row for row in image_rows if row["image_name"] in train_images]
    test_image_rows = [row for row in image_rows if row["image_name"] in test_images]
    train_particle_rows = [row for row in particle_rows if row["image_name"] in train_images]
    test_particle_rows = [row for row in particle_rows if row["image_name"] in test_images]
    if not train_particle_rows:
        fail("Train split contains no particles; choose a different --test-fraction or --seed")
    if not test_particle_rows:
        fail("Test split contains no particles; choose a different --test-fraction or --seed")

    write_tsv(output_dir / "images_all.tsv", ["image_name", "path"], image_rows)
    write_tsv(output_dir / "particles_all.tsv", ["image_name", "x_coord", "y_coord"], particle_rows)
    write_tsv(output_dir / "train_images.tsv", ["image_name", "path"], train_image_rows)
    write_tsv(output_dir / "train_particles.tsv", ["image_name", "x_coord", "y_coord"], train_particle_rows)
    write_tsv(output_dir / "test_images.tsv", ["image_name", "path"], test_image_rows)
    write_tsv(output_dir / "test_particles.tsv", ["image_name", "x_coord", "y_coord"], test_particle_rows)

    summary = {
        "particles_cs": str(particle_cs),
        "micrographs_dir": str(micrographs_dir),
        "particles_dir": str(particles_dir),
        "output_dir": str(output_dir),
        "downsample_factor": args.downsample_factor,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "num_particles": len(particle_rows),
        "num_micrographs": len(image_rows),
        "num_train_micrographs": len(train_image_rows),
        "num_test_micrographs": len(test_image_rows),
        "num_train_particles": len(train_particle_rows),
        "num_test_particles": len(test_particle_rows),
        "particles_per_micrograph": summarize_counts(particles_per_image),
        "shape_counts": dict(sorted(shape_counter.items())),
        "pixel_size_A_counts": dict(sorted(psize_counter.items())),
        "outputs": {
            "images_all": str(output_dir / "images_all.tsv"),
            "particles_all": str(output_dir / "particles_all.tsv"),
            "train_images": str(output_dir / "train_images.tsv"),
            "train_particles": str(output_dir / "train_particles.tsv"),
            "test_images": str(output_dir / "test_images.tsv"),
            "test_particles": str(output_dir / "test_particles.tsv"),
        },
        "warnings": [],
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
