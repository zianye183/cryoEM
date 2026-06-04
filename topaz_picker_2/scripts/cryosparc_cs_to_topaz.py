#!/usr/bin/env python3
"""Convert CryoSPARC particle .cs exports into Topaz picker TSV metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = {
    "location/micrograph_path",
    "location/micrograph_shape",
    "location/center_x_frac",
    "location/center_y_frac",
}

IMAGE_EXTENSIONS = {
    ".mrc",
    ".mrcs",
    ".tif",
    ".tiff",
}


def decode_path(value: Any) -> str:
    """Decode CryoSPARC byte-string paths into plain strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def load_particles_cs(path: Path) -> np.ndarray:
    """Load a CryoSPARC .cs file and verify that it is a structured array."""
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise ValueError(f"{path} is not a structured CryoSPARC .cs array")
    return arr


def validate_required_fields(arr: np.ndarray, source: Path) -> None:
    """Require the CryoSPARC location fields needed for Topaz coordinates."""
    names = set(arr.dtype.names or ())
    missing = sorted(REQUIRED_FIELDS - names)
    if missing:
        raise ValueError(f"{source} missing required fields: {', '.join(missing)}")


def find_particles_cs(particles_dir: Path, explicit_particles_cs: Path | None = None) -> Path:
    """Find the single particle .cs file with the fields needed by this converter."""
    if explicit_particles_cs is not None:
        if not explicit_particles_cs.exists():
            raise FileNotFoundError(f"Particle .cs file does not exist: {explicit_particles_cs}")
        return explicit_particles_cs

    candidates = sorted(particles_dir.rglob("*.cs"))
    compatible: list[Path] = []
    incompatible: list[str] = []
    for candidate in candidates:
        try:
            arr = load_particles_cs(candidate)
            validate_required_fields(arr, candidate)
        except Exception as exc:  # keep scanning: export dirs may contain unrelated .cs files
            incompatible.append(f"{candidate}: {exc}")
            continue
        compatible.append(candidate)

    if not compatible:
        details = "\n".join(incompatible[:10])
        error_type = ValueError if candidates else FileNotFoundError
        raise error_type(
            f"No compatible particle .cs file found under {particles_dir}."
            + (f"\nChecked files:\n{details}" if details else "")
        )
    if len(compatible) > 1:
        joined = "\n".join(str(path) for path in compatible)
        raise ValueError(
            "Multiple compatible particle .cs files found; pass --particles-cs explicitly:\n"
            f"{joined}"
        )
    return compatible[0]


def topaz_image_name(path: str | Path) -> str:
    """Return the extensionless image identifier Topaz uses for coordinate tables."""
    return Path(path).stem


def build_micrograph_index(micrographs_dir: Path) -> dict[str, Path]:
    """Index exported micrograph files by extensionless Topaz image name."""
    index: dict[str, Path] = {}
    conflicts: dict[str, list[Path]] = {}
    for path in micrographs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = topaz_image_name(path)
        if name in index:
            conflicts.setdefault(name, [index[name]]).append(path)
        else:
            index[name] = path
    if conflicts:
        formatted = "\n".join(
            f"{name}: " + ", ".join(str(path) for path in paths)
            for name, paths in sorted(conflicts.items())
        )
        raise ValueError(
            "Micrograph export has duplicate Topaz image_name stems; "
            "Topaz image_name would be ambiguous:\n"
            f"{formatted}"
        )
    return index


def resolve_micrograph_path(
    cryosparc_micrograph_path: str,
    micrographs_dir: Path,
    micrograph_index: dict[str, Path],
) -> Path:
    """Resolve a CryoSPARC-relative micrograph path against an export directory."""
    raw_path = Path(cryosparc_micrograph_path)
    direct_candidates = [
        micrographs_dir / raw_path,
        micrographs_dir / raw_path.name,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    indexed = micrograph_index.get(topaz_image_name(raw_path))
    if indexed is not None:
        return indexed

    raise FileNotFoundError(
        f"Could not resolve micrograph {cryosparc_micrograph_path!r} under {micrographs_dir}"
    )


def split_micrographs(
    image_names: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split image names into non-overlapping train/test sets."""
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be greater than 0 and less than 1")
    if len(image_names) < 2:
        raise ValueError("Need at least two micrographs to create non-empty train/test splits")

    shuffled = list(image_names)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    test_count = min(test_count, len(shuffled) - 1)
    test_names = set(shuffled[:test_count])
    train_names = set(shuffled[test_count:])
    return train_names, test_names


TOPAZ_COORD_FIELDS = ["image_name", "x_coord", "y_coord"]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a tab-delimited metadata file."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_topaz_coord_rows(rows: list[dict[str, Any]], label: str) -> None:
    """Validate rows against the exact Topaz coord schema used by this pipeline."""
    if not rows:
        raise ValueError(f"{label} target rows are empty")
    for idx, row in enumerate(rows):
        missing = [field for field in TOPAZ_COORD_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"{label} row {idx} missing Topaz coordinate fields: {', '.join(missing)}"
            )
        image_name = str(row["image_name"])
        if not image_name:
            raise ValueError(f"{label} row {idx} has empty image_name")
        if Path(image_name).suffix:
            raise ValueError(
                f"{label} row {idx} image_name must be extensionless for Topaz: {image_name}"
            )
        for field in ("x_coord", "y_coord"):
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} row {idx} has non-numeric {field}: {row[field]!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{label} row {idx} has non-finite {field}: {value}")


def write_topaz_coord_targets(
    path_stem: Path,
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, str]:
    """Write Topaz coordinate targets as both archival TSV and Topaz-readable TXT."""
    validate_topaz_coord_rows(rows, label)
    outputs = {
        "tsv": path_stem.with_suffix(".tsv"),
        "txt": path_stem.with_suffix(".txt"),
    }
    for path in outputs.values():
        write_tsv(path, TOPAZ_COORD_FIELDS, rows)
    return {key: str(path) for key, path in outputs.items()}


def convert_cryosparc_cs_to_topaz(
    *,
    particles_dir: Path,
    micrographs_dir: Path,
    output_dir: Path,
    particles_cs: Path | None = None,
    downsample_factor: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Convert CryoSPARC particle locations to Topaz images/particles TSV files."""
    particles_dir = Path(particles_dir)
    micrographs_dir = Path(micrographs_dir)
    output_dir = Path(output_dir)
    particles_cs = Path(particles_cs) if particles_cs is not None else None

    if downsample_factor <= 0:
        raise ValueError("--downsample-factor must be positive")
    if not particles_dir.exists():
        raise FileNotFoundError(f"Particles directory does not exist: {particles_dir}")
    if not micrographs_dir.exists():
        raise FileNotFoundError(f"Micrographs directory does not exist: {micrographs_dir}")

    selected_particles_cs = find_particles_cs(particles_dir, particles_cs)
    arr = load_particles_cs(selected_particles_cs)
    validate_required_fields(arr, selected_particles_cs)
    if len(arr) == 0:
        raise ValueError(f"{selected_particles_cs} contains no particles")

    micrograph_index = build_micrograph_index(micrographs_dir)
    image_paths: dict[str, Path] = {}
    particles_rows: list[dict[str, str]] = []
    original_bounds_errors: list[str] = []
    scaled_bounds_errors: list[str] = []

    for idx, particle in enumerate(arr):
        rel_path = decode_path(particle["location/micrograph_path"])
        shape = particle["location/micrograph_shape"]
        if len(shape) != 2:
            raise ValueError(f"Particle row {idx} has invalid micrograph shape: {shape}")
        ny = int(shape[0])
        nx = int(shape[1])
        x_frac = float(particle["location/center_x_frac"])
        y_frac = float(particle["location/center_y_frac"])
        if not all(math.isfinite(value) for value in (x_frac, y_frac)):
            raise ValueError(f"Particle row {idx} has non-finite fractional coordinates")

        x_original = x_frac * nx
        y_original = y_frac * ny
        if not (0 <= x_original < nx and 0 <= y_original < ny):
            original_bounds_errors.append(
                f"row {idx}: {rel_path} -> x={x_original:.3f}, y={y_original:.3f}, shape=({ny}, {nx})"
            )
            continue

        x_topaz = x_original / downsample_factor
        y_topaz = y_original / downsample_factor
        nx_topaz = nx / downsample_factor
        ny_topaz = ny / downsample_factor
        if not (0 <= x_topaz < nx_topaz and 0 <= y_topaz < ny_topaz):
            scaled_bounds_errors.append(
                f"row {idx}: {rel_path} -> x={x_topaz:.3f}, y={y_topaz:.3f}, "
                f"scaled_shape=({ny_topaz:.3f}, {nx_topaz:.3f})"
            )
            continue

        image_name = topaz_image_name(rel_path)
        image_paths.setdefault(
            image_name,
            resolve_micrograph_path(rel_path, micrographs_dir, micrograph_index),
        )
        particles_rows.append(
            {
                "image_name": image_name,
                "x_coord": f"{x_topaz:.3f}",
                "y_coord": f"{y_topaz:.3f}",
            }
        )

    if original_bounds_errors:
        raise ValueError(
            "Some CryoSPARC coordinates fall outside their source image bounds:\n"
            + "\n".join(original_bounds_errors[:20])
        )
    if scaled_bounds_errors:
        raise ValueError(
            "Some converted Topaz coordinates fall outside scaled image bounds:\n"
            + "\n".join(scaled_bounds_errors[:20])
        )
    if not particles_rows:
        raise ValueError("No particle rows were converted")

    image_rows = [
        {"image_name": image_name, "path": str(image_paths[image_name])}
        for image_name in sorted(image_paths)
    ]
    image_names = [row["image_name"] for row in image_rows]
    train_names, test_names = split_micrographs(image_names, test_fraction, seed)

    train_image_rows = [row for row in image_rows if row["image_name"] in train_names]
    test_image_rows = [row for row in image_rows if row["image_name"] in test_names]
    train_particle_rows = [row for row in particles_rows if row["image_name"] in train_names]
    test_particle_rows = [row for row in particles_rows if row["image_name"] in test_names]
    if not train_particle_rows or not test_particle_rows:
        raise ValueError(
            "Train/test split produced an empty particle target file; choose another seed or split"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    images_all_path = output_dir / "images_all.tsv"
    train_images_path = output_dir / "train_images.tsv"
    test_images_path = output_dir / "test_images.tsv"
    write_tsv(images_all_path, ["image_name", "path"], image_rows)
    write_tsv(train_images_path, ["image_name", "path"], train_image_rows)
    write_tsv(test_images_path, ["image_name", "path"], test_image_rows)
    particles_all_outputs = write_topaz_coord_targets(
        output_dir / "particles_all",
        particles_rows,
        "all particles",
    )
    train_particle_outputs = write_topaz_coord_targets(
        output_dir / "train_particles",
        train_particle_rows,
        "train particles",
    )
    test_particle_outputs = write_topaz_coord_targets(
        output_dir / "test_particles",
        test_particle_rows,
        "test particles",
    )

    particle_counts = Counter(row["image_name"] for row in particles_rows)
    summary: dict[str, Any] = {
        "particles_cs": str(selected_particles_cs),
        "micrographs_dir": str(micrographs_dir),
        "output_dir": str(output_dir),
        "downsample_factor": downsample_factor,
        "test_fraction": test_fraction,
        "seed": seed,
        "num_micrographs": len(image_rows),
        "num_particles": len(particles_rows),
        "num_train_micrographs": len(train_image_rows),
        "num_test_micrographs": len(test_image_rows),
        "num_train_particles": len(train_particle_rows),
        "num_test_particles": len(test_particle_rows),
        "outputs": {
            "images_all": str(images_all_path),
            "particles_all": particles_all_outputs,
            "train_images": str(train_images_path),
            "train_particles": train_particle_outputs,
            "test_images": str(test_images_path),
            "test_particles": test_particle_outputs,
        },
        "particles_per_micrograph": dict(sorted(particle_counts.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CryoSPARC particle .cs exports into Topaz picker TSV files."
    )
    parser.add_argument("--micrographs-dir", required=True, type=Path)
    parser.add_argument("--particles-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--particles-cs",
        type=Path,
        help="Optional exact particle .cs file when the particles directory contains multiple compatible files.",
    )
    parser.add_argument("--downsample-factor", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_cryosparc_cs_to_topaz(
        particles_dir=args.particles_dir,
        micrographs_dir=args.micrographs_dir,
        output_dir=args.output_dir,
        particles_cs=args.particles_cs,
        downsample_factor=args.downsample_factor,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
