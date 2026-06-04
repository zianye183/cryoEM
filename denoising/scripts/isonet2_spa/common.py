#!/usr/bin/env python3
"""Shared helpers for IsoNet2-on-SPA smoke-test scripts."""

from __future__ import annotations

import csv
import json
import math
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path("/nfs/bartesaghilab2/zy223/denoising")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "fuller_subset"
DEFAULT_WORK_DIR = PROJECT_ROOT / "experiments" / "isonet2_spa_smoke"
DEFAULT_MODEL = Path(
    "/nfs/bartesaghilab/nextpyp/workshop_dhvi/10453/"
    "isonet_network_isonet2-n2n_unet-medium_96_full.pt"
)


def collect_mrc_files(
    paths: Sequence[str | Path], *, limit: int | None = None
) -> list[Path]:
    """Return sorted MRC files from input files/directories/globs."""
    files: list[Path] = []
    for raw_path in paths:
        raw = str(raw_path)
        expanded = Path(raw).expanduser()
        if any(char in raw for char in "*?["):
            files.extend(
                sorted(
                    Path().glob(raw)
                    if not expanded.is_absolute()
                    else expanded.parent.glob(expanded.name)
                )
            )
        elif expanded.is_dir():
            files.extend(sorted(expanded.glob("*.mrc")))
        elif expanded.is_file():
            files.append(expanded)
        else:
            raise FileNotFoundError(f"Input path does not exist: {expanded}")

    unique = sorted(
        dict.fromkeys(path.resolve() for path in files if path.suffix.lower() == ".mrc")
    )
    if limit is not None:
        unique = unique[:limit]
    if not unique:
        raise FileNotFoundError("No .mrc files found in the provided inputs")
    return unique


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def quote_command(command: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: Sequence[str | Path], *, dry_run: bool = False) -> None:
    print(quote_command(command))
    if dry_run:
        print("Dry run only; command was not executed.")
        return
    subprocess.run([str(part) for part in command], check=True)


def safe_stem(path: Path) -> str:
    return path.name[:-4] if path.name.lower().endswith(".mrc") else path.stem


def infer_pixel_size_from_mrc(mrc_path: Path) -> float | None:
    """Read voxel_size.x with mrcfile; return None if unavailable/zero."""
    import mrcfile

    with mrcfile.open(mrc_path, permissive=True, header_only=True) as mrc:
        value = float(mrc.voxel_size.x)
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def write_star(
    path: Path, rows: list[dict[str, object]], columns: Sequence[str]
) -> None:
    """Write a simple RELION-style STAR loop compatible with starfile."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("data_\n\n")
        handle.write("loop_\n")
        for idx, column in enumerate(columns, start=1):
            handle.write(f"_{column} #{idx}\n")
        for row in rows:
            values = [format_star_value(row.get(column, "None")) for column in columns]
            handle.write("\t".join(values) + "\n")


def format_star_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def robust_z(array):
    import numpy as np

    med = np.median(array)
    mad = np.median(np.abs(array - med)) + 1e-6
    return (array - med) / (1.4826 * mad)
