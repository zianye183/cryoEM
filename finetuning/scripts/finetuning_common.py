#!/usr/bin/env python3
"""Shared helpers for Prestin denoiser fine-tuning scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/nfs/bartesaghilab2/zy223/finetuning")
DEFAULT_FINETUNING_DATA_DIR = Path("/nfs/bartesaghilab2/zy223/data/finetuning_data")
DEFAULT_SAMPLE_DATASET = Path(
    "/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/"
    "projects/Prestin-test-WNXj0kBZzgoPdMbE/sp-preprocessing-gkym1IT0yo4s7y1L/mrc"
)
DEFAULT_ACCEPTED_EXPOSURES_CS = Path(
    "/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/"
    "J164_exposures_accepted/J164_exposures_accepted_exported.cs"
)
DEFAULT_MANIFEST_DIR = DEFAULT_FINETUNING_DATA_DIR / "topaz_prestin"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

PAIR_COLUMNS = [
    "micrograph_id",
    "even_path",
    "odd_path",
    "complete_path",
    "mask_path",
    "source_micrograph_path",
    "cryosparc_uid",
    "ctf_fit_to_a",
    "df1_a",
    "df2_a",
    "ice_thickness_rel",
    "session_id",
    "defocus_group",
    "pixel_size_a",
    "shape_y",
    "shape_x",
    "notes",
]

SPLIT_COLUMNS = PAIR_COLUMNS + ["split"]


def require_executable(name: str) -> str:
    """Return executable path or raise a clear error."""
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"Required command not found on PATH: {name}. Activate the correct environment first."
        )
    return found


def quote_command(command: Sequence[str | Path]) -> str:
    """Shell-quote a command for readable logging."""
    return " ".join(shlex.quote(str(part)) for part in command)


def run_command(command: Sequence[str | Path], *, dry_run: bool = False) -> None:
    """Print and optionally execute a command."""
    print(quote_command(command))
    if dry_run:
        print("Dry run only; command was not executed.")
        return
    subprocess.run([str(part) for part in command], check=True)


def ensure_empty_or_create_output_dir(path: Path, *, overwrite: bool = False) -> None:
    """Create an output directory and guard against accidental overwrite."""
    path.mkdir(parents=True, exist_ok=True)
    contents = [entry for entry in path.iterdir() if entry.name != ".gitkeep"]
    if contents and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to allow writing there."
        )


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file with a header into dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"TSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_tsv(
    path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]
) -> None:
    """Write dictionaries to a tab-separated file with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    """Compute SHA256 for a small metadata/manifest file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_summary(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Return a compact manifest summary for logs/reports."""
    shape_counts: dict[str, int] = {}
    mask_count = 0
    complete_count = 0
    for row in rows:
        shape = "x".join(
            part for part in [row.get("shape_y", ""), row.get("shape_x", "")] if part
        )
        if shape:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        if row.get("mask_path"):
            mask_count += 1
        if row.get("complete_path"):
            complete_count += 1
    return {
        "pairs": len(rows),
        "with_complete_micrograph": complete_count,
        "with_mask": mask_count,
        "shape_counts": shape_counts,
    }


def get_mrc_metadata(path: Path) -> tuple[str, str, str]:
    """Return `(shape_y, shape_x, pixel_size_a)` if `mrcfile` is available.

    Missing metadata is returned as empty strings so manifest generation can run
    in lightweight environments. Dimension validation should be rerun in an
    environment with `mrcfile` before training.
    """
    try:
        import mrcfile  # type: ignore
    except ImportError:
        return "", "", ""

    with mrcfile.open(path, permissive=True) as mrc:
        data_shape = tuple(int(dim) for dim in mrc.data.shape)
        if len(data_shape) == 2:
            shape_y, shape_x = data_shape
        elif len(data_shape) == 3:
            shape_y, shape_x = data_shape[-2], data_shape[-1]
        else:
            shape_y, shape_x = "", ""
        try:
            voxel_size = mrc.voxel_size
            pixel_size = float(voxel_size.x)
            pixel_size_text = f"{pixel_size:.6g}" if pixel_size > 0 else ""
        except Exception:
            pixel_size_text = ""
    return str(shape_y), str(shape_x), pixel_size_text


def validate_readable_file(path_text: str, *, label: str) -> Path:
    """Resolve and validate a required file path."""
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def paths_from_manifest(rows: Sequence[dict[str, str]], column: str) -> list[Path]:
    """Collect unique existing file paths from a manifest column."""
    paths: list[Path] = []
    for row in rows:
        raw_path = row.get(column, "")
        if not raw_path:
            continue
        paths.append(validate_readable_file(raw_path, label=column))
    unique = sorted(dict.fromkeys(paths))
    if not unique:
        raise ValueError(f"No paths found in manifest column: {column}")
    return unique


def write_command_log(path: Path, commands: Sequence[Sequence[str | Path]]) -> None:
    """Write shell-quoted commands to a text log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(quote_command(command))
            handle.write("\n")
