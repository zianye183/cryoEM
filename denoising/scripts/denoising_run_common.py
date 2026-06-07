#!/usr/bin/env python3
"""Shared helpers for denoising model runner scripts."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import argparse
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PROJECT = Path("/nfs/bartesaghilab2/zy223/denoising")
DEFAULT_SUBSET = DEFAULT_PROJECT / "data" / "subset"
DEFAULT_RESULTS = DEFAULT_PROJECT / "results"


def collect_mrc_files(paths: Sequence[str | Path]) -> list[Path]:
    """Return sorted MRC files from a mix of files and directories."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.mrc")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError("No .mrc files found in the provided inputs")
    return files


def collect_manifest_mrc_files(manifest: str | Path) -> list[Path]:
    """Return MRC paths from a newline or TSV manifest.

    The first column is used so manifests can be either a simple path list or a
    TSV with extra metadata columns. A header row is skipped when the first
    column looks like a path field name.
    """
    path = Path(manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    files: list[Path] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split("\t", 1)[0].strip()
            if line_number == 1 and first.lower() in {
                "path",
                "mrc",
                "micrograph",
                "micrograph_path",
                "input",
                "input_path",
            }:
                continue
            files.append(Path(first).expanduser().resolve())

    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No MRC paths found in manifest: {path}")
    return files


def add_sharding_args(parser: argparse.ArgumentParser) -> None:
    """Add common Slurm-array-friendly input sharding options."""
    parser.add_argument(
        "--manifest",
        help="Optional newline/TSV manifest of input MRC files. First column is used.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index, usually SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=None,
        help="Total number of balanced shards. Not required when --chunk-size is used.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Number of files per shard. Use with --shard-index for Slurm arrays.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inputs whose expected output file already exists.",
    )


def files_for_shard(
    files: Sequence[Path],
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    chunk_size: int = 0,
) -> list[Path]:
    """Return the files assigned to one shard while preserving deterministic order."""
    files = list(files)
    if shard_index is None:
        return files
    if shard_index < 0:
        raise ValueError("--shard-index must be >= 0")
    if chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    if chunk_size > 0:
        start = shard_index * chunk_size
        end = start + chunk_size
        return files[start:end]
    if shard_count is None:
        raise ValueError("--shard-count is required when --shard-index is used without --chunk-size")
    if shard_count <= 0:
        raise ValueError("--shard-count must be > 0")
    if shard_index >= shard_count:
        return []

    base = len(files) // shard_count
    remainder = len(files) % shard_count
    start = shard_index * base + min(shard_index, remainder)
    size = base + (1 if shard_index < remainder else 0)
    end = start + size
    return files[start:end]


def output_path_for_input(input_path: Path, output_dir: Path, suffix: str) -> Path:
    """Return the denoised output path for an input MRC and model suffix."""
    return output_dir / f"{input_path.stem}{suffix}.mrc"


def select_pending_files(
    files: Sequence[Path],
    *,
    output_dir: Path,
    suffix: str,
    skip_existing: bool,
) -> list[Path]:
    """Drop files whose expected output already exists when resume mode is enabled."""
    if not skip_existing:
        return list(files)
    return [
        path
        for path in files
        if not output_path_for_input(path, output_dir, suffix).exists()
    ]


def allow_existing_output_dir(*, overwrite: bool, skip_existing: bool) -> bool:
    """Return whether model runners may write into a non-empty output directory."""
    return overwrite or skip_existing


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
