#!/usr/bin/env python3
"""Shared helpers for denoising model runner scripts."""

from __future__ import annotations

import shlex
import shutil
import subprocess
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
