#!/usr/bin/env python3
"""Run DRACO pretrained Denoise/B inference.

This runner expands input directories, checks project paths, and executes the
local DRACO denoising adapter by default. The adapter is based on the official
DRACO notebook/draco_denoise.ipynb and expects a local clone of the DRACO repo
plus the Denoise/B checkpoint.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from denoising_run_common import (
    DEFAULT_PROJECT,
    DEFAULT_RESULTS,
    DEFAULT_SUBSET,
    collect_mrc_files,
    ensure_empty_or_create_output_dir,
    run_command,
)

DEFAULT_DRACO_REPO = DEFAULT_PROJECT / "external" / "draco"
DEFAULT_DRACO_WEIGHTS = DEFAULT_PROJECT / "models" / "draco_denoise_b.ckpt"
DEFAULT_DRACO_ADAPTER = DEFAULT_PROJECT / "scripts" / "draco_denoise_adapter.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_SUBSET)],
        help="Input .mrc files and/or directories. Defaults to data/subset.",
    )
    parser.add_argument(
        "--repo",
        default=str(DEFAULT_DRACO_REPO),
        help="Local DRACO repository path.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_DRACO_WEIGHTS),
        help="Path to pretrained DRACO Denoise/B weights/checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS / "draco_pretrained"),
        help="Directory for DRACO denoised outputs.",
    )
    parser.add_argument(
        "--adapter-script",
        default=str(DEFAULT_DRACO_ADAPTER),
        help="Python script that performs DRACO inference using --inputs/--output-dir/--weights/--repo.",
    )
    parser.add_argument(
        "--command-template",
        help=(
            "Explicit command template to run instead of --adapter-script. "
            "Available placeholders: {inputs}, {output_dir}, {weights}, {repo}."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Additional raw argument passed to the adapter script; repeat as needed. "
            "Examples: --extra-arg --device --extra-arg cuda"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    return parser.parse_args()


def build_adapter_command(
    adapter_script: str,
    files: list[Path],
    output_dir: Path,
    weights: Path,
    repo: Path,
    extra_args: list[str],
    *,
    dry_run: bool = False,
) -> list[str]:
    script = Path(adapter_script).expanduser().resolve()
    if not dry_run and not script.is_file():
        raise FileNotFoundError(f"DRACO adapter script does not exist: {script}")

    command = [
        sys.executable,
        str(script),
        "--output-dir",
        str(output_dir),
        "--weights",
        str(weights),
        "--repo",
        str(repo),
        "--inputs",
    ]
    command.extend(str(path) for path in files)
    command.extend(extra_args)
    return command


def build_template_command(
    template: str,
    files: list[Path],
    output_dir: Path,
    weights: Path,
    repo: Path,
) -> list[str]:
    rendered = template.format(
        inputs=" ".join(shlex.quote(str(path)) for path in files),
        output_dir=shlex.quote(str(output_dir)),
        weights=shlex.quote(str(weights)),
        repo=shlex.quote(str(repo)),
    )
    return shlex.split(rendered)


def main() -> None:
    args = parse_args()
    files = collect_mrc_files(args.inputs)
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()

    if args.command_template and args.adapter_script != str(DEFAULT_DRACO_ADAPTER):
        raise SystemExit(
            "Provide either --command-template or --adapter-script, not both."
        )
    if args.command_template:
        args.adapter_script = None
    if not args.dry_run:
        if not repo.is_dir():
            raise FileNotFoundError(f"DRACO repo does not exist: {repo}")
        if not weights.is_file():
            raise FileNotFoundError(
                f"DRACO weights/checkpoint does not exist: {weights}"
            )
        ensure_empty_or_create_output_dir(output_dir, overwrite=args.overwrite)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRACO pretrained Denoise/B adapter on {len(files)} MRC file(s).")
    print(f"DRACO repo: {repo}")
    print(f"Weights: {weights}")
    print(f"Output directory: {output_dir}")

    if args.adapter_script:
        command = build_adapter_command(
            args.adapter_script,
            files,
            output_dir,
            weights,
            repo,
            args.extra_arg,
            dry_run=args.dry_run,
        )
    else:
        command = build_template_command(
            args.command_template,
            files,
            output_dir,
            weights,
            repo,
        )

    run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
