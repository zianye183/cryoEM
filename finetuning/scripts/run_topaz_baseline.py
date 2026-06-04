#!/usr/bin/env python3
"""Run pretrained Topaz denoising on manifest-listed held-out micrographs.

Use this before fine-tuning to establish a frozen baseline on validation/test
splits. The script can denoise half1, half2, and optionally complete
micrographs. It does not modify source data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RESULTS_DIR,
    ensure_empty_or_create_output_dir,
    paths_from_manifest,
    read_tsv,
    require_executable,
    run_command,
    write_command_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_DIR / "test_pairs.tsv"),
        help="Pair manifest/split TSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "topaz_pretrained_pilot"),
        help="Output directory for denoised files and command logs.",
    )
    parser.add_argument("--model", default="unet", help="Topaz pretrained model name")
    parser.add_argument("--device", default="0")
    parser.add_argument("--patch-size", default="4000")
    parser.add_argument("--patch-padding", default="500")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["even_path", "odd_path"],
        choices=["even_path", "odd_path", "complete_path"],
        help="Manifest path columns to denoise.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument for `topaz denoise`; repeat as needed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    files: list[Path],
    output_dir: Path,
) -> list[str]:
    command = [
        "topaz",
        "denoise",
        "--model",
        args.model,
        "--device",
        args.device,
        "--patch-size",
        args.patch_size,
        "--patch-padding",
        args.patch_padding,
        "--output",
        str(output_dir),
    ]
    command.extend(args.extra_arg)
    command.extend(str(path) for path in files)
    return command


def subdir_for_column(output_dir: Path, column: str) -> Path:
    names = {
        "even_path": "denoised_even",
        "odd_path": "denoised_odd",
        "complete_path": "denoised_complete",
    }
    return output_dir / names[column]


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        require_executable("topaz")

    manifest = Path(args.manifest).expanduser().resolve()
    rows = read_tsv(manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        ensure_empty_or_create_output_dir(output_dir, overwrite=args.overwrite)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str | Path]] = []
    for column in args.columns:
        files = paths_from_manifest(rows, column)
        column_output = subdir_for_column(output_dir, column)
        column_output.mkdir(parents=True, exist_ok=True)
        command = build_command(args, files, column_output)
        commands.append(command)
        print(f"Prepared Topaz baseline command for {column}: {len(files)} file(s)")
        run_command(command, dry_run=args.dry_run)

    command_log = output_dir / "commands.txt"
    write_command_log(command_log, commands)
    print(f"Command log: {command_log}")


if __name__ == "__main__":
    main()
