#!/usr/bin/env python3
"""Run Topaz-Denoise pretrained inference on a directory or list of MRC files.

This script performs the actual `topaz denoise` call. Run it inside an
interactive GPU allocation or from an sbatch job after activating the Topaz
environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from denoising_run_common import (
    DEFAULT_RESULTS,
    DEFAULT_SUBSET,
    add_sharding_args,
    allow_existing_output_dir,
    collect_manifest_mrc_files,
    collect_mrc_files,
    ensure_empty_or_create_output_dir,
    files_for_shard,
    require_executable,
    run_command,
    select_pending_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_SUBSET)],
        help="Input .mrc files and/or directories. Defaults to data/subset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS / "topaz_pretrained"),
        help="Directory for Topaz denoised outputs.",
    )
    parser.add_argument("--model", default="unet", help="Topaz pretrained model name")
    parser.add_argument("--device", default="0", help="CUDA device passed to Topaz")
    parser.add_argument("--patch-size", default="4000", help="Topaz patch size")
    parser.add_argument("--patch-padding", default="500", help="Topaz patch padding")
    parser.add_argument(
        "--suffix",
        default="_topaz_denoised",
        help="Suffix Topaz inserts before .mrc in output filenames.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument for `topaz denoise`; repeat as needed.",
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
    add_sharding_args(parser)
    return parser.parse_args()


def build_command(
    args: argparse.Namespace, files: list[Path], output_dir: Path
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
        "--suffix",
        args.suffix,
        "--output",
        str(output_dir),
    ]
    command.extend(args.extra_arg)
    command.extend(str(path) for path in files)
    return command


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        _ = require_executable("topaz")
    files = (
        collect_manifest_mrc_files(args.manifest)
        if args.manifest
        else collect_mrc_files(args.inputs)
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        ensure_empty_or_create_output_dir(
            output_dir,
            overwrite=allow_existing_output_dir(
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            ),
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = files_for_shard(
        files,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        chunk_size=args.chunk_size,
    )
    files = select_pending_files(
        files,
        output_dir=output_dir,
        suffix=args.suffix,
        skip_existing=args.skip_existing,
    )

    print(f"Running Topaz-Denoise pretrained model on {len(files)} MRC file(s).")
    if args.shard_index is not None:
        print(
            f"Shard: index={args.shard_index} chunk_size={args.chunk_size} "
            f"shard_count={args.shard_count}"
        )
    print(f"Output directory: {output_dir}")
    if not files:
        print("No pending files for this shard.")
        return
    command = build_command(args, files, output_dir)
    run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
