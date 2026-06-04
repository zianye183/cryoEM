#!/usr/bin/env python3
"""Run Topaz-Denoise retraining from paired noisy observations.

Topaz denoising retraining is Noise2Noise-style: each training example should
have two independently noisy observations of the same underlying micrograph,
for example even-frame and odd-frame sums. This script expects those paired
files to already exist and launches the installed Topaz training command.

Because Topaz CLI flags can vary by version, use --extra-arg to pass any
site/version-specific options after checking `topaz denoise-train --help`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from denoising_run_common import (
    DEFAULT_RESULTS,
    ensure_empty_or_create_output_dir,
    require_executable,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--even-dir",
        help="Directory containing even-frame/even-observation .mrc files.",
    )
    parser.add_argument(
        "--odd-dir",
        help="Directory containing odd-frame/odd-observation .mrc files with matching names.",
    )
    parser.add_argument(
        "--pairs-tsv",
        help="Optional TSV with two columns: even_path<TAB>odd_path. Overrides --even-dir/--odd-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS / "topaz_retrained"),
        help="Directory for trained Topaz model/checkpoints.",
    )
    parser.add_argument("--device", default="0", help="CUDA device passed to Topaz")
    parser.add_argument(
        "--command",
        default="denoise-train",
        help="Topaz training subcommand to use. Default: denoise-train.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument for the Topaz training command; repeat as needed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and generated pair list without executing training.",
    )
    return parser.parse_args()


def collect_pairs_from_dirs(even_dir: str, odd_dir: str) -> list[tuple[Path, Path]]:
    even_root = Path(even_dir).expanduser().resolve()
    odd_root = Path(odd_dir).expanduser().resolve()
    if not even_root.is_dir():
        raise FileNotFoundError(f"Even directory does not exist: {even_root}")
    if not odd_root.is_dir():
        raise FileNotFoundError(f"Odd directory does not exist: {odd_root}")

    pairs: list[tuple[Path, Path]] = []
    for even_path in sorted(even_root.glob("*.mrc")):
        odd_path = odd_root / even_path.name
        if not odd_path.is_file():
            raise FileNotFoundError(
                f"Missing matching odd file for {even_path.name}: {odd_path}"
            )
        pairs.append((even_path, odd_path))

    if not pairs:
        raise FileNotFoundError(f"No .mrc files found in even directory: {even_root}")
    return pairs


def collect_pairs_from_tsv(path: str) -> list[tuple[Path, Path]]:
    pairs_path = Path(path).expanduser().resolve()
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pair TSV does not exist: {pairs_path}")

    pairs: list[tuple[Path, Path]] = []
    with pairs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cols = stripped.split("\t")
            if len(cols) != 2:
                raise ValueError(
                    f"Expected two tab-separated columns at {pairs_path}:{line_number}"
                )
            even_path = Path(cols[0]).expanduser().resolve()
            odd_path = Path(cols[1]).expanduser().resolve()
            if not even_path.is_file():
                raise FileNotFoundError(
                    f"Missing even file at {pairs_path}:{line_number}: {even_path}"
                )
            if not odd_path.is_file():
                raise FileNotFoundError(
                    f"Missing odd file at {pairs_path}:{line_number}: {odd_path}"
                )
            pairs.append((even_path, odd_path))

    if not pairs:
        raise ValueError(f"No training pairs found in {pairs_path}")
    return pairs


def write_pair_list(pairs: list[tuple[Path, Path]], output_dir: Path) -> Path:
    pair_list = output_dir / "topaz_denoise_pairs.tsv"
    with pair_list.open("w", encoding="utf-8") as handle:
        for even_path, odd_path in pairs:
            handle.write(f"{even_path}\t{odd_path}\n")
    return pair_list


def build_command(
    args: argparse.Namespace, pair_list: Path, output_dir: Path
) -> list[str]:
    # Topaz denoise-training flags have changed across releases. The pair-list
    # flags below document the intended handoff; use --dry-run and
    # `topaz denoise-train --help` before a full training job on a new install.
    command = [
        "topaz",
        args.command,
        "--device",
        args.device,
        "--output",
        str(output_dir),
        "--train-pairs",
        str(pair_list),
    ]
    command.extend(args.extra_arg)
    return command


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        _ = require_executable("topaz")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        ensure_empty_or_create_output_dir(output_dir, overwrite=args.overwrite)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_tsv:
        pairs = collect_pairs_from_tsv(args.pairs_tsv)
    elif args.even_dir and args.odd_dir:
        pairs = collect_pairs_from_dirs(args.even_dir, args.odd_dir)
    else:
        raise SystemExit("Provide either --pairs-tsv or both --even-dir and --odd-dir.")

    pair_list = write_pair_list(pairs, output_dir)
    print(f"Prepared {len(pairs)} Topaz retraining pair(s).")
    print(f"Pair list: {pair_list}")
    print(f"Output directory: {output_dir}")
    command = build_command(args, pair_list, output_dir)
    run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
