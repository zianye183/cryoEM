#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Launch IsoNet2 prediction on SPA pseudo-volumes from a prepared STAR file."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_MODEL, DEFAULT_WORK_DIR, ensure_dir, run_command, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--star",
        default=str(DEFAULT_WORK_DIR / "stars" / "spa_pseudo.star"),
        help="IsoNet2 STAR file from make_spa_star.py.",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="IsoNet2 .pt checkpoint path."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_WORK_DIR / "predict"),
        help="IsoNet2 prediction output directory.",
    )
    parser.add_argument(
        "--gpu-id", default="0", help="GPU ID string passed to IsoNet2 --gpuID."
    )
    parser.add_argument(
        "--isonet-command",
        default="isonet.py",
        help="IsoNet2 CLI executable. Activate module/conda env before running, or pass full path.",
    )
    parser.add_argument(
        "--apply-mw-x1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether IsoNet2 should apply missing-wedge mask. Default: false for SPA.",
    )
    parser.add_argument(
        "--is-ctf-flipped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Passed as --isCTFflipped. Default true to skip IsoNet2 CTF sign filtering.",
    )
    parser.add_argument(
        "--padding-factor",
        type=float,
        default=1.5,
        help="IsoNet2 prediction padding_factor.",
    )
    parser.add_argument("--tomo-idx", help="Optional IsoNet2 tomo_idx subset string.")
    parser.add_argument(
        "--output-prefix", default="spa_smoke", help="Output prefix for IsoNet2 files."
    )
    parser.add_argument(
        "--save-slices", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    star = Path(args.star).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if not args.dry_run:
        if not star.is_file():
            raise FileNotFoundError(f"STAR file does not exist: {star}")
        if not model.is_file():
            raise FileNotFoundError(f"IsoNet2 checkpoint does not exist: {model}")

    command = [
        args.isonet_command,
        "predict",
        str(star),
        str(model),
        "--output_dir",
        str(output_dir),
        "--gpuID",
        args.gpu_id,
        "--apply_mw_x1",
        str(args.apply_mw_x1),
        "--isCTFflipped",
        str(args.is_ctf_flipped),
        "--padding_factor",
        str(args.padding_factor),
        "--output_prefix",
        args.output_prefix,
        "--save_slices",
        str(args.save_slices),
    ]
    if args.tomo_idx:
        command.extend(["--tomo_idx", args.tomo_idx])

    manifest = {
        "star": str(star),
        "model": str(model),
        "output_dir": str(output_dir),
        "gpu_id": args.gpu_id,
        "apply_mw_x1": args.apply_mw_x1,
        "is_ctf_flipped": args.is_ctf_flipped,
        "padding_factor": args.padding_factor,
        "output_prefix": args.output_prefix,
        "command": [str(part) for part in command],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
