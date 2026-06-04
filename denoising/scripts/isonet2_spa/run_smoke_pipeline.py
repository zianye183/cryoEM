#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Run the minimal IsoNet2-on-SPA smoke-test pipeline.

This driver prepares pseudo-volumes and a STAR file, then optionally launches
IsoNet2 prediction. It intentionally does not train, deconvolve, or make masks.
Activate an IsoNet2-capable module/conda environment before running prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_WORK_DIR,
    run_command,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_INPUT_DIR)],
        help="Input .mrc files/directories/globs.",
    )
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="IsoNet2 checkpoint path."
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Experiment working directory.",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Number of micrographs to test."
    )
    parser.add_argument(
        "--depth", default="auto", help="Pseudo-volume depth or auto from checkpoint."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help=(
            "Center-crop each micrograph before fake-Z repetition. "
            "Default 1024 keeps the first smoke test small enough for IsoNet2 prediction. "
            "Use --crop-size 0 to disable cropping."
        ),
    )
    parser.add_argument(
        "--stack-policy", choices=["error", "mean", "first"], default="error"
    )
    parser.add_argument(
        "--pixel-size",
        default="auto",
        help="STAR pixel size in Å or auto from MRC header.",
    )
    parser.add_argument("--fallback-pixel-size", type=float, default=1.0)
    parser.add_argument("--defocus-a", type=float, default=10000.0)
    parser.add_argument("--voltage-kv", type=float, default=300.0)
    parser.add_argument("--cs-mm", type=float, default=2.7)
    parser.add_argument("--amplitude-contrast", type=float, default=0.1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--isonet-command", default="isonet.py")
    parser.add_argument(
        "--run-predict",
        action="store_true",
        help="Actually launch isonet.py predict. Default prepares inputs only.",
    )
    parser.add_argument(
        "--dry-run-predict",
        action="store_true",
        help="Print prediction command without executing it.",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip slice extraction/QC after prediction.",
    )
    parser.add_argument("--invert-input", action="store_true")
    return parser.parse_args()


def run_step(command: list[str | Path], *, dry_run: bool = False) -> None:
    run_command(command, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    metadata_json = work_dir / "checkpoint_metadata.json"
    pseudo_dir = work_dir / "pseudo_volumes"
    star_path = work_dir / "stars" / "spa_pseudo.star"
    predict_dir = work_dir / "predict"
    slices_dir = work_dir / "extracted_slices"
    qc_dir = work_dir / "qc"

    pipeline_manifest = {
        "model": str(Path(args.model).expanduser()),
        "inputs": args.inputs,
        "work_dir": str(work_dir),
        "limit": args.limit,
        "depth": args.depth,
        "stack_policy": args.stack_policy,
        "crop_size": args.crop_size,
        "pixel_size": args.pixel_size,
        "fallback_pixel_size": args.fallback_pixel_size,
        "defocus_A": args.defocus_a,
        "run_predict": args.run_predict,
        "dry_run_predict": args.dry_run_predict,
    }
    write_json(work_dir / "run_manifest.json", pipeline_manifest)

    run_step(
        [
            sys.executable,
            SCRIPT_DIR / "inspect_checkpoint.py",
            "--model",
            args.model,
            "--out",
            metadata_json,
        ]
    )

    pseudo_cmd: list[str | Path] = [
        sys.executable,
        SCRIPT_DIR / "make_pseudo_volumes.py",
        *args.inputs,
        "--output-dir",
        pseudo_dir,
        "--metadata-json",
        metadata_json,
        "--depth",
        args.depth,
        "--limit",
        str(args.limit),
        "--stack-policy",
        args.stack_policy,
        "--manifest",
        pseudo_dir / "manifest.csv",
    ]
    if args.crop_size:
        pseudo_cmd.extend(["--crop-size", str(args.crop_size)])
    if args.invert_input:
        pseudo_cmd.append("--invert-input")
    run_step(pseudo_cmd)

    run_step(
        [
            sys.executable,
            SCRIPT_DIR / "make_spa_star.py",
            str(pseudo_dir),
            "--out",
            star_path,
            "--pixel-size",
            args.pixel_size,
            "--fallback-pixel-size",
            str(args.fallback_pixel_size),
            "--defocus-a",
            str(args.defocus_a),
            "--voltage-kv",
            str(args.voltage_kv),
            "--cs-mm",
            str(args.cs_mm),
            "--amplitude-contrast",
            str(args.amplitude_contrast),
        ]
    )

    if args.run_predict or args.dry_run_predict:
        run_step(
            [
                sys.executable,
                SCRIPT_DIR / "run_predict.py",
                "--star",
                star_path,
                "--model",
                args.model,
                "--output-dir",
                predict_dir,
                "--gpu-id",
                args.gpu_id,
                "--isonet-command",
                args.isonet_command,
                "--no-apply-mw-x1",
                "--is-ctf-flipped",
                "--output-prefix",
                "spa_smoke",
                *(
                    ["--dry-run"]
                    if args.dry_run_predict and not args.run_predict
                    else []
                ),
            ]
        )

    if args.run_predict and not args.skip_qc:
        run_step(
            [
                sys.executable,
                SCRIPT_DIR / "extract_slices.py",
                "--predict-dir",
                predict_dir,
                "--output-dir",
                slices_dir,
                "--manifest",
                slices_dir / "manifest.csv",
            ]
        )
        run_step(
            [
                sys.executable,
                SCRIPT_DIR / "qc_outputs.py",
                "--pseudo-manifest",
                pseudo_dir / "manifest.csv",
                "--slices-manifest",
                slices_dir / "manifest.csv",
                "--output-dir",
                qc_dir,
                "--stack-policy",
                args.stack_policy,
            ]
        )

    print(f"IsoNet2 SPA smoke-test setup complete under {work_dir}")


if __name__ == "__main__":
    main()
