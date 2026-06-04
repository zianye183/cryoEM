#!/usr/bin/env python3
"""Run Topaz extraction and held-out precision-recall evaluation for a baseline run.

This helper intentionally starts after metadata conversion/training. It expects
converter-generated test_images.tsv/test_particles.tsv and a trained Topaz .sav
model in the run's models directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Topaz extract on held-out test images and evaluate against test particles."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/j42_baseline"),
        help="Baseline run directory. Default: runs/j42_baseline.",
    )
    parser.add_argument(
        "--topaz-bin",
        type=Path,
        default=Path("/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz"),
        help="Topaz executable path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Specific Topaz .sav model. If omitted, choose the newest .sav in RUN_DIR/models.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=10,
        help="Extraction/evaluation radius in image pixels. Default: 10.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-6.0,
        help="Topaz extract score threshold. Default: -6.0.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Output predictions coordinate file. Default: RUN_DIR/predictions/test_predictions.tab.",
    )
    parser.add_argument(
        "--pr-curve",
        type=Path,
        default=None,
        help="Output precision-recall curve TSV. Default: RUN_DIR/eval/pr_curve.tsv.",
    )
    parser.add_argument(
        "--skip-pr",
        action="store_true",
        help="Only run topaz extract; skip precision_recall_curve.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def read_test_image_paths(test_images_tsv: Path) -> list[str]:
    if not test_images_tsv.is_file():
        fail(f"test image metadata not found: {test_images_tsv}")

    with test_images_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            fail(f"empty TSV: {test_images_tsv}")
        required = {"image_name", "path"}
        missing = required.difference(fieldnames)
        if missing:
            fail(f"{test_images_tsv} is missing columns: {', '.join(sorted(missing))}")

        image_paths = []
        for row_number, row in enumerate(reader, start=2):
            path = (row.get("path") or "").strip()
            if not path:
                fail(f"missing path in {test_images_tsv} row {row_number}")
            image_paths.append(path)

    if not image_paths:
        fail(f"no test image paths found in {test_images_tsv}")
    return image_paths


def choose_model(run_dir: Path, explicit_model: Path | None) -> Path:
    if explicit_model is not None:
        if not explicit_model.is_file():
            fail(f"model file not found: {explicit_model}")
        return explicit_model

    models_dir = run_dir / "models"
    candidates = sorted(
        models_dir.glob("*.sav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        fail(f"no .sav model found in {models_dir}; run training first or pass --model")
    return candidates[0]


def run_command(
    command: list[str], log_path: Path, dry_run: bool, stdout_path: Path | None = None
) -> None:
    printable = " ".join(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n$ {printable}\n")
        print(printable)
        if dry_run:
            log.write("[dry-run] not executed\n")
            return

        if stdout_path is None:
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            if result.stdout:
                print(result.stdout, end="")
                log.write(result.stdout)
        else:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with stdout_path.open("w") as stdout_handle:
                result = subprocess.run(
                    command, text=True, stdout=stdout_handle, stderr=subprocess.PIPE
                )
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
                log.write(result.stderr)

        if result.returncode != 0:
            fail(f"command failed with exit code {result.returncode}; see {log_path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    metadata_dir = run_dir / "metadata"
    predictions_path = args.predictions or (
        run_dir / "predictions" / "test_predictions.tab"
    )
    pr_curve_path = args.pr_curve or (run_dir / "eval" / "pr_curve.tsv")
    log_path = run_dir / "eval" / "topaz_extract_eval.log"
    test_images_tsv = metadata_dir / "test_images.tsv"
    test_particles_tsv = metadata_dir / "test_particles.tsv"
    test_particles_coord = metadata_dir / "test_particles.tab"

    if not args.topaz_bin.is_file():
        fail(f"Topaz executable not found: {args.topaz_bin}")
    if not test_particles_tsv.is_file():
        fail(f"test particle metadata not found: {test_particles_tsv}")

    image_paths = read_test_image_paths(test_images_tsv)
    model_path = choose_model(run_dir, args.model)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
    test_particles_coord.write_text(test_particles_tsv.read_text())

    extract_command = [
        str(args.topaz_bin),
        "extract",
        "-m",
        str(model_path),
        "-r",
        str(args.radius),
        "-t",
        str(args.threshold),
        "-o",
        str(predictions_path),
        *image_paths,
    ]
    run_command(extract_command, log_path, args.dry_run)

    if not args.skip_pr:
        pr_command = [
            str(args.topaz_bin),
            "precision_recall_curve",
            "--predicted",
            str(predictions_path),
            "--targets",
            str(test_particles_coord),
            "-r",
            str(args.radius),
        ]
        run_command(pr_command, log_path, args.dry_run, stdout_path=pr_curve_path)

    manifest = {
        "topaz_bin": str(args.topaz_bin),
        "run_dir": str(run_dir),
        "model": str(model_path),
        "radius": args.radius,
        "threshold": args.threshold,
        "num_test_images": len(image_paths),
        "predictions": str(predictions_path),
        "test_particles_coord": str(test_particles_coord),
        "pr_curve": None if args.skip_pr else str(pr_curve_path),
        "dry_run": args.dry_run,
    }
    manifest_path = run_dir / "eval" / "extract_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
