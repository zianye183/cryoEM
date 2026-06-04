#!/usr/bin/env python3
"""Launch Topaz Noise2Noise fine-tuning from train/validation manifests.

This project-local Topaz install exposes denoising training through
`topaz denoise --dir-a ... --dir-b ...`, but that CLI only supports a random
`--holdout` split over individual directional examples. That is unsafe when
explicitly duplicating even->odd and odd->even examples, because opposite
directions from the same micrograph can land in train vs validation.

This launcher therefore prepares separate train/validation symlink directories
from already-split manifests and, by default, trains via Topaz's Python API with
explicit train and validation datasets. Source MRC files are never copied or
modified.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_topaz_training_log import parse_log, plot_losses, write_loss_tsv
from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RESULTS_DIR,
    file_sha256,
    quote_command,
    read_tsv,
    require_executable,
    validate_readable_file,
    write_command_log,
    write_json,
)

KNOWN_TOPAZ_MODEL_NAMES = ("unet", "unet-small", "fcnn", "affine", "unet-v0.2.1")
GENERATED_OUTPUT_NAMES = {
    "checkpoints",
    "commands.txt",
    "run_metadata.json",
    "logs",
    "loss_history.tsv",
    "loss_history.json",
    "loss_curves.png",
    "topaz_train_pairs.tsv",
    "topaz_val_pairs.tsv",
    "train_a",
    "train_b",
    "val_a",
    "val_b",
    "denoised_after_training",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "train_pairs.tsv"),
        help="Training split TSV with even_path and odd_path columns.",
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "val_pairs.tsv"),
        help="Validation split TSV used for explicit Topaz API validation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "topaz_finetuned_pilot"),
        help="Output directory for symlinked pair dirs, checkpoints, and logs.",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU id; use -1 for CPU.")
    parser.add_argument(
        "--arch",
        default="unet",
        choices=KNOWN_TOPAZ_MODEL_NAMES,
        help=(
            "Topaz denoising model name used to initialize training. In this Topaz "
            "install, training initialization is controlled by this value."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop", type=int, default=800)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--criteria",
        default="L2",
        choices=("L0", "L1", "L2"),
        help="Training loss passed to Topaz Denoise.train.",
    )
    parser.add_argument(
        "--optim",
        default="adagrad",
        choices=("adam", "adagrad", "sgd"),
        help="Optimizer passed to Topaz Denoise.train.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers passed to Topaz Denoise.train.",
    )
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Checkpoint save prefix. Defaults to <output-dir>/checkpoints/topaz_finetuned.",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create both even->odd and odd->even examples within each already-split "
            "manifest. This is safe here because train and validation dirs are built "
            "separately before training."
        ),
    )
    parser.add_argument("--train-dir-a-name", default="train_a")
    parser.add_argument("--train-dir-b-name", default="train_b")
    parser.add_argument("--val-dir-a-name", default="val_a")
    parser.add_argument("--val-dir-b-name", default="val_b")
    parser.add_argument(
        "--validate-mrc-metadata",
        action="store_true",
        help="If mrcfile is installed, validate even/odd dimensions and pixel sizes before training.",
    )
    parser.add_argument(
        "--cli-holdout",
        type=float,
        default=None,
        help=(
            "Advanced escape hatch: run `topaz denoise --dir-a/--dir-b` with this holdout instead "
            "of explicit API validation. Not safe with --bidirectional."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument for CLI mode only; repeat as needed.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Training stdout/stderr log. Defaults to <output-dir>/logs/topaz_train.log.",
    )
    parser.add_argument(
        "--skip-loss-graph",
        action="store_true",
        help="Do not parse the training log into loss_history.tsv/loss_curves.png after training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_symlink(target: Path, link: Path, *, overwrite: bool) -> None:
    """Create a symlink, guarding against accidental replacement."""
    if link.exists() or link.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Symlink already exists: {link}")
        link.unlink()
    link.symlink_to(target)


def clean_generated_outputs(output_dir: Path) -> None:
    """Remove only this launcher's known generated artifacts from an output dir."""
    for name in GENERATED_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def ensure_output_dir(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contents = [entry for entry in path.iterdir() if entry.name != ".gitkeep"]
    if not contents:
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to replace this script's generated artifacts."
        )
    if dry_run:
        return
    clean_generated_outputs(path)


def _required(row: dict[str, str], column: str, *, manifest_path: Path) -> str:
    value = row.get(column, "")
    if not value:
        raise ValueError(f"Missing required column value `{column}` in {manifest_path}")
    return value


def _mrc_shape_and_pixel_size(
    path: Path,
) -> tuple[tuple[int, int], float | None] | None:
    try:
        import mrcfile  # type: ignore
    except ImportError:
        return None

    with mrcfile.open(path, permissive=True) as mrc:
        shape = tuple(int(dim) for dim in mrc.data.shape)
        if len(shape) < 2:
            raise ValueError(f"MRC data is not at least 2D: {path} shape={shape}")
        image_shape = (shape[-2], shape[-1])
        try:
            pixel_size = float(mrc.voxel_size.x)
        except Exception:
            pixel_size = None
        if pixel_size is not None and pixel_size <= 0:
            pixel_size = None
    return image_shape, pixel_size


def validate_pair_metadata(
    even_path: Path, odd_path: Path, *, crop: int
) -> dict[str, Any]:
    """Validate MRC pair compatibility when mrcfile is available."""
    even_meta = _mrc_shape_and_pixel_size(even_path)
    odd_meta = _mrc_shape_and_pixel_size(odd_path)
    if even_meta is None or odd_meta is None:
        return {"metadata_checked": False}

    even_shape, even_pixel = even_meta
    odd_shape, odd_pixel = odd_meta
    if even_shape != odd_shape:
        raise ValueError(
            f"Shape mismatch: {even_path} {even_shape} vs {odd_path} {odd_shape}"
        )
    if min(even_shape) < crop:
        raise ValueError(
            f"Crop {crop} exceeds image shape {even_shape} for {even_path}"
        )
    if (
        even_pixel is not None
        and odd_pixel is not None
        and abs(even_pixel - odd_pixel) > 1e-6
    ):
        raise ValueError(
            f"Pixel size mismatch: {even_path} {even_pixel} vs {odd_path} {odd_pixel}"
        )
    return {
        "metadata_checked": True,
        "shape_y": even_shape[0],
        "shape_x": even_shape[1],
        "pixel_size_a": even_pixel,
    }


def prepare_pair_dirs(
    manifest_path: Path,
    dir_a: Path,
    dir_b: Path,
    *,
    bidirectional: bool,
    overwrite: bool,
    crop: int,
    validate_mrc_metadata: bool,
) -> dict[str, Any]:
    """Create matched Topaz `dir_a/dir_b` symlink directories from one split manifest."""
    rows = read_tsv(manifest_path)
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    directional_count = 0
    checked_metadata_count = 0
    for row_index, row in enumerate(rows):
        micrograph_id = row.get("micrograph_id") or f"row_{row_index:06d}"
        even_path = validate_readable_file(
            _required(row, "even_path", manifest_path=manifest_path), label="even_path"
        )
        odd_path = validate_readable_file(
            _required(row, "odd_path", manifest_path=manifest_path), label="odd_path"
        )

        if validate_mrc_metadata:
            metadata = validate_pair_metadata(even_path, odd_path, crop=crop)
            if metadata.get("metadata_checked"):
                checked_metadata_count += 1

        pair_name = f"{row_index:06d}_{micrograph_id}_even_to_odd.mrc"
        safe_symlink(even_path, dir_a / pair_name, overwrite=overwrite)
        safe_symlink(odd_path, dir_b / pair_name, overwrite=overwrite)
        directional_count += 1

        if bidirectional:
            pair_name = f"{row_index:06d}_{micrograph_id}_odd_to_even.mrc"
            safe_symlink(odd_path, dir_a / pair_name, overwrite=overwrite)
            safe_symlink(even_path, dir_b / pair_name, overwrite=overwrite)
            directional_count += 1

    if directional_count == 0:
        raise ValueError(f"No pairs prepared from {manifest_path}")
    return {
        "manifest_pairs": len(rows),
        "directional_examples": directional_count,
        "metadata_checked_pairs": checked_metadata_count,
    }


def build_cli_command(
    args: argparse.Namespace,
    train_a: Path,
    train_b: Path,
    output_dir: Path,
    save_prefix: Path,
) -> list[str]:
    if args.bidirectional:
        raise ValueError(
            "CLI mode with --bidirectional is unsafe because Topaz holdout is not grouped by micrograph."
        )
    if args.cli_holdout is None or args.cli_holdout <= 0:
        raise ValueError(
            "CLI mode requires --cli-holdout > 0 because this Topaz trainer expects non-empty validation data."
        )
    command = [
        "topaz",
        "denoise",
        "--device",
        str(args.device),
        "--dir-a",
        str(train_a),
        "--dir-b",
        str(train_b),
        "--method",
        "noise2noise",
        "--arch",
        args.arch,
        "--lr",
        str(args.learning_rate),
        "--num-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--crop",
        str(args.crop),
        "--holdout",
        str(args.cli_holdout),
        "--save-prefix",
        str(save_prefix),
        "--save-interval",
        str(args.save_interval),
    ]
    command.extend(args.extra_arg)
    return command


def api_command_for_log(script_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        str(script_path),
        "--train-manifest",
        str(Path(args.train_manifest).expanduser().resolve()),
        "--val-manifest",
        str(Path(args.val_manifest).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--device",
        str(args.device),
        "--arch",
        args.arch,
        "--learning-rate",
        str(args.learning_rate),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--crop",
        str(args.crop),
        "--save-interval",
        str(args.save_interval),
        "--criteria",
        args.criteria,
        "--optim",
        args.optim,
        "--num-workers",
        str(args.num_workers),
    ]
    command.append("--bidirectional" if args.bidirectional else "--no-bidirectional")
    if args.save_prefix:
        command.extend(
            ["--save-prefix", str(Path(args.save_prefix).expanduser().resolve())]
        )
    if args.validate_mrc_metadata:
        command.append("--validate-mrc-metadata")
    if args.overwrite:
        command.append("--overwrite")
    return command


class Tee:
    """Write stream output to both terminal and a log file."""

    def __init__(self, *streams: Any, normalize_carriage_returns: bool = False) -> None:
        self.streams = streams
        self.normalize_carriage_returns = normalize_carriage_returns

    def write(self, text: str) -> int:
        output_text = (
            text.replace("\r", "\n") if self.normalize_carriage_returns else text
        )
        for stream in self.streams:
            stream.write(output_text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_cli_command_logged(command: list[str], log_path: Path) -> None:
    """Run a subprocess while teeing stdout/stderr to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(quote_command(command))
        handle.write("\n")
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def write_loss_outputs(log_path: Path, output_dir: Path) -> None:
    """Parse the training log and write loss TSV/JSON/PNG artifacts."""
    rows = parse_log(log_path)
    write_loss_tsv(output_dir / "loss_history.tsv", rows)
    write_json(
        output_dir / "loss_history.json",
        {
            "log": str(log_path),
            "parsed_epoch_count": len(rows),
            "rows": rows,
            "parser_note": "Permissive Topaz log parser; verify against raw log for unfamiliar Topaz output formats.",
        },
    )
    plotted = plot_losses(output_dir / "loss_curves.png", rows)
    print(f"Parsed epochs with losses: {len(rows)}")
    print(f"Loss TSV: {output_dir / 'loss_history.tsv'}")
    if plotted:
        print(f"Loss plot: {output_dir / 'loss_curves.png'}")
    else:
        print(
            "Loss plot not written; matplotlib may be unavailable or no losses were parsed."
        )


def run_topaz_api(
    args: argparse.Namespace,
    train_a: Path,
    train_b: Path,
    val_a: Path,
    val_b: Path,
    save_prefix: Path,
) -> None:
    """Train Topaz with explicit validation directories, avoiding random holdout leakage."""
    from topaz.cuda import set_device  # type: ignore[import-not-found]
    from topaz.denoise import Denoise  # type: ignore[import-not-found]
    from topaz.denoising.datasets import PairedImages  # type: ignore[import-not-found]
    from topaz.torch import set_num_threads  # type: ignore[import-not-found]

    set_num_threads(0)
    use_cuda = set_device(args.device)
    print(f"# using device={args.device} with cuda={use_cuda}")

    train_files = sorted(train_a.glob("*.mrc"))
    val_files = sorted(val_a.glob("*.mrc"))
    if not train_files:
        raise ValueError(f"No training symlinks found in {train_a}")
    if not val_files:
        raise ValueError(f"No validation symlinks found in {val_a}")

    train_x = [str(path) for path in train_files]
    train_y = [str(train_b / path.name) for path in train_files]
    val_x = [str(path) for path in val_files]
    val_y = [str(val_b / path.name) for path in val_files]
    missing_targets = [path for path in train_y + val_y if not Path(path).is_file()]
    if missing_targets:
        raise FileNotFoundError(
            f"Missing paired target symlink(s), first: {missing_targets[0]}"
        )

    train_data = PairedImages(train_x, train_y, crop=args.crop, xform=True)
    val_data = PairedImages(val_x, val_y, crop=args.crop, xform=True)
    denoiser = Denoise(args.arch, use_cuda, dims=2)
    denoiser.train(
        train_data,
        val_data,
        loss_fn=args.criteria,
        optim=args.optim,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        shuffle=True,
        num_workers=args.num_workers,
        verbose=True,
        save_best=True,
        save_interval=args.save_interval,
        save_prefix=str(save_prefix),
    )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()

    if args.cli_holdout is not None and not args.dry_run:
        require_executable("topaz")

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    train_manifest = Path(args.train_manifest).expanduser().resolve()
    val_manifest = Path(args.val_manifest).expanduser().resolve()
    train_a = output_dir / args.train_dir_a_name
    train_b = output_dir / args.train_dir_b_name
    val_a = output_dir / args.val_dir_a_name
    val_b = output_dir / args.val_dir_b_name
    save_prefix = (
        Path(args.save_prefix).expanduser().resolve()
        if args.save_prefix
        else output_dir / "checkpoints" / "topaz_finetuned"
    )
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else output_dir / "logs" / "topaz_train.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_summary = prepare_pair_dirs(
        train_manifest,
        train_a,
        train_b,
        bidirectional=args.bidirectional,
        overwrite=args.overwrite or args.dry_run,
        crop=args.crop,
        validate_mrc_metadata=args.validate_mrc_metadata,
    )
    val_summary = prepare_pair_dirs(
        val_manifest,
        val_a,
        val_b,
        bidirectional=args.bidirectional,
        overwrite=args.overwrite or args.dry_run,
        crop=args.crop,
        validate_mrc_metadata=args.validate_mrc_metadata,
    )

    if args.cli_holdout is None:
        command = api_command_for_log(script_path, args)
        execution_mode = "topaz_python_api_explicit_validation"
    else:
        command = build_cli_command(args, train_a, train_b, output_dir, save_prefix)
        execution_mode = "topaz_cli_random_holdout"

    write_command_log(output_dir / "commands.txt", [command])
    write_json(
        output_dir / "run_metadata.json",
        {
            "execution_mode": execution_mode,
            "topaz_model_initialization": args.arch,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": file_sha256(train_manifest),
            "val_manifest": str(val_manifest),
            "val_manifest_sha256": file_sha256(val_manifest),
            "train_summary": train_summary,
            "val_summary": val_summary,
            "bidirectional": args.bidirectional,
            "bidirectional_split_policy": "directions are duplicated only within already-separated train/validation manifests",
            "train_a": str(train_a),
            "train_b": str(train_b),
            "val_a": str(val_a),
            "val_b": str(val_b),
            "save_prefix": str(save_prefix),
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "crop": args.crop,
            "criteria": args.criteria,
            "optim": args.optim,
            "save_interval": args.save_interval,
            "command": command,
            "training_log": str(log_path),
            "loss_history_tsv": str(output_dir / "loss_history.tsv"),
            "loss_curves_png": str(output_dir / "loss_curves.png"),
            "dry_run": args.dry_run,
            "note": (
                "Default mode uses Topaz Python API with explicit validation directories to avoid "
                "reverse-direction train/validation leakage from CLI random holdout."
            ),
        },
    )

    print(f"Prepared train --dir-a: {train_a}")
    print(f"Prepared train --dir-b: {train_b}")
    print(f"Prepared val   --dir-a: {val_a}")
    print(f"Prepared val   --dir-b: {val_b}")
    print(f"Train manifest pairs:      {train_summary['manifest_pairs']}")
    print(f"Train directional examples:{train_summary['directional_examples']}")
    print(f"Val manifest pairs:        {val_summary['manifest_pairs']}")
    print(f"Val directional examples:  {val_summary['directional_examples']}")
    print(f"Output directory:          {output_dir}")
    print(f"Training log:              {log_path}")
    print(quote_command(command))

    if args.dry_run:
        print("Dry run only; training was not executed.")
        return

    if args.cli_holdout is None:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(quote_command(command))
            log_handle.write("\n")
            tee_stdout = Tee(sys.stdout, log_handle, normalize_carriage_returns=True)
            tee_stderr = Tee(sys.stderr, log_handle, normalize_carriage_returns=True)
            with (
                contextlib.redirect_stdout(tee_stdout),
                contextlib.redirect_stderr(tee_stderr),
            ):
                run_topaz_api(args, train_a, train_b, val_a, val_b, save_prefix)
    else:
        run_cli_command_logged(command, log_path)

    if not args.skip_loss_graph:
        write_loss_outputs(log_path, output_dir)


if __name__ == "__main__":
    main()
