#!/usr/bin/env python3
"""Compare Topaz denoising checkpoints on explicit validation pairs.

This script evaluates pretrained models and saved checkpoints outside the
training loop. For each model it denoises validation even/odd halves, computes
Noise2Noise-style cross losses and split consistency, writes metrics tables, and
creates a compact Markdown report. It does not modify source micrographs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RESULTS_DIR,
    ensure_empty_or_create_output_dir,
    paths_from_manifest,
    quote_command,
    read_tsv,
    require_executable,
    validate_readable_file,
    write_command_log,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "val_pairs.tsv"),
        help="Validation split TSV with even_path and odd_path columns.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=[],
        help="Glob for checkpoint files. Repeatable. Quote shell wildcards.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Explicit checkpoint path. Repeatable. Explicit checkpoints are always evaluated.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["key", "all"],
        default="key",
        help=(
            "How to select checkpoints matched by --checkpoint-glob. "
            "'key' evaluates epoch01, the checkpoint with lowest validation loss, and latest epoch. "
            "'all' evaluates every matched checkpoint. Default: key."
        ),
    )
    parser.add_argument(
        "--include-pretrained",
        action="append",
        default=["unet"],
        help="Pretrained Topaz model name to include. Repeatable. Use --no-pretrained to omit defaults.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not include the default pretrained unet baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "topaz_checkpoint_comparison_pilot"),
        help="Output directory for checkpoint comparison artifacts.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--patch-size", default="4000")
    parser.add_argument("--patch-padding", default="500")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Evaluate only the first N validation pairs for smoke tests.",
    )
    parser.add_argument(
        "--highfreq-fraction",
        type=float,
        default=0.35,
        help="Lower normalized radial-frequency cutoff for high-frequency power summary.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument for `topaz denoise`; repeat as needed.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="After checkpoint evaluation, render side-by-side PNG panels for selected train/validation pairs.",
    )
    parser.add_argument(
        "--train-visual-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "train_pairs.tsv"),
        help="Train split manifest used when --visualize is set.",
    )
    parser.add_argument(
        "--visual-count",
        type=int,
        default=5,
        help="Number of train and validation pairs to render when --visualize is set.",
    )
    parser.add_argument(
        "--visual-output-dir",
        default=None,
        help="Visualization output directory. Default: <output-dir>/visualization.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def model_slug(model: str) -> str:
    path = Path(model)
    text = path.stem if path.suffix else model
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch(\d+)", path.stem)
    return int(match.group(1)) if match else None


def loss_history_for_checkpoint(path: Path) -> Path:
    # Expected layout: <run>/checkpoints/topaz_finetuned_epochNN.sav.
    return path.parent.parent / "loss_history.tsv"


def best_val_checkpoint(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    loss_path = loss_history_for_checkpoint(candidates[0])
    if not loss_path.is_file():
        return None
    try:
        rows = read_tsv(loss_path)
    except Exception:
        return None

    best_epoch: int | None = None
    best_loss = math.inf
    for row in rows:
        try:
            loss = float(row.get("val_loss", ""))
            # Training log epochs are zero-based; checkpoint files are one-based.
            checkpoint_epoch_number = int(row["epoch"]) + 1
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_epoch = checkpoint_epoch_number
    if best_epoch is None:
        return None

    for checkpoint in candidates:
        if checkpoint_epoch(checkpoint) == best_epoch:
            return checkpoint
    return None


def select_key_checkpoints(matches: list[Path]) -> list[Path]:
    if not matches:
        return []
    with_epochs = [path for path in matches if checkpoint_epoch(path) is not None]
    if not with_epochs:
        return matches

    selected: list[Path] = []
    epoch_one = [path for path in with_epochs if checkpoint_epoch(path) == 1]
    if epoch_one:
        selected.append(epoch_one[0])

    best = best_val_checkpoint(with_epochs)
    if best is not None:
        selected.append(best)

    latest = max(with_epochs, key=lambda path: checkpoint_epoch(path) or -1)
    selected.append(latest)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def discover_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if not args.no_pretrained:
        models.extend(args.include_pretrained)
    for checkpoint in args.checkpoint:
        models.append(str(Path(checkpoint).expanduser().resolve()))
    for pattern in args.checkpoint_glob:
        matches = sorted(
            Path(match).resolve()
            for match in glob.glob(str(Path(pattern).expanduser()))
        )
        selected = (
            matches
            if args.checkpoint_selection == "all"
            else select_key_checkpoints(matches)
        )
        if args.checkpoint_selection == "key" and matches:
            print(
                "Selected key checkpoints from glob: "
                + ", ".join(path.name for path in selected)
            )
        models.extend(str(path) for path in selected)

    unique: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        unique.append(model)
    if not unique:
        raise ValueError("No models/checkpoints requested for comparison")
    return unique


def build_denoise_command(
    args: argparse.Namespace,
    model: str,
    input_paths: list[Path],
    output_dir: Path,
) -> list[str]:
    wrapper = Path(__file__).with_name("topaz_denoise_model.py")
    command = [
        sys.executable,
        str(wrapper),
        "--model",
        model,
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
    command.extend(str(path) for path in input_paths)
    return command


def run_command_logged(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool,
    current_label: str,
) -> None:
    print(f"current: {current_label}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(quote_command(command))
            handle.write("\nDry run only; command was not executed.\n")
        print("Dry run only; command was not executed.")
        return
    with log_path.open("w", encoding="utf-8") as handle:
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


def load_mrc(path: Path) -> Any:
    try:
        import mrcfile  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mrcfile is required for checkpoint metrics. Activate the Topaz/fine-tuning environment."
        ) from exc
    with mrcfile.open(path, permissive=True) as mrc:
        import numpy as np

        return np.asarray(mrc.data, dtype=np.float32).copy()


def normalize_image(image: Any) -> Any:
    import numpy as np

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(arr[finite].mean())
    std = float(arr[finite].std())
    if std <= 0:
        std = 1.0
    out = (arr - mean) / std
    out[~finite] = 0
    return out.astype(np.float32)


def mse(a: Any, b: Any) -> float:
    import numpy as np

    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(diff * diff))


def pearson_corr(a: Any, b: Any) -> float:
    import numpy as np

    ax = np.asarray(a, dtype=np.float32).ravel()
    bx = np.asarray(b, dtype=np.float32).ravel()
    ax = ax - ax.mean()
    bx = bx - bx.mean()
    denom = float(np.sqrt(np.sum(ax * ax) * np.sum(bx * bx)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(ax * bx) / denom)


def highfreq_power_ratio(raw: Any, denoised: Any, cutoff: float) -> float:
    import numpy as np

    raw_arr = normalize_image(raw)
    den_arr = normalize_image(denoised)
    fy = np.fft.fftfreq(raw_arr.shape[0])
    fx = np.fft.fftfreq(raw_arr.shape[1])
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    max_radius = float(radius.max()) or 1.0
    mask = radius >= cutoff * max_radius
    raw_power = np.abs(np.fft.fft2(raw_arr)) ** 2
    den_power = np.abs(np.fft.fft2(den_arr)) ** 2
    raw_mean = float(raw_power[mask].mean())
    if raw_mean <= 0:
        return float("nan")
    return float(den_power[mask].mean() / raw_mean)


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float((finite[mid - 1] + finite[mid]) / 2)


def find_output_file(output_dir: Path, input_path: Path) -> Path:
    candidates = [
        output_dir / input_path.name,
        output_dir / f"{input_path.stem}.mrc",
        output_dir / f"{input_path.stem}_denoised.mrc",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(output_dir.glob(f"{input_path.stem}*.mrc"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find denoised output for {input_path} in {output_dir}"
    )


def evaluate_model(
    args: argparse.Namespace,
    model: str,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    slug = model_slug(model)
    model_dir = output_dir / "model_outputs" / slug
    even_output = model_dir / "denoised_even"
    odd_output = model_dir / "denoised_odd"
    log_dir = output_dir / "logs"
    even_output.mkdir(parents=True, exist_ok=True)
    odd_output.mkdir(parents=True, exist_ok=True)

    even_paths = [
        validate_readable_file(row["even_path"], label="even_path") for row in rows
    ]
    odd_paths = [
        validate_readable_file(row["odd_path"], label="odd_path") for row in rows
    ]

    commands = [
        build_denoise_command(args, model, even_paths, even_output),
        build_denoise_command(args, model, odd_paths, odd_output),
    ]
    print(f"# current checkpoint: {model}")
    run_command_logged(
        commands[0],
        log_dir / f"{slug}_even.log",
        dry_run=args.dry_run,
        current_label="half1",
    )
    run_command_logged(
        commands[1],
        log_dir / f"{slug}_odd.log",
        dry_run=args.dry_run,
        current_label="half2",
    )

    if args.dry_run:
        return {
            "model": model,
            "model_slug": slug,
            "n_val_pairs": len(rows),
            "status": "dry_run",
            "notes": "Commands were written but not executed.",
        }

    cross_even_odd: list[float] = []
    cross_odd_even: list[float] = []
    consistency: list[float] = []
    raw_to_denoised: list[float] = []
    hf_ratios: list[float] = []

    if len(even_paths) != len(odd_paths):
        raise ValueError(
            f"Mismatched validation path counts: {len(even_paths)} even vs {len(odd_paths)} odd"
        )

    for even_path, odd_path in zip(even_paths, odd_paths):
        den_even_path = find_output_file(even_output, even_path)
        den_odd_path = find_output_file(odd_output, odd_path)

        even = normalize_image(load_mrc(even_path))
        odd = normalize_image(load_mrc(odd_path))
        den_even = normalize_image(load_mrc(den_even_path))
        den_odd = normalize_image(load_mrc(den_odd_path))

        cross_even_odd.append(mse(den_even, odd))
        cross_odd_even.append(mse(den_odd, even))
        consistency.append(pearson_corr(den_even, den_odd))
        raw_to_denoised.extend([mse(den_even, even), mse(den_odd, odd)])
        hf_ratios.extend(
            [
                highfreq_power_ratio(even, den_even, args.highfreq_fraction),
                highfreq_power_ratio(odd, den_odd, args.highfreq_fraction),
            ]
        )

    if len(cross_even_odd) != len(cross_odd_even):
        raise ValueError(
            f"Mismatched cross-loss counts: {len(cross_even_odd)} even→odd vs {len(cross_odd_even)} odd→even"
        )
    cross_mean_values = [(a + b) / 2 for a, b in zip(cross_even_odd, cross_odd_even)]
    return {
        "model": model,
        "model_slug": slug,
        "n_val_pairs": len(rows),
        "mse_deven_vs_odd_mean": mean(cross_even_odd),
        "mse_dodd_vs_even_mean": mean(cross_odd_even),
        "mse_cross_mean": mean(cross_mean_values),
        "mse_cross_median": median(cross_mean_values),
        "corr_deven_dodd_mean": mean(consistency),
        "corr_deven_dodd_median": median(consistency),
        "raw_to_denoised_mse_mean": mean(raw_to_denoised),
        "highfreq_power_ratio_mean": mean(hf_ratios),
        "status": "candidate",
        "notes": "Review visual/artifact behavior before using this checkpoint downstream.",
    }


def write_metrics_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "rank_by_mse",
        "model_slug",
        "model",
        "n_val_pairs",
        "mse_deven_vs_odd_mean",
        "mse_dodd_vs_even_mean",
        "mse_cross_mean",
        "mse_cross_median",
        "corr_deven_dodd_mean",
        "corr_deven_dodd_median",
        "raw_to_denoised_mse_mean",
        "highfreq_power_ratio_mean",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_checkpoint_metrics(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    metric_rows = [row for row in rows if isinstance(row.get("mse_cross_mean"), float)]
    if not metric_rows:
        return False
    labels = [row["model_slug"] for row in metric_rows]
    mse_values = [row["mse_cross_mean"] for row in metric_rows]
    corr_values = [row["corr_deven_dodd_mean"] for row in metric_rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(7, len(labels) * 1.2), 7), sharex=True)
    axes[0].bar(labels, mse_values)
    axes[0].set_ylabel("Cross MSE")
    axes[0].set_title("Validation Noise2Noise-style loss by checkpoint")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, corr_values)
    axes[1].set_ylabel("corr(D_even, D_odd)")
    axes[1].set_xlabel("Model/checkpoint")
    axes[1].grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return True


def write_report(
    path: Path, rows: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    metric_rows = [row for row in rows if isinstance(row.get("mse_cross_mean"), float)]
    recommended = metric_rows[0] if metric_rows else None
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Topaz checkpoint comparison\n\n")
        handle.write(
            f"Validation manifest: `{Path(args.val_manifest).expanduser().resolve()}`\n\n"
        )
        if recommended:
            handle.write(
                f"## Recommended by validation cross-MSE\n\n`{recommended['model_slug']}`\n\n"
            )
            handle.write(
                "This is a numeric recommendation only. Review visual panels/artifacts and downstream MiLoPYP behavior before scaling.\n\n"
            )
        else:
            handle.write(
                "No numeric recommendation was produced. This may have been a dry run.\n\n"
            )
        handle.write("## Metrics\n\n")
        handle.write(
            "| rank | model | cross MSE | corr(D_even,D_odd) | highfreq ratio | status |\n"
        )
        handle.write("|---:|---|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| {row.get('rank_by_mse', '')} | `{row.get('model_slug', '')}` | "
                f"{row.get('mse_cross_mean', '')} | {row.get('corr_deven_dodd_mean', '')} | "
                f"{row.get('highfreq_power_ratio_mean', '')} | {row.get('status', '')} |\n"
            )
        handle.write("\n## Notes\n\n")
        handle.write(
            "- Primary metric: mean of `MSE(D(even), odd)` and `MSE(D(odd), even)` on explicit validation pairs.\n"
        )
        handle.write(
            "- Consistency is useful only if it reflects real shared signal; suspicious background correlation still needs visual/masked review.\n"
        )
        handle.write(
            "- High-frequency ratio is a coarse artifact/smoothing signal, not a standalone selection metric.\n"
        )


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_rows = [row for row in rows if isinstance(row.get("mse_cross_mean"), float)]
    non_metric_rows = [
        row for row in rows if not isinstance(row.get("mse_cross_mean"), float)
    ]
    metric_rows.sort(key=lambda row: row["mse_cross_mean"])
    for index, row in enumerate(metric_rows, start=1):
        row["rank_by_mse"] = index
    for row in non_metric_rows:
        row["rank_by_mse"] = ""
    return metric_rows + non_metric_rows


def run_visualization(args: argparse.Namespace, output_dir: Path) -> None:
    if args.dry_run:
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("visualize_topaz_checkpoint_panels.py")),
        "--eval-dir",
        str(output_dir),
        "--val-manifest",
        str(Path(args.val_manifest).expanduser().resolve()),
        "--train-manifest",
        str(Path(args.train_visual_manifest).expanduser().resolve()),
        "--count",
        str(args.visual_count),
        "--device",
        args.device,
        "--patch-size",
        args.patch_size,
        "--patch-padding",
        args.patch_padding,
    ]
    if args.visual_output_dir:
        command.extend(
            ["--output-dir", str(Path(args.visual_output_dir).expanduser().resolve())]
        )
    print("Rendering visualization panels...")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        # Topaz is imported by topaz_denoise_model.py; check metric deps before
        # launching expensive denoising subprocesses so failures are immediate.
        load_mrc
        try:
            import mrcfile  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mrcfile is required for checkpoint metrics. Install it in the active Topaz environment: python -m pip install mrcfile"
            ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        ensure_empty_or_create_output_dir(output_dir, overwrite=args.overwrite)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    val_manifest = Path(args.val_manifest).expanduser().resolve()
    rows = read_tsv(val_manifest)
    if args.limit_pairs is not None:
        rows = rows[: args.limit_pairs]
    if not rows:
        raise ValueError(f"No validation rows available from {val_manifest}")

    # Validate both columns early for clearer failures before Topaz launches.
    paths_from_manifest(rows, "even_path")
    paths_from_manifest(rows, "odd_path")

    models = discover_models(args)
    all_commands: list[list[str]] = []
    metrics: list[dict[str, Any]] = []
    for model in models:
        slug = model_slug(model)
        even_paths = [Path(row["even_path"]).expanduser().resolve() for row in rows]
        odd_paths = [Path(row["odd_path"]).expanduser().resolve() for row in rows]
        all_commands.append(
            build_denoise_command(
                args,
                model,
                even_paths,
                output_dir / "model_outputs" / slug / "denoised_even",
            )
        )
        all_commands.append(
            build_denoise_command(
                args,
                model,
                odd_paths,
                output_dir / "model_outputs" / slug / "denoised_odd",
            )
        )
        metrics.append(evaluate_model(args, model, rows, output_dir))

    ranked = rank_rows(metrics)
    write_command_log(output_dir / "commands.txt", all_commands)
    write_metrics_tsv(output_dir / "checkpoint_metrics.tsv", ranked)
    write_json(
        output_dir / "checkpoint_metrics.json",
        {
            "val_manifest": str(val_manifest),
            "n_val_pairs": len(rows),
            "models": models,
            "metrics": ranked,
            "dry_run": args.dry_run,
        },
    )
    plotted = plot_checkpoint_metrics(output_dir / "checkpoint_metrics.png", ranked)
    write_report(output_dir / "report.md", ranked, args)

    print(f"Metrics TSV: {output_dir / 'checkpoint_metrics.tsv'}")
    print(f"Report:      {output_dir / 'report.md'}")
    if plotted:
        print(f"Plot:        {output_dir / 'checkpoint_metrics.png'}")
    if args.visualize:
        run_visualization(args, output_dir)


if __name__ == "__main__":
    main()
