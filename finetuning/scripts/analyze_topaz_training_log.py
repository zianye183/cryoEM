#!/usr/bin/env python3
"""Parse Topaz fine-tuning logs and plot train/validation loss curves.

The installed Topaz version may change its console wording, so the parser is
intentionally permissive. It looks for epoch numbers plus loss-like key/value
pairs containing train/training and val/valid/validation/holdout/test labels.
Unknown formats are preserved in the raw log; unsupported lines are skipped.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from finetuning_common import write_json

EPOCH_RE = re.compile(r"\b(?:epoch|epochs?)\s*[:=]?\s*\[?\s*(\d+)", re.IGNORECASE)
TOPAZ_TABLE_ROW_RE = re.compile(
    r"^#?\s*\[(\d+)/(\d+)\]\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
LOSS_PAIR_RE = re.compile(
    r"\b([A-Za-z_/-]*(?:loss|mse|error)[A-Za-z_/-]*)\b\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
LABELED_VALUE_RE = re.compile(
    r"\b(train(?:ing)?|val(?:id(?:ation)?)?|validation|holdout|test)\b[^\n:=]{0,40}?"
    r"(?:loss|mse|error)?\s*[:=]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", required=True, help="Topaz stdout/stderr training log."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for loss_history.tsv, loss_history.json, and loss_curves.png.",
    )
    parser.add_argument(
        "--plot-file",
        default="loss_curves.png",
        help="Plot filename within --output-dir. Set to empty to skip plotting.",
    )
    return parser.parse_args()


def classify_metric(label: str) -> str | None:
    text = label.lower().replace("-", "_").replace("/", "_")
    if "holdout" in text or "valid" in text or text.startswith("val") or "test" in text:
        return "val_loss"
    if "train" in text or "training" in text:
        return "train_loss"
    return None


def parse_log(log_path: Path) -> list[dict[str, Any]]:
    current_epoch: int | None = None
    by_epoch: dict[int, dict[str, Any]] = defaultdict(dict)
    sequential_epoch = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            table_match = TOPAZ_TABLE_ROW_RE.search(line)
            if table_match:
                epoch = int(table_match.group(1))
                row = by_epoch[epoch]
                row.setdefault("epoch", epoch)
                row.setdefault("first_log_line", line_number)
                row["train_loss"] = float(table_match.group(3))
                row["val_loss"] = float(table_match.group(4))
                row["train_loss_source"] = "topaz_table_train_loss"
                row["val_loss_source"] = "topaz_table_val_loss"
                row["last_log_line"] = line_number
                current_epoch = None
                continue

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            elif current_epoch is None and (
                "loss" in line.lower() or "mse" in line.lower()
            ):
                # Fallback for logs that print one aggregate loss line per epoch
                # without an explicit epoch token. Better an ordered curve than no
                # curve, and the raw log remains the source of truth.
                sequential_epoch += 1
                current_epoch = sequential_epoch

            if current_epoch is None:
                continue

            row = by_epoch[current_epoch]
            row.setdefault("epoch", current_epoch)
            row.setdefault("first_log_line", line_number)

            for label, value_text in LOSS_PAIR_RE.findall(line):
                metric = classify_metric(label)
                if metric:
                    row[metric] = float(value_text)
                    row[f"{metric}_source"] = label
                    row["last_log_line"] = line_number

            for label, value_text in LABELED_VALUE_RE.findall(line):
                metric = classify_metric(label)
                if metric:
                    row[metric] = float(value_text)
                    row[f"{metric}_source"] = label
                    row["last_log_line"] = line_number

            # Avoid assigning all later unlabeled loss lines to the previous epoch.
            if not epoch_match and sequential_epoch:
                current_epoch = None

    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    return [row for row in rows if "train_loss" in row or "val_loss" in row]


def write_loss_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_loss_source",
        "val_loss_source",
        "first_log_line",
        "last_log_line",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_losses(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [row["epoch"] for row in rows]
    has_any = False
    plt.figure(figsize=(7, 4.5))
    for key, label in [
        ("train_loss", "train loss"),
        ("val_loss", "validation/holdout loss"),
    ]:
        points = [(row["epoch"], row[key]) for row in rows if key in row]
        if not points:
            continue
        has_any = True
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=label)
    if not has_any:
        plt.close()
        return False
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Topaz fine-tuning loss")
    if len(epochs) > 1:
        plt.xticks(sorted(set(epochs)))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
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

    plotted = False
    if args.plot_file:
        plotted = plot_losses(output_dir / args.plot_file, rows)

    print(f"Parsed epochs with losses: {len(rows)}")
    print(f"Loss TSV: {output_dir / 'loss_history.tsv'}")
    if plotted:
        print(f"Loss plot: {output_dir / args.plot_file}")
    elif args.plot_file:
        print(
            "Loss plot not written; matplotlib may be unavailable or no losses were parsed."
        )


if __name__ == "__main__":
    main()
