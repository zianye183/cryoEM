#!/usr/bin/env python3
"""Create reproducible train/validation/test manifests from a pair manifest."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    SPLIT_COLUMNS,
    file_sha256,
    manifest_summary,
    read_tsv,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_MANIFEST_DIR / "all_pairs.tsv"),
        help="Input all-pairs TSV from prepare_pairs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_MANIFEST_DIR),
        help="Directory for train_pairs.tsv, val_pairs.tsv, test_pairs.tsv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Optional sorted-then-shuffled pair limit for the pilot. Use 0 for all rows.",
    )
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument(
        "--visual-count",
        type=int,
        default=12,
        help="Number of held-out test rows to copy into visual_panel.tsv.",
    )
    return parser.parse_args()


def assign_splits(
    rows: list[dict[str, str]],
    *,
    seed: int,
    limit: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    sorted_rows = sorted(rows, key=lambda row: row["micrograph_id"])
    if limit > 0:
        sorted_rows = sorted_rows[:limit]

    required = train_count + val_count + test_count
    if len(sorted_rows) < required:
        raise ValueError(
            f"Need at least {required} rows for requested split, found {len(sorted_rows)}"
        )

    rng = random.Random(seed)
    shuffled = sorted_rows[:]
    rng.shuffle(shuffled)

    train_rows = shuffled[:train_count]
    val_rows = shuffled[train_count : train_count + val_count]
    test_rows = shuffled[train_count + val_count : train_count + val_count + test_count]

    for split_name, split_rows in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        for row in split_rows:
            row["split"] = split_name

    return train_rows, val_rows, test_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = read_tsv(input_path)

    train_rows, val_rows, test_rows = assign_splits(
        rows,
        seed=args.seed,
        limit=args.limit,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )

    train_path = output_dir / "train_pairs.tsv"
    val_path = output_dir / "val_pairs.tsv"
    test_path = output_dir / "test_pairs.tsv"
    visual_path = output_dir / "visual_panel.tsv"
    summary_path = output_dir / "split_summary.json"

    write_tsv(train_path, train_rows, SPLIT_COLUMNS)
    write_tsv(val_path, val_rows, SPLIT_COLUMNS)
    write_tsv(test_path, test_rows, SPLIT_COLUMNS)
    write_tsv(visual_path, test_rows[: args.visual_count], SPLIT_COLUMNS)

    summary = {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "seed": args.seed,
        "limit": args.limit,
        "train": manifest_summary(train_rows),
        "val": manifest_summary(val_rows),
        "test": manifest_summary(test_rows),
        "outputs": {
            "train_pairs": str(train_path),
            "val_pairs": str(val_path),
            "test_pairs": str(test_path),
            "visual_panel": str(visual_path),
        },
    }
    write_json(summary_path, summary)

    print(f"Wrote train split: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote val split:   {val_path} ({len(val_rows)} rows)")
    print(f"Wrote test split:  {test_path} ({len(test_rows)} rows)")
    print(
        f"Wrote visual set:  {visual_path} ({min(args.visual_count, len(test_rows))} rows)"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
