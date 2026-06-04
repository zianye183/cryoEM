#!/usr/bin/env python3
"""Export DoG pick TSV coordinates to per-micrograph .box files.

The input is the dog_picks.tsv written by evaluate_dog_picking.py. Outputs are
organized as boxes/<method>/<micrograph_stem>.box. Coordinates are converted from
pick centers to top-left box corners using a fixed box size.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, required=True, help="dog_picks.tsv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output boxes directory")
    parser.add_argument("--box-size", type=float, required=True, help="Box size in pixels")
    parser.add_argument(
        "--max-picks-per-micrograph",
        type=int,
        default=None,
        help="Optional maximum picks exported per method/micrograph, preserving rank order",
    )
    parser.add_argument(
        "--include-score",
        action="store_true",
        help="Append detector score as a fifth column after x y w h",
    )
    return parser.parse_args()


def micrograph_stem(name: str) -> str:
    path = Path(name)
    return path.stem if path.suffix else name


def main() -> None:
    args = parse_args()
    if args.box_size <= 0:
        raise SystemExit("--box-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with args.picks.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"micrograph", "method", "rank", "x", "y", "detector_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {args.picks}: {sorted(missing)}")
        for row in reader:
            grouped.setdefault((row["method"], row["micrograph"]), []).append(row)

    written = 0
    half = args.box_size / 2.0
    for (method, micrograph), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["rank"]))
        if args.max_picks_per_micrograph is not None:
            rows = rows[: args.max_picks_per_micrograph]
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_dir / f"{micrograph_stem(micrograph)}.box"
        with out_path.open("w") as handle:
            for row in rows:
                x = float(row["x"]) - half
                y = float(row["y"]) - half
                if args.include_score:
                    handle.write(
                        f"{x:.1f}\t{y:.1f}\t{args.box_size:.1f}\t{args.box_size:.1f}\t{float(row['detector_score']):.6g}\n"
                    )
                else:
                    handle.write(f"{x:.1f}\t{y:.1f}\t{args.box_size:.1f}\t{args.box_size:.1f}\n")
        written += 1

    print(f"Wrote {written} .box files under {args.output_dir}")


if __name__ == "__main__":
    main()
