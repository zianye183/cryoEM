#!/usr/bin/env python3
"""Plot identical-display micrograph overlays for DoG particle-picking results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: numpy") from exc

try:
    import mrcfile
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: mrcfile") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: matplotlib") from exc


METHOD_COLORS = {
    "raw": "tab:blue",
    "topaz": "tab:orange",
    "draco": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, required=True, help="dog_picks.tsv")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw micrograph directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Overlay PNG directory")
    parser.add_argument("--glob", default="*.mrc", help="Raw MRC glob")
    parser.add_argument("--limit", type=int, default=None, help="Optional micrograph limit")
    parser.add_argument("--max-picks", type=int, default=300, help="Max plotted picks per method/micrograph")
    parser.add_argument("--marker-size", type=float, default=30.0, help="Scatter marker size")
    parser.add_argument("--downsample", type=int, default=1, help="Integer display downsampling factor")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI")
    parser.add_argument(
        "--contrast-percentiles",
        default="1,99",
        help="Comma-separated display percentiles, shared across overlays per micrograph",
    )
    return parser.parse_args()


def read_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D MRC: {path}; got shape {data.shape}")
    return data


def load_picks(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"micrograph", "method", "rank", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")
        for row in reader:
            grouped.setdefault((row["micrograph"], row["method"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return grouped


def parse_percentiles(text: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError("Expected low,high percentiles such as 1,99")
    return parts[0], parts[1]


def main() -> None:
    args = parse_args()
    if args.downsample < 1:
        raise SystemExit("--downsample must be >= 1")
    low_pct, high_pct = parse_percentiles(args.contrast_percentiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    picks = load_picks(args.picks)
    raw_paths = sorted(args.raw_dir.glob(args.glob))
    if args.limit is not None:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths:
        raise SystemExit(f"No raw MRCs found in {args.raw_dir} with glob {args.glob}")

    written = 0
    for raw_path in raw_paths:
        image = read_mrc(raw_path)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            continue
        vmin, vmax = np.percentile(finite, [low_pct, high_pct])
        display = image[:: args.downsample, :: args.downsample]
        methods = sorted(
            {method for (micrograph, method) in picks if micrograph == raw_path.name},
            key=lambda name: (name != "raw", name),
        )
        if not methods:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        for method in methods:
            rows = picks.get((raw_path.name, method), [])[: args.max_picks]
            if not rows:
                continue
            xs = np.asarray([float(row["x"]) for row in rows]) / args.downsample
            ys = np.asarray([float(row["y"]) for row in rows]) / args.downsample
            ax.scatter(
                xs,
                ys,
                s=args.marker_size,
                facecolors="none",
                edgecolors=METHOD_COLORS.get(method, None),
                linewidths=0.8,
                label=f"{method} n={len(rows)}",
            )
        ax.set_title(raw_path.name)
        ax.set_axis_off()
        ax.legend(loc="upper right", fontsize="small", framealpha=0.75)
        fig.tight_layout(pad=0)
        out_path = args.output_dir / f"{raw_path.stem}_dog_overlay.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        written += 1

    print(f"Wrote {written} overlay PNGs under {args.output_dir}")


if __name__ == "__main__":
    main()
