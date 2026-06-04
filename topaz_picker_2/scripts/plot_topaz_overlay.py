#!/usr/bin/env python3
"""Plot Topaz target/prediction overlays on exported micrographs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            {key: value.rstrip("\r") for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def normalize_image_name(value: str) -> str:
    return Path(value.rstrip("\r")).stem


def choose_column(row: dict[str, str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise ValueError(f"Could not find {label} column. Available columns: {sorted(row)}")


def group_points(
    path: Path,
    *,
    min_score: float | None = None,
) -> dict[str, list[tuple[float, float, float | None]]]:
    rows = read_table(path)
    if not rows:
        return {}

    first = rows[0]
    image_col = choose_column(first, ("image_name", "image", "micrograph"), "image name")
    x_col = choose_column(first, ("x_coord", "x"), "x coordinate")
    y_col = choose_column(first, ("y_coord", "y"), "y coordinate")
    score_col = next((col for col in ("score", "log_probability", "probability") if col in first), None)

    points: dict[str, list[tuple[float, float, float | None]]] = defaultdict(list)
    for row in rows:
        score = float(row[score_col]) if score_col and row[score_col] else None
        if min_score is not None and score is not None and score < min_score:
            continue
        points[normalize_image_name(row[image_col])].append(
            (float(row[x_col]), float(row[y_col]), score)
        )
    return points


def load_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".mrc", ".mrcs"}:
        try:
            import mrcfile
        except ImportError as exc:
            raise SystemExit(
                "Reading MRC files requires mrcfile. Activate the Topaz environment or install mrcfile there."
            ) from exc
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data)
    else:
        from PIL import Image

        data = np.asarray(Image.open(path))

    if data.ndim > 2:
        data = data[0]
    return np.asarray(data, dtype=np.float32)


def display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1, 99])
    if math.isclose(float(vmin), float(vmax)):
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        return mean - 3 * std, mean + 3 * std
    return float(vmin), float(vmax)


def plot_overlay(
    *,
    image_name: str,
    image_path: Path,
    target_points: list[tuple[float, float, float | None]],
    prediction_points: list[tuple[float, float, float | None]],
    output_path: Path,
    crop_radius: int | None,
) -> None:
    image = load_image(image_path)
    vmin, vmax = display_limits(image)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")

    if target_points:
        xs = [point[0] for point in target_points]
        ys = [point[1] for point in target_points]
        ax.scatter(
            xs,
            ys,
            s=55,
            facecolors="none",
            edgecolors="#ff3b30",
            linewidths=1.2,
            label=f"targets ({len(target_points)})",
        )

    if prediction_points:
        xs = [point[0] for point in prediction_points]
        ys = [point[1] for point in prediction_points]
        ax.scatter(
            xs,
            ys,
            s=38,
            marker="x",
            c="#00d2ff",
            linewidths=1.0,
            label=f"predictions ({len(prediction_points)})",
        )

    if crop_radius and (target_points or prediction_points):
        all_x = [point[0] for point in target_points + prediction_points]
        all_y = [point[1] for point in target_points + prediction_points]
        center_x = sum(all_x) / len(all_x)
        center_y = sum(all_y) / len(all_y)
        ax.set_xlim(max(0, center_x - crop_radius), min(image.shape[1], center_x + crop_radius))
        ax.set_ylim(min(image.shape[0], center_y + crop_radius), max(0, center_y - crop_radius))

    ax.set_title(f"{image_name}\n{image_path}", fontsize=9)
    ax.set_axis_off()
    if target_points or prediction_points:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Topaz image list TSV")
    parser.add_argument("--targets", type=Path, help="Topaz target coord file")
    parser.add_argument("--predictions", type=Path, help="Topaz extract predictions TSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--crop-radius",
        type=int,
        default=None,
        help="Optional pixel radius around labels/predictions for zoomed overlays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_rows = read_table(args.images)
    if not image_rows:
        raise SystemExit(f"No image rows found in {args.images}")

    target_points = group_points(args.targets) if args.targets else {}
    prediction_points = (
        group_points(args.predictions, min_score=args.min_score) if args.predictions else {}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in image_rows:
        image_name = normalize_image_name(row["image_name"])
        image_path = Path(row["path"].rstrip("\r"))
        output_path = args.output_dir / f"{image_name}_overlay.png"
        plot_overlay(
            image_name=image_name,
            image_path=image_path,
            target_points=target_points.get(image_name, []),
            prediction_points=prediction_points.get(image_name, []),
            output_path=output_path,
            crop_radius=args.crop_radius,
        )
        print(f"Wrote {output_path}")
        written += 1
        if written >= args.max_images:
            break


if __name__ == "__main__":
    main()
