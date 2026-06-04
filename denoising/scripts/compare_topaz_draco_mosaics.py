#!/usr/bin/env python3
"""Create row-wise mosaics comparing raw, Topaz-denoised, and DRACO-denoised MRCs.

For each source micrograph, one row is written with three panels:
raw | Topaz denoised | DRACO denoised.

Two PNG mosaics are emitted:
1. raw_intensity mosaic: preserves the numeric output arrays and displays them with
   one global min/max across all panels.
2. p01_p99_normalized mosaic: independently clips/scales each panel to its own
   1st-99th percentile contrast range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

DEFAULT_RAW_DIR = Path("/nfs/bartesaghilab2/zy223/denoising/data/subset")
DEFAULT_TOPAZ_DIR = Path("/nfs/bartesaghilab2/zy223/denoising/results/topaz_pretrained")
DEFAULT_DRACO_DIR = Path("/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained")
DEFAULT_OUTPUT_DIR = Path("/nfs/bartesaghilab2/zy223/denoising/results/comparisons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--topaz-dir", default=str(DEFAULT_TOPAZ_DIR))
    parser.add_argument("--draco-dir", default=str(DEFAULT_DRACO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--topaz-suffix",
        default="_topaz_denoised",
        help="Suffix inserted before .mrc in Topaz-denoised filenames.",
    )
    parser.add_argument(
        "--draco-suffix",
        default="_draco_denoised",
        help="Suffix inserted before .mrc in DRACO-denoised filenames.",
    )
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=99.0)
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution. Increase for publication-quality output.",
    )
    parser.add_argument(
        "--max-display-pixels",
        type=int,
        default=1600,
        help="Downsample each panel for PNG display if the largest dimension is above this. Does not modify MRC data.",
    )
    return parser.parse_args()


def read_mrc_2d(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data)
    if data.ndim == 2:
        return data.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D MRC or singleton stack, got {data.shape}: {path}")


def find_denoised(raw_path: Path, denoised_dir: Path, suffix: str, method: str) -> Path:
    candidates = [
        denoised_dir / f"{raw_path.stem}{suffix}.mrc",
        denoised_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No {method} output found for {raw_path.name}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def decimate_for_display(image: np.ndarray, max_display_pixels: int) -> np.ndarray:
    if max_display_pixels <= 0:
        return image
    max_dim = max(image.shape)
    if max_dim <= max_display_pixels:
        return image
    step = int(np.ceil(max_dim / max_display_pixels))
    return image[::step, ::step]


def finite_min_max(images: list[np.ndarray]) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for image in images:
        finite = image[np.isfinite(image)]
        if finite.size:
            mins.append(float(np.min(finite)))
            maxs.append(float(np.max(finite)))
    if not mins:
        return 0.0, 1.0
    vmin = min(mins)
    vmax = max(maxs)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    vmin, vmax = np.percentile(finite, [low, high])
    if np.isclose(vmin, vmax):
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image.astype(np.float32, copy=False) - float(vmin)) / float(
        vmax - vmin
    )
    return np.clip(normalized, 0.0, 1.0)


def write_mosaic(
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str,
    mode: str,
    low: float,
    high: float,
    dpi: int,
) -> None:
    nrows = len(rows)
    ncols = 3
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    all_images = [
        image for _, raw, topaz, draco in rows for image in (raw, topaz, draco)
    ]
    global_vmin, global_vmax = finite_min_max(all_images)

    col_titles = ["Raw", "Topaz denoised", "DRACO denoised"]
    for row_index, (name, raw, topaz, draco) in enumerate(rows):
        row_images = [raw, topaz, draco]
        for col_index, (axis, image, col_title) in enumerate(
            zip(axes[row_index], row_images, col_titles)
        ):
            if mode == "raw":
                shown = image
                vmin, vmax = global_vmin, global_vmax
            elif mode == "percentile":
                shown = percentile_normalize(image, low, high)
                vmin, vmax = 0.0, 1.0
            else:
                raise ValueError(f"Unknown mode: {mode}")
            axis.imshow(shown, cmap="gray", vmin=vmin, vmax=vmax)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(col_title, fontsize=14)
            if col_index == 0:
                axis.text(
                    0.01,
                    0.99,
                    name,
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    color="white",
                    fontsize=8,
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.55,
                        "pad": 3,
                        "edgecolor": "none",
                    },
                )

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    topaz_dir = Path(args.topaz_dir).expanduser().resolve()
    draco_dir = Path(args.draco_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    raw_paths = sorted(raw_dir.glob("*.mrc"))
    if args.max_files > 0:
        raw_paths = raw_paths[: args.max_files]
    if not raw_paths:
        raise SystemExit(f"No raw .mrc files found in {raw_dir}")

    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for raw_path in raw_paths:
        topaz_path = find_denoised(
            raw_path, topaz_dir, args.topaz_suffix, "Topaz"
        )
        draco_path = find_denoised(
            raw_path, draco_dir, args.draco_suffix, "DRACO"
        )
        print(f"Reading {raw_path.name}")
        raw = decimate_for_display(read_mrc_2d(raw_path), args.max_display_pixels)
        topaz = decimate_for_display(read_mrc_2d(topaz_path), args.max_display_pixels)
        draco = decimate_for_display(read_mrc_2d(draco_path), args.max_display_pixels)
        if raw.shape != topaz.shape or raw.shape != draco.shape:
            raise ValueError(
                f"Shape mismatch for {raw_path.name}: raw={raw.shape}, topaz={topaz.shape}, draco={draco.shape}"
            )
        rows.append((raw_path.stem, raw, topaz, draco))

    raw_output = output_dir / "raw_topaz_draco_mosaic_raw_intensity.png"
    norm_output = output_dir / "raw_topaz_draco_mosaic_p01_p99_normalized.png"
    write_mosaic(
        rows,
        raw_output,
        title="Raw vs Topaz vs DRACO denoising - raw intensity display",
        mode="raw",
        low=args.low,
        high=args.high,
        dpi=args.dpi,
    )
    write_mosaic(
        rows,
        norm_output,
        title=f"Raw vs Topaz vs DRACO denoising - per-panel {args.low:g}-{args.high:g}% contrast",
        mode="percentile",
        low=args.low,
        high=args.high,
        dpi=args.dpi,
    )
    print(f"Wrote {raw_output}")
    print(f"Wrote {norm_output}")


if __name__ == "__main__":
    main()
