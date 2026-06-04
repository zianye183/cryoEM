#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Generate lightweight QC stats and panels for IsoNet2 SPA smoke-test outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING

from common import DEFAULT_WORK_DIR, ensure_dir, robust_z, safe_stem, write_csv

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pseudo-manifest",
        default=str(DEFAULT_WORK_DIR / "pseudo_volumes" / "manifest.csv"),
        help="Manifest from make_pseudo_volumes.py.",
    )
    parser.add_argument(
        "--slices-manifest",
        default=str(DEFAULT_WORK_DIR / "extracted_slices" / "manifest.csv"),
        help="Manifest from extract_slices.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_WORK_DIR / "qc"),
        help="QC output directory.",
    )
    parser.add_argument(
        "--stack-policy",
        choices=["error", "mean", "first"],
        default="error",
        help="How to reload 3D input MRCs if encountered for QC.",
    )
    parser.add_argument(
        "--make-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG QC panels if matplotlib is available.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_mrc(path: Path) -> tuple[np.ndarray, object]:
    import mrcfile

    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32), mrc.voxel_size


def load_input_image(path: Path, stack_policy: str) -> np.ndarray:
    data, _ = read_mrc(path)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 3 and stack_policy == "mean":
        return data.mean(axis=0)
    if data.ndim == 3 and stack_policy == "first":
        return data[0]
    raise ValueError(
        f"Expected 2D input micrograph for QC; got shape {data.shape}: {path}"
    )


def center_crop(image: np.ndarray, crop_size_raw: str | int | None) -> np.ndarray:
    if crop_size_raw in [None, "", "None"]:
        return image
    crop_size = int(crop_size_raw)
    if crop_size <= 0:
        return image
    height, width = image.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def stats(prefix: str, array: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_p1": float(np.percentile(array, 1)),
        f"{prefix}_p5": float(np.percentile(array, 5)),
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
        f"{prefix}_max": float(np.max(array)),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    if a.size != b.size:
        return float("nan")
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std == 0 or b_std == 0:
        return float("nan")
    return float(np.mean(((a - np.mean(a)) / a_std) * ((b - np.mean(b)) / b_std)))


def radial_power_summary(array: np.ndarray) -> dict[str, float]:
    arr = robust_z(array)
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    y, x = np.indices(power.shape)
    cy = (power.shape[0] - 1) / 2.0
    cx = (power.shape[1] - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = radius.max()
    bands = {
        "low": radius <= 0.15 * rmax,
        "mid": (radius > 0.15 * rmax) & (radius <= 0.45 * rmax),
        "high": radius > 0.45 * rmax,
    }
    total = float(power.mean()) + 1e-12
    return {
        f"power_{name}_mean_ratio": float(power[mask].mean() / total)
        for name, mask in bands.items()
    }


def save_panel(
    path: Path,
    title: str,
    input_img: np.ndarray,
    center: np.ndarray,
    mean_z: np.ndarray,
    zstd: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    diff = robust_z(mean_z) - robust_z(input_img)
    images = [input_img, center, mean_z, diff, zstd]
    labels = ["input", "IsoNet2 center", "IsoNet2 mean-Z", "robust diff", "Z std"]
    cmaps = ["gray", "gray", "gray", "coolwarm", "magma"]

    fig, axes = plt.subplots(
        1, len(images), figsize=(4 * len(images), 4), constrained_layout=True
    )
    fig.suptitle(title)
    for ax, image, label, cmap in zip(axes, images, labels, cmaps, strict=True):
        display = robust_z(image) if label != "Z std" else image
        vmin, vmax = (np.percentile(display, 1), np.percentile(display, 99))
        ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def match_pseudo_row(
    slice_row: dict[str, str], pseudo_rows: list[dict[str, str]]
) -> dict[str, str] | None:
    prediction_name = Path(slice_row["prediction_path"]).name
    for row in pseudo_rows:
        pseudo_stem = safe_stem(Path(row["pseudo_volume_path"]))
        if pseudo_stem in prediction_name:
            return row
    return None


def main() -> None:
    args = parse_args()

    global np
    import numpy as np

    pseudo_rows = read_csv(Path(args.pseudo_manifest).expanduser().resolve())
    slice_rows = read_csv(Path(args.slices_manifest).expanduser().resolve())
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    panel_dir = ensure_dir(output_dir / "panels")

    qc_rows: list[dict[str, object]] = []
    for slice_row in slice_rows:
        pseudo_row = match_pseudo_row(slice_row, pseudo_rows)
        if pseudo_row is None:
            print(f"Skipping unmatched prediction: {slice_row['prediction_path']}")
            continue

        input_path = Path(pseudo_row["input_path"])
        input_img = load_input_image(input_path, args.stack_policy)
        input_img = center_crop(input_img, pseudo_row.get("crop_size"))
        center, _ = read_mrc(Path(slice_row["center_path"]))
        mean_z, _ = read_mrc(Path(slice_row["meanZ_path"]))
        zstd, _ = read_mrc(Path(slice_row["zstd_path"]))

        if input_img.shape != center.shape:
            raise ValueError(
                f"Input/output shape mismatch for {input_path}: input {input_img.shape}, output {center.shape}"
            )

        input_rz = robust_z(input_img)
        center_rz = robust_z(center)
        meanz_rz = robust_z(mean_z)
        row: dict[str, object] = {
            "input_path": str(input_path),
            "prediction_path": slice_row["prediction_path"],
            "input_shape": "x".join(map(str, input_img.shape)),
            "center_path": slice_row["center_path"],
            "meanZ_path": slice_row["meanZ_path"],
            "zstd_path": slice_row["zstd_path"],
            "pearson_input_center_robust": pearson(input_rz, center_rz),
            "pearson_input_meanZ_robust": pearson(input_rz, meanz_rz),
            "zstd_mean": float(np.mean(zstd)),
            "zstd_p95": float(np.percentile(zstd, 95)),
            "zstd_max": float(np.max(zstd)),
        }
        row.update(stats("input", input_img))
        row.update(stats("center", center))
        row.update(stats("meanZ", mean_z))
        for key, value in radial_power_summary(input_img).items():
            row[f"input_{key}"] = value
        for key, value in radial_power_summary(mean_z).items():
            row[f"meanZ_{key}"] = value
        qc_rows.append(row)

        if args.make_panels:
            try:
                panel_path = panel_dir / f"{safe_stem(input_path)}_qc.png"
                save_panel(panel_path, input_path.name, input_img, center, mean_z, zstd)
            except ImportError:
                print("matplotlib is unavailable; skipping PNG panels")
                args.make_panels = False

    write_csv(output_dir / "image_stats.csv", qc_rows)
    print(
        f"Wrote QC stats for {len(qc_rows)} image(s) to {output_dir / 'image_stats.csv'}"
    )


if __name__ == "__main__":
    main()
