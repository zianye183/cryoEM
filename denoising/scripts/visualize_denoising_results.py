#!/usr/bin/env python3
"""Save side-by-side PNG previews of original and denoised MRC micrographs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dir",
        default="/nfs/bartesaghilab2/zy223/denoising/data/subset",
        help="Directory containing original .mrc files.",
    )
    parser.add_argument(
        "--denoised-dir",
        default="/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained",
        help="Directory containing denoised .mrc files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained/previews",
        help="Directory where PNG previews will be saved.",
    )
    parser.add_argument(
        "--suffix",
        default="_draco_denoised",
        help="Suffix inserted before .mrc in denoised filenames.",
    )
    parser.add_argument(
        "--method-label",
        default="Denoised",
        help="Label used for the denoised panel title.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to visualize. 0 means all.",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower display percentile.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper display percentile.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="If >0, save a centered square crop of this size instead of the full image.",
    )
    parser.add_argument(
        "--crop-x",
        type=int,
        help="Optional crop left coordinate. Requires --crop-size.",
    )
    parser.add_argument(
        "--crop-y",
        type=int,
        help="Optional crop top coordinate. Requires --crop-size.",
    )
    parser.add_argument(
        "--difference-scale",
        type=float,
        default=1.0,
        help="Multiply the residual image before display to make removed noise easier to see.",
    )
    parser.add_argument(
        "--independent-contrast",
        action="store_true",
        help="Scale original and denoised panels independently instead of sharing display limits.",
    )
    return parser.parse_args()


def read_mrc_2d(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
    if data.ndim == 2:
        return data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32)
    raise ValueError(f"Expected a 2D MRC image, got shape {data.shape}: {path}")


def display_limits(*images: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite_values = np.concatenate(
        [image[np.isfinite(image)].ravel() for image in images]
    )
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite_values, [low, high])
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def crop_image(
    image: np.ndarray, *, crop_size: int, crop_x: int | None, crop_y: int | None
) -> np.ndarray:
    if crop_size <= 0:
        return image
    height, width = image.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"--crop-size {crop_size} is larger than image shape {image.shape}"
        )
    x0 = (width - crop_size) // 2 if crop_x is None else crop_x
    y0 = (height - crop_size) // 2 if crop_y is None else crop_y
    x0 = max(0, min(x0, width - crop_size))
    y0 = max(0, min(y0, height - crop_size))
    return image[y0 : y0 + crop_size, x0 : x0 + crop_size]


def save_preview(
    original_path: Path,
    denoised_path: Path,
    output_path: Path,
    *,
    low: float,
    high: float,
    crop_size: int,
    crop_x: int | None,
    crop_y: int | None,
    difference_scale: float,
    independent_contrast: bool,
    method_label: str,
) -> None:
    original = crop_image(
        read_mrc_2d(original_path), crop_size=crop_size, crop_x=crop_x, crop_y=crop_y
    )
    denoised = crop_image(
        read_mrc_2d(denoised_path), crop_size=crop_size, crop_x=crop_x, crop_y=crop_y
    )
    difference = (original - denoised) * difference_scale

    if independent_contrast:
        orig_vmin, orig_vmax = display_limits(original, low=low, high=high)
        den_vmin, den_vmax = display_limits(denoised, low=low, high=high)
    else:
        orig_vmin, orig_vmax = display_limits(original, denoised, low=low, high=high)
        den_vmin, den_vmax = orig_vmin, orig_vmax
    diff_vmin, diff_vmax = display_limits(difference, low=low, high=high)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    diff_title = "Original - denoised"
    if difference_scale != 1.0:
        diff_title += f" × {difference_scale:g}"
    panels = [
        (original, "Original", orig_vmin, orig_vmax),
        (denoised, method_label, den_vmin, den_vmax),
        (difference, diff_title, diff_vmin, diff_vmax),
    ]
    for axis, (image, title, vmin, vmax) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(original_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    original_dir = Path(args.original_dir).expanduser().resolve()
    denoised_dir = Path(args.denoised_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    originals = sorted(original_dir.glob("*.mrc"))
    if args.max_files > 0:
        originals = originals[: args.max_files]
    if not originals:
        raise SystemExit(f"No original .mrc files found in {original_dir}")

    written = 0
    for original_path in originals:
        denoised_path = denoised_dir / f"{original_path.stem}{args.suffix}.mrc"
        if not denoised_path.is_file():
            print(f"Skipping; missing denoised file: {denoised_path}")
            continue
        output_path = output_dir / f"{original_path.stem}_comparison.png"
        print(f"Writing {output_path}")
        save_preview(
            original_path,
            denoised_path,
            output_path,
            low=args.percentile_low,
            high=args.percentile_high,
            crop_size=args.crop_size,
            crop_x=args.crop_x,
            crop_y=args.crop_y,
            difference_scale=args.difference_scale,
            independent_contrast=args.independent_contrast,
            method_label=args.method_label,
        )
        written += 1

    print(f"Wrote {written} preview PNG(s) to {output_dir}")


if __name__ == "__main__":
    main()
