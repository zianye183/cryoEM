#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Create side-by-side PNGs comparing input crop vs IsoNet2 output slice."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING

from common import DEFAULT_WORK_DIR, ensure_dir, robust_z, safe_stem

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
        default=str(DEFAULT_WORK_DIR / "qc" / "side_by_side"),
        help="Directory for side-by-side PNGs.",
    )
    parser.add_argument(
        "--output-kind",
        choices=["meanZ", "center"],
        default="meanZ",
        help="Which extracted IsoNet2 2D slice to compare.",
    )
    parser.add_argument(
        "--stack-policy",
        choices=["error", "mean", "first"],
        default="error",
        help="How to load original input if it is unexpectedly 3D.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_mrc(path: Path):
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32)


def load_input_image(path: Path, stack_policy: str):
    data = read_mrc(path)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    if data.ndim == 3 and stack_policy == "mean":
        return data.mean(axis=0)
    if data.ndim == 3 and stack_policy == "first":
        return data[0]
    raise ValueError(f"Expected 2D input micrograph; got {data.shape}: {path}")


def center_crop(image, crop_size_raw: str | int | None):
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


def match_pseudo_row(
    slice_row: dict[str, str], pseudo_rows: list[dict[str, str]]
) -> dict[str, str] | None:
    prediction_name = Path(slice_row["prediction_path"]).name
    for row in pseudo_rows:
        pseudo_stem = safe_stem(Path(row["pseudo_volume_path"]))
        if pseudo_stem in prediction_name:
            return row
    return None


def save_side_by_side(
    path: Path, title: str, input_img, output_img, output_label: str
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [("Input crop", input_img), (f"IsoNet2 {output_label}", output_img)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(title)
    for ax, (label, image) in zip(axes, panels, strict=True):
        display = robust_z(image)
        vmin, vmax = np.percentile(display, [1, 99])
        ax.imshow(display, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pseudo_rows = read_csv(Path(args.pseudo_manifest).expanduser().resolve())
    slice_rows = read_csv(Path(args.slices_manifest).expanduser().resolve())
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    created = []
    for slice_row in slice_rows:
        pseudo_row = match_pseudo_row(slice_row, pseudo_rows)
        if pseudo_row is None:
            print(f"Skipping unmatched prediction: {slice_row['prediction_path']}")
            continue

        input_path = Path(pseudo_row["input_path"])
        input_img = load_input_image(input_path, args.stack_policy)
        input_img = center_crop(input_img, pseudo_row.get("crop_size"))

        output_path = Path(slice_row[f"{args.output_kind}_path"])
        output_img = read_mrc(output_path)
        if input_img.shape != output_img.shape:
            raise ValueError(
                f"Input/output shape mismatch for {input_path.name}: "
                f"{input_img.shape} vs {output_img.shape}"
            )

        png_path = (
            output_dir / f"{safe_stem(input_path)}_{args.output_kind}_side_by_side.png"
        )
        save_side_by_side(
            png_path, input_path.name, input_img, output_img, args.output_kind
        )
        created.append(png_path)
        print(png_path)

    if not created:
        raise SystemExit("No side-by-side PNGs created")


if __name__ == "__main__":
    main()
