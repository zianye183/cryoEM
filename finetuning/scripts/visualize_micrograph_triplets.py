#!/usr/bin/env python3
"""Render side-by-side original even/odd/full micrograph PNG panels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from finetuning_common import read_tsv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="TSV with even_path, odd_path, complete_path columns.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for PNG panels.")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-display-pixels", type=int, default=900)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--contrast",
        choices=["per-image", "shared"],
        default="per-image",
        help="Display contrast scaling. per-image avoids half/full dynamic-range mismatch; shared uses one scale across all three panels.",
    )
    return parser.parse_args()


def load_mrc(path: Path) -> Any:
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as mrc:
        arr = np.asarray(mrc.data, dtype=np.float32).copy()
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    return arr


def display_image(arr: Any, max_pixels: int) -> Any:
    import numpy as np

    image = np.asarray(arr, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = image.shape[-2:]
    scale = min(1.0, max_pixels / float(max(h, w)))
    if scale < 1.0:
        step = max(1, int(round(1.0 / scale)))
        image = image[::step, ::step]
    return image


def robust_limits(image: Any) -> tuple[float, float]:
    import numpy as np

    vmin, vmax = np.percentile(image, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.min(image)), float(np.max(image))
    return float(vmin), float(vmax)


def render_triplet(
    row: dict[str, str],
    output_path: Path,
    max_pixels: int,
    dpi: int,
    contrast: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    items = [
        ("even / half1", Path(row["even_path"]).expanduser().resolve()),
        ("odd / half2", Path(row["odd_path"]).expanduser().resolve()),
        ("full", Path(row["complete_path"]).expanduser().resolve()),
    ]
    images = [display_image(load_mrc(path), max_pixels) for _, path in items]
    shared_limits = None
    if contrast == "shared":
        stack = np.concatenate([image.ravel() for image in images])
        shared_limits = robust_limits(stack)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    fig.suptitle(
        f"{row.get('micrograph_id', 'micrograph')} ({contrast} 1–99% contrast)",
        fontsize=9,
    )
    for ax, (title, path), image in zip(axes[0], items, images):
        vmin, vmax = shared_limits if shared_limits else robust_limits(image)
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(path.name, fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_tsv(Path(args.manifest).expanduser().resolve())[: args.count]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for row in rows:
        if not row.get("complete_path"):
            raise ValueError(f"Missing complete_path for {row.get('micrograph_id')}")
        safe_id = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", row.get("micrograph_id", "micrograph")
        )
        out = output_dir / f"{safe_id}_even_odd_full.png"
        render_triplet(row, out, args.max_display_pixels, args.dpi, args.contrast)
        rendered.append(str(out))
        print(f"Rendered: {out}")
    write_json(
        output_dir / "triplet_visualization_manifest.json",
        {
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "rendered_panels": rendered,
            "contrast": args.contrast,
        },
    )
    print(f"Triplet visualization directory: {output_dir}")


if __name__ == "__main__":
    main()
