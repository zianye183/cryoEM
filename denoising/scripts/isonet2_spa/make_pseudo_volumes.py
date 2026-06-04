#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Create repeated-depth pseudo-volumes from 2D SPA micrographs for IsoNet2 prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from common import (
    DEFAULT_INPUT_DIR,
    DEFAULT_WORK_DIR,
    collect_mrc_files,
    ensure_dir,
    load_json,
    safe_stem,
    write_csv,
)

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_INPUT_DIR)],
        help="Input .mrc files/directories/globs. Defaults to data/fuller_subset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_WORK_DIR / "pseudo_volumes"),
        help="Directory for pseudo-volume MRC files.",
    )
    parser.add_argument(
        "--metadata-json",
        default=str(DEFAULT_WORK_DIR / "checkpoint_metadata.json"),
        help="Checkpoint metadata JSON; used when --depth auto.",
    )
    parser.add_argument(
        "--depth",
        default="auto",
        help="Pseudo-volume Z depth, or 'auto' to use checkpoint cube_size.",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Limit number of input micrographs."
    )
    parser.add_argument(
        "--stack-policy",
        choices=["error", "mean", "first"],
        default="error",
        help=(
            "What to do if an input MRC is a non-singleton 3D stack. "
            "For confirmed micrographs this should not be needed."
        ),
    )
    parser.add_argument(
        "--invert-input",
        action="store_true",
        help="Invert input before repeating along Z; off by default.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help=(
            "Optional center-crop size in pixels before Z repetition. "
            "Strongly recommended for first IsoNet2 smoke tests to avoid huge pseudo-volumes."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_WORK_DIR / "pseudo_volumes" / "manifest.csv"),
        help="CSV manifest mapping input micrographs to pseudo-volumes.",
    )
    return parser.parse_args()


def resolve_depth(raw_depth: str, metadata_json: Path) -> int:
    if raw_depth != "auto":
        depth = int(raw_depth)
    else:
        metadata = load_json(metadata_json)
        depth = int(metadata.get("cube_size") or 96)
    if depth < 16:
        raise ValueError(f"Pseudo-volume depth is suspiciously small: {depth}")
    return depth


def load_micrograph(
    path: Path, stack_policy: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
    original_shape = tuple(int(dim) for dim in data.shape)

    if data.ndim == 2:
        image = data
    elif data.ndim == 3 and data.shape[0] == 1:
        image = data[0]
    elif data.ndim == 3 and stack_policy == "mean":
        image = data.mean(axis=0)
    elif data.ndim == 3 and stack_policy == "first":
        image = data[0]
    else:
        raise ValueError(
            f"Expected a 2D micrograph or singleton stack for {path}; got shape {original_shape}. "
            "If this is intentionally a frame stack, rerun with --stack-policy mean or first."
        )
    return np.asarray(image, dtype=np.float32), original_shape


def center_crop(image: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None:
        return image
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    height, width = image.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def write_pseudo_volume(
    input_path: Path, output_path: Path, image: np.ndarray, depth: int, invert: bool
) -> None:
    import mrcfile
    import numpy as np

    image = np.nan_to_num(
        image,
        nan=float(np.nanmedian(image)),
        posinf=float(np.nanmax(image)),
        neginf=float(np.nanmin(image)),
    )
    if invert:
        image = -image
    volume = np.repeat(image[np.newaxis, :, :], depth, axis=0).astype(
        np.float32, copy=False
    )

    with mrcfile.open(input_path, permissive=True, header_only=True) as src:
        voxel_size = src.voxel_size
    with mrcfile.new(output_path, overwrite=True) as dst:
        dst.set_data(volume)
        dst.voxel_size = voxel_size


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    depth = resolve_depth(args.depth, Path(args.metadata_json).expanduser().resolve())
    files = collect_mrc_files(args.inputs, limit=args.limit)

    rows: list[dict[str, object]] = []
    for input_path in files:
        image, original_shape = load_micrograph(input_path, args.stack_policy)
        image = center_crop(image, args.crop_size)
        crop_label = f".crop{args.crop_size}" if args.crop_size else ""
        output_path = (
            output_dir / f"{safe_stem(input_path)}{crop_label}.pseudoZ{depth}.mrc"
        )
        write_pseudo_volume(input_path, output_path, image, depth, args.invert_input)
        rows.append(
            {
                "input_path": str(input_path),
                "pseudo_volume_path": str(output_path),
                "input_shape": "x".join(map(str, original_shape)),
                "image_shape": "x".join(map(str, image.shape)),
                "pseudo_shape": f"{depth}x{image.shape[0]}x{image.shape[1]}",
                "crop_size": args.crop_size if args.crop_size else "None",
                "inverted": args.invert_input,
            }
        )
        print(f"{input_path} -> {output_path}")

    write_csv(Path(args.manifest).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
