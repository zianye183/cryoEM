#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Extract 2D center/mean-Z/z-std slices from IsoNet2 pseudo-volume predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from common import DEFAULT_WORK_DIR, ensure_dir, safe_stem, write_csv

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict-dir",
        default=str(DEFAULT_WORK_DIR / "predict"),
        help="Directory containing IsoNet2 predicted .mrc volumes.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_WORK_DIR / "extracted_slices"),
        help="Directory for extracted 2D MRC files.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_WORK_DIR / "extracted_slices" / "manifest.csv"),
        help="CSV manifest of extracted slices.",
    )
    parser.add_argument(
        "--glob",
        default="*.mrc",
        help="Glob for predicted volumes inside --predict-dir.",
    )
    return parser.parse_args()


def collect_predictions(predict_dir: Path, pattern: str) -> list[Path]:
    files = sorted(predict_dir.glob(pattern))
    files = [
        path
        for path in files
        if path.is_file() and not path.name.endswith("_slice.mrc")
    ]
    if not files:
        raise FileNotFoundError(
            f"No prediction MRC files found in {predict_dir} with pattern {pattern}"
        )
    return files


def write_mrc(path: Path, data: np.ndarray, voxel_size) -> None:
    import mrcfile
    import numpy as np

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel_size


def main() -> None:
    args = parse_args()
    predict_dir = Path(args.predict_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    predictions = collect_predictions(predict_dir, args.glob)

    import mrcfile
    import numpy as np

    rows: list[dict[str, object]] = []
    for pred_path in predictions:
        with mrcfile.open(pred_path, permissive=True) as mrc:
            volume = np.asarray(mrc.data, dtype=np.float32)
            voxel_size = mrc.voxel_size

        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D predicted volume from IsoNet2, got {volume.shape}: {pred_path}"
            )

        center_idx = volume.shape[0] // 2
        center = volume[center_idx]
        mean_z = volume.mean(axis=0)
        z_std = volume.std(axis=0)

        stem = safe_stem(pred_path)
        center_path = output_dir / f"{stem}_center.mrc"
        meanz_path = output_dir / f"{stem}_meanZ.mrc"
        zstd_path = output_dir / f"{stem}_zstd.mrc"

        write_mrc(center_path, center, voxel_size)
        write_mrc(meanz_path, mean_z, voxel_size)
        write_mrc(zstd_path, z_std, voxel_size)

        rows.append(
            {
                "prediction_path": str(pred_path),
                "prediction_shape": "x".join(map(str, volume.shape)),
                "center_z_index": center_idx,
                "center_path": str(center_path),
                "meanZ_path": str(meanz_path),
                "zstd_path": str(zstd_path),
                "zstd_mean": float(z_std.mean()),
                "zstd_p95": float(np.percentile(z_std, 95)),
                "zstd_max": float(z_std.max()),
            }
        )
        print(f"Extracted slices from {pred_path}")

    write_csv(Path(args.manifest).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
