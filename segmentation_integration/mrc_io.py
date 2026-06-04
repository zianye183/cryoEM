"""Minimal MRC I/O helpers without plotting dependencies."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_mode2_mrc(path: Path) -> np.ndarray:
    """Read a simple MRC mode-2 float32 image."""
    with path.open("rb") as handle:
        header = handle.read(1024)
        if len(header) < 1024:
            raise ValueError(f"MRC header is too short: {path}")
        nx, ny, nz, mode = struct.unpack("<4i", header[:16])
        if mode != 2:
            raise ValueError(f"Only MRC mode 2 float32 images are supported without mrcfile: {path}")
        count = nx * ny * max(nz, 1)
        data = np.fromfile(handle, dtype="<f4", count=count)
    if data.size != count:
        raise ValueError(f"MRC data is shorter than expected: {path}")
    data = data.reshape((max(nz, 1), ny, nx))
    return data[0].astype(np.float32) if data.shape[0] == 1 else data.astype(np.float32)[0]


def write_mode2_mrc_for_test(path: Path, image: np.ndarray) -> None:
    """Write a minimal mode-2 MRC file for tests."""
    image = np.asarray(image, dtype="<f4")
    if image.ndim != 2:
        raise ValueError("Test MRC writer only supports 2D images")
    ny, nx = image.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, 1, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(image.tobytes(order="C"))
