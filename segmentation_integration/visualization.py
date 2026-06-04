"""QC overlay rendering for vesicle-mask particle filtering."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def render_overlay(
    *,
    micrograph_path: Path,
    output_path: Path,
    membrane_band: np.ndarray | None,
    downsample: int,
    original_xy: np.ndarray,
    kept_xy: np.ndarray,
    rejected_xy: np.ndarray,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    max_picks_per_class: int = 500,
    dpi: int = 180,
) -> None:
    """Render raw micrograph, membrane band, and original/kept/rejected picks."""
    image = read_mode2_mrc(micrograph_path)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(f"Micrograph has no finite pixels: {micrograph_path}")
    vmin, vmax = np.percentile(finite, contrast_percentiles)
    display = image[::display_downsample, ::display_downsample]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    if membrane_band is not None and membrane_band.size:
        band_full = np.repeat(np.repeat(membrane_band, downsample, axis=0), downsample, axis=1)
        band_display = band_full[: image.shape[0], : image.shape[1]]
        band_display = band_display[::display_downsample, ::display_downsample]
        ax.imshow(
            np.ma.masked_where(~band_display, np.ones_like(band_display)),
            cmap="Wistia",
            alpha=0.30,
            interpolation="nearest",
        )
    _scatter(ax, original_xy, display_downsample, "tab:blue", "o", "original", max_picks_per_class, alpha=0.35)
    _scatter(ax, kept_xy, display_downsample, "lime", "o", "kept", max_picks_per_class, alpha=0.95)
    _scatter(ax, rejected_xy, display_downsample, "red", "x", "rejected", max_picks_per_class, alpha=0.95)
    ax.set_title(micrograph_path.name)
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize="small", framealpha=0.75)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_mask_png(
    *,
    output_path: Path,
    membrane_band: np.ndarray,
    title: str,
    dpi: int = 180,
) -> None:
    """Render a standalone binary membrane-band mask PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(membrane_band, dtype=np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _scatter(
    ax,
    xy: np.ndarray,
    display_downsample: int,
    color: str,
    marker: str,
    label: str,
    max_picks: int,
    alpha: float,
) -> None:
    if xy.size == 0:
        return
    xy = xy[:max_picks]
    ax.scatter(
        xy[:, 0] / display_downsample,
        xy[:, 1] / display_downsample,
        s=24,
        facecolors="none" if marker == "o" else color,
        edgecolors=color if marker == "o" else None,
        marker=marker,
        linewidths=0.8,
        alpha=alpha,
        label=f"{label} n={len(xy)}",
    )
