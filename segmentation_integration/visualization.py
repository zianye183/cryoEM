"""QC overlay rendering for vesicle-mask particle filtering."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from segmentation_integration.mrc_io import read_mode2_mrc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage


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


def render_mask_comparison_png(
    *,
    output_path: Path,
    found_mask: np.ndarray,
    filtered_mask: np.ndarray,
    title: str,
    found_count: int,
    filtered_count: int,
    dpi: int = 180,
) -> None:
    """Render found SAM mask union next to post-filter mask union."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = [
        (axes[0], found_mask, f"found n={found_count}"),
        (axes[1], filtered_mask, f"filtered n={filtered_count}"),
    ]
    for ax, mask, label in panels:
        ax.imshow(np.asarray(mask, dtype=np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_mask_overlay_comparison_png(
    *,
    micrograph_path: Path,
    output_path: Path,
    found_mask: np.ndarray,
    filtered_mask: np.ndarray,
    downsample: int,
    title: str,
    found_count: int,
    filtered_count: int,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    dpi: int = 180,
) -> None:
    """Render found and filtered mask overlays on the same micrograph."""
    image = read_mode2_mrc(micrograph_path)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(f"Micrograph has no finite pixels: {micrograph_path}")
    vmin, vmax = np.percentile(finite, contrast_percentiles)
    display = image[::display_downsample, ::display_downsample]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    panels = [
        (axes[0], found_mask, f"found n={found_count}", "Wistia"),
        (axes[1], filtered_mask, f"filtered n={filtered_count}", "winter"),
    ]
    for ax, mask, label, cmap in panels:
        ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        full = np.repeat(np.repeat(np.asarray(mask, dtype=bool), downsample, axis=0), downsample, axis=1)
        full = full[: image.shape[0], : image.shape[1]]
        shown = full[::display_downsample, ::display_downsample]
        ax.imshow(
            np.ma.masked_where(~shown, np.ones_like(shown)),
            cmap=cmap,
            alpha=0.35,
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_mask_filter_debug_png(
    *,
    output_path: Path,
    found_kept_masks: list[np.ndarray],
    found_rejected_masks: list[np.ndarray],
    filtered_masks: list[np.ndarray],
    title: str,
    micrograph_path: Path | None = None,
    downsample: int = 4,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    dpi: int = 180,
) -> None:
    """Render useful upstream QC: contour-only found kept/rejected and filtered masks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if micrograph_path is not None:
        image = read_mode2_mrc(micrograph_path)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError(f"Micrograph has no finite pixels: {micrograph_path}")
        vmin, vmax = np.percentile(finite, contrast_percentiles)
        display = image[::display_downsample, ::display_downsample]
    else:
        image = None
        display = None
        vmin = vmax = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, label in zip(axes, ("found: kept green, rejected red", "filtered output")):
        if display is not None:
            ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        else:
            shape = _first_mask_shape(found_kept_masks, found_rejected_masks, filtered_masks)
            ax.imshow(np.zeros(shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title(label)
        ax.set_axis_off()

    _imshow_mask_contours(
        axes[0],
        found_rejected_masks,
        color=(1.0, 0.0, 0.0, 0.95),
        downsample=downsample,
        display_downsample=display_downsample,
        image_shape=image.shape if image is not None else None,
    )
    _imshow_mask_contours(
        axes[0],
        found_kept_masks,
        color=(0.0, 1.0, 0.0, 0.95),
        downsample=downsample,
        display_downsample=display_downsample,
        image_shape=image.shape if image is not None else None,
    )
    _imshow_mask_contours(
        axes[1],
        filtered_masks,
        color=(0.0, 0.85, 1.0, 0.95),
        downsample=downsample,
        display_downsample=display_downsample,
        image_shape=image.shape if image is not None else None,
    )
    fig.suptitle(title)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_mask_metrics_png(
    *,
    output_path: Path,
    masks: list[dict],
    title: str,
    rejected_masks: list[dict] | None = None,
    micrograph_path: Path | None = None,
    downsample: int = 4,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    dpi: int = 180,
) -> None:
    """Render mask contours with per-vesicle area and roundness labels."""
    rejected_masks = rejected_masks or []
    all_masks = [*masks, *rejected_masks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if micrograph_path is not None:
        image = read_mode2_mrc(micrograph_path)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError(f"Micrograph has no finite pixels: {micrograph_path}")
        vmin, vmax = np.percentile(finite, contrast_percentiles)
        display = image[::display_downsample, ::display_downsample]
    else:
        image = None
        display = None
        vmin = vmax = None

    fig, ax = plt.subplots(figsize=(8, 8))
    if display is not None:
        ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    else:
        shape = _first_mask_shape([np.asarray(mask["segmentation"], dtype=bool) for mask in all_masks])
        ax.imshow(np.zeros(shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper")

    _imshow_mask_contours(
        ax,
        [np.asarray(mask["segmentation"], dtype=bool) for mask in rejected_masks],
        color=(1.0, 0.0, 0.0, 0.95),
        downsample=downsample,
        display_downsample=display_downsample,
        image_shape=image.shape if image is not None else None,
    )
    _imshow_mask_contours(
        ax,
        [np.asarray(mask["segmentation"], dtype=bool) for mask in masks],
        color=(0.0, 1.0, 0.0, 0.95),
        downsample=downsample,
        display_downsample=display_downsample,
        image_shape=image.shape if image is not None else None,
    )
    for mask in all_masks:
        _annotate_mask_metrics(
            ax,
            mask,
            downsample=downsample,
            display_downsample=display_downsample,
            image_shape=image.shape if image is not None else None,
        )
    ax.set_title("vesicle metrics: accepted green, rejected red")
    ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _first_mask_shape(*mask_groups: list[np.ndarray]) -> tuple[int, int]:
    for group in mask_groups:
        for mask in group:
            return np.asarray(mask).shape
    return (1, 1)


def _imshow_mask_contours(
    ax,
    masks: list[np.ndarray],
    *,
    color: tuple[float, float, float, float],
    downsample: int,
    display_downsample: int,
    image_shape: tuple[int, int] | None,
) -> None:
    if not masks:
        return
    union = np.zeros(np.asarray(masks[0], dtype=bool).shape, dtype=bool)
    for mask in masks:
        segmentation = np.asarray(mask, dtype=bool)
        if not segmentation.any():
            continue
        edge = segmentation & ~ndimage.binary_erosion(segmentation)
        edge = ndimage.binary_dilation(edge, iterations=1)
        union |= edge
    if image_shape is not None:
        union = np.repeat(np.repeat(union, downsample, axis=0), downsample, axis=1)
        union = union[: image_shape[0], : image_shape[1]]
    effective_display_downsample = display_downsample if image_shape is not None else 1
    shown = union[::effective_display_downsample, ::effective_display_downsample]
    rgba = np.zeros((*shown.shape, 4), dtype=float)
    rgba[shown] = color
    ax.imshow(rgba, interpolation="nearest", origin="upper")


def _annotate_mask_metrics(
    ax,
    mask: dict,
    *,
    downsample: int,
    display_downsample: int,
    image_shape: tuple[int, int] | None,
) -> None:
    segmentation = np.asarray(mask["segmentation"], dtype=bool)
    if not segmentation.any():
        return
    rows, cols = np.nonzero(segmentation)
    if image_shape is not None:
        scale = downsample / display_downsample
    else:
        scale = 1
    x = float(np.mean(cols)) * scale
    y = float(np.mean(rows)) * scale
    area = _format_metric(mask.get("area"))
    roundness = _format_metric(mask.get("roundness"), digits=2)
    ax.annotate(
        f"A={area}\nR={roundness}",
        xy=(x, y),
        xytext=(4, 4),
        textcoords="offset points",
        color="white",
        fontsize=5.5,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "edgecolor": "none", "alpha": 0.65},
    )


def _format_metric(value, *, digits: int = 0) -> str:
    if value in (None, ""):
        return "NA"
    number = float(value)
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


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
