"""Preview rendering for visual segmentation comparisons."""

from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from segmentation_integration.mrc_io import read_mode2_mrc
from segmentation_integration.vesicle_masks import load_vesicle_masks

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_cryovia_label_mask(path: Path) -> np.ndarray:
    """Load a CryoVIA sparse label output and return a 2D boolean union mask."""
    path = Path(path)
    try:
        import sparse

        labels = sparse.load_npz(path).todense()
    except Exception:
        with np.load(path) as payload:
            labels = payload["data"] if "data" in payload else payload[payload.files[0]]
    labels = np.asarray(labels)
    if labels.ndim == 3:
        labels = np.any(labels != 0, axis=0)
    elif labels.ndim != 2:
        raise ValueError(f"CryoVIA label file must contain a 2D or 3D array: {path}")
    return np.asarray(labels != 0, dtype=bool)


def load_vesicle_picker_union_mask(path: Path) -> np.ndarray:
    """Load Vesicle Picker masks and return their boolean union."""
    masks = load_vesicle_masks(Path(path))
    if not masks:
        return np.zeros((1, 1), dtype=bool)
    union = np.zeros(np.asarray(masks[0]["segmentation"], dtype=bool).shape, dtype=bool)
    for mask in masks:
        union |= np.asarray(mask["segmentation"], dtype=bool)
    return union


def render_segmentation_overlay(
    *,
    micrograph_path: Path,
    output_path: Path,
    masks: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    mask_downsample: int,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    dpi: int = 180,
) -> None:
    """Render one or more segmentation mask contours on a raw micrograph."""
    image, display, vmin, vmax = _display_image(micrograph_path, contrast_percentiles, display_downsample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    drew_any_mask = False
    for mask, label, color in zip(masks, labels, colors):
        drew_any_mask |= _draw_mask_contour(
            ax,
            mask,
            color=color,
            label=label,
            mask_downsample=mask_downsample,
            display_downsample=display_downsample,
            image_shape=image.shape,
        )
    ax.set_title(title)
    ax.set_axis_off()
    if drew_any_mask:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.75)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def render_segmentation_comparison(
    *,
    micrograph_path: Path,
    output_path: Path,
    cryovia_mask: np.ndarray | None,
    sam_found_mask: np.ndarray | None,
    sam_filtered_mask: np.ndarray | None,
    mask_downsample: int,
    title: str,
    cryovia_mask_downsample: int | None = None,
    sam_mask_downsample: int | None = None,
    contrast_percentiles: tuple[float, float] = (1, 99),
    display_downsample: int = 4,
    dpi: int = 180,
) -> None:
    """Render raw, CryoVIA, SAM found, and SAM filtered panels side by side."""
    image, display, vmin, vmax = _display_image(micrograph_path, contrast_percentiles, display_downsample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        ("raw", None, "white", mask_downsample),
        ("CryoVIA Default_thin", cryovia_mask, "lime", cryovia_mask_downsample or mask_downsample),
        ("SAM found", sam_found_mask, "orange", sam_mask_downsample or mask_downsample),
        ("SAM filtered", sam_filtered_mask, "cyan", sam_mask_downsample or mask_downsample),
    ]
    for ax, (label, mask, color, panel_downsample) in zip(axes, panels):
        ax.imshow(display, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        if mask is not None:
            _draw_mask_contour(
                ax,
                mask,
                color=color,
                label=label,
                mask_downsample=panel_downsample,
                display_downsample=display_downsample,
                image_shape=image.shape,
            )
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_review_index(*, output_path: Path, rows: list[dict[str, str]], title: str) -> None:
    """Write a static HTML page for manual segmentation review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_rows = []
    for row in rows:
        cells = [
            f"<td><code>{html.escape(row['micrograph'])}</code></td>",
            _image_cell(row.get("comparison", "")),
            _image_cell(row.get("cryovia", "")),
            _image_cell(row.get("sam_found", "")),
            _image_cell(row.get("sam_filtered", "")),
        ]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    output_path.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    img {{ width: 260px; max-width: 100%; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <table>
    <thead><tr><th>Micrograph</th><th>Comparison</th><th>CryoVIA</th><th>SAM found</th><th>SAM filtered</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
""".format(title=html.escape(title), rows="\n      ".join(body_rows))
    )


def _display_image(
    micrograph_path: Path,
    contrast_percentiles: tuple[float, float],
    display_downsample: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    image = read_mode2_mrc(Path(micrograph_path))
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(f"Micrograph has no finite pixels: {micrograph_path}")
    vmin, vmax = np.percentile(finite, contrast_percentiles)
    return image, image[::display_downsample, ::display_downsample], float(vmin), float(vmax)


def _draw_mask_contour(
    ax,
    mask: np.ndarray,
    *,
    color: str,
    label: str,
    mask_downsample: int,
    display_downsample: int,
    image_shape: tuple[int, int],
) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return False
    full = _scale_mask_to_image_shape(mask, mask_downsample=mask_downsample, image_shape=image_shape)
    edge = full & ~ndimage.binary_erosion(full)
    edge = ndimage.binary_dilation(edge, iterations=1)
    shown = edge[::display_downsample, ::display_downsample]
    rgba = np.zeros((*shown.shape, 4), dtype=float)
    rgba[shown] = matplotlib.colors.to_rgba(color, alpha=0.95)
    ax.imshow(rgba, interpolation="nearest", origin="upper", label=label)
    ax.plot([], [], color=color, label=label)
    return True


def _scale_mask_to_image_shape(mask: np.ndarray, *, mask_downsample: int, image_shape: tuple[int, int]) -> np.ndarray:
    if mask_downsample <= 0:
        raise ValueError("mask_downsample must be positive")
    full = np.repeat(np.repeat(np.asarray(mask, dtype=bool), mask_downsample, axis=0), mask_downsample, axis=1)
    if full.shape == image_shape:
        return full
    zoom = (image_shape[0] / full.shape[0], image_shape[1] / full.shape[1])
    scaled = ndimage.zoom(full.astype(np.uint8), zoom, order=0) > 0
    if scaled.shape[0] < image_shape[0] or scaled.shape[1] < image_shape[1]:
        padded = np.zeros(image_shape, dtype=bool)
        padded[: scaled.shape[0], : scaled.shape[1]] = scaled
        return padded
    return scaled[: image_shape[0], : image_shape[1]]


def _image_cell(path: str) -> str:
    if not path:
        return "<td>missing</td>"
    return f'<td><a href="{html.escape(path)}"><img src="{html.escape(path)}"></a></td>'
