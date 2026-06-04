"""Vesicle Picker mask reconstruction and membrane-band helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


def load_vesicle_masks(path: Path) -> list[dict[str, Any]]:
    """Load Vesicle Picker's prime-encoded mask pickle format."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("composite_mask") is None:
        return []
    composite = payload["composite_mask"]
    masks = []
    for mask in payload.get("masks", []):
        if "prime_key" not in mask:
            raise ValueError(f"Mask in {path} is missing prime_key")
        reconstructed = dict(mask)
        reconstructed["segmentation"] = composite % mask["prime_key"] == 0
        masks.append(reconstructed)
    return masks


def mask_path_for_uid(mask_dir: Path, micrograph_uid: int) -> Path:
    """Return the CryoSPARC-style Vesicle Picker filtered-mask path."""
    return mask_dir / f"{micrograph_uid}_vesicles_filtered.pkl"


def build_membrane_band(masks: list[dict[str, Any]], radius_px: int) -> np.ndarray:
    """Build a boolean contour band from Vesicle Picker segmentation masks."""
    if not masks:
        raise ValueError("Cannot build membrane band from an empty mask list")
    radius_px = max(0, int(radius_px))
    first_shape = np.asarray(masks[0]["segmentation"]).shape
    band = np.zeros(first_shape, dtype=bool)
    for mask in masks:
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        if segmentation.shape != first_shape:
            raise ValueError("All masks for a micrograph must have the same shape")
        if not segmentation.any():
            continue
        edge = segmentation & ~ndimage.binary_erosion(segmentation)
        if radius_px > 0:
            edge = ndimage.binary_dilation(edge, iterations=radius_px)
        band |= edge
    return band


def membrane_radius_px(membrane_band_a: float, psize_a: float, downsample: int) -> int:
    """Convert membrane band width in Angstroms to downsampled mask pixels."""
    if membrane_band_a < 0:
        raise ValueError("--membrane-band-a must be non-negative")
    if psize_a <= 0:
        raise ValueError("--psize-a must be positive")
    if downsample <= 0:
        raise ValueError("--downsample must be positive")
    return int(np.round(membrane_band_a / (psize_a * downsample)))

