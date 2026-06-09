"""Utilities for CryoSPARC particle-location result groups."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_LOCATION_FIELDS = {
    "location/micrograph_uid",
    "location/micrograph_path",
    "location/micrograph_shape",
    "location/center_x_frac",
    "location/center_y_frac",
}

IMAGE_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff"}
CRYOSPARC_EXPORT_PREFIX_RE = re.compile(r"^\d+_(.+)$")


def decode_path(value: Any) -> str:
    """Decode CryoSPARC byte-string paths into plain strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def load_particles_cs(path: Path) -> np.ndarray:
    """Load a CryoSPARC structured `.cs` array and validate location fields."""
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.names is None:
        raise ValueError(f"{path} is not a structured CryoSPARC .cs array")
    missing = sorted(REQUIRED_LOCATION_FIELDS - set(arr.dtype.names))
    if missing:
        raise ValueError(f"{path} missing required fields: {', '.join(missing)}")
    return arr


def load_particles_cs_many(paths: list[Path] | tuple[Path, ...]) -> np.ndarray:
    """Load and concatenate one or more compatible CryoSPARC particle `.cs` files."""
    if not paths:
        raise ValueError("At least one particle .cs file is required")
    arrays = [load_particles_cs(path) for path in paths]
    dtype = arrays[0].dtype
    mismatched = [str(path) for path, arr in zip(paths, arrays) if arr.dtype != dtype]
    if mismatched:
        raise ValueError("Particle .cs files have incompatible schemas: " + ", ".join(mismatched))
    return np.concatenate(arrays)


def write_cs(path: Path, rows: np.ndarray) -> None:
    """Write a structured array to a `.cs` path without appending `.npy`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, rows)


def build_micrograph_index(micrographs_dir: Path) -> dict[str, Path]:
    """Index micrographs by extensionless filename stem."""
    index: dict[str, Path] = {}
    conflicts: dict[str, list[Path]] = {}
    for path in micrographs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in index:
            conflicts.setdefault(stem, [index[stem]]).append(path)
        else:
            index[stem] = path
    if conflicts:
        formatted = "\n".join(
            f"{stem}: " + ", ".join(str(path) for path in paths)
            for stem, paths in sorted(conflicts.items())
        )
        raise ValueError(f"Duplicate micrograph stems under {micrographs_dir}:\n{formatted}")
    return index


def resolve_micrograph_path(
    cryosparc_micrograph_path: str,
    micrographs_dir: Path,
    micrograph_index: dict[str, Path],
) -> Path:
    """Resolve a CryoSPARC micrograph path against an export directory."""
    raw = Path(cryosparc_micrograph_path)
    for candidate in (micrographs_dir / raw, micrographs_dir / raw.name):
        if candidate.exists():
            return candidate
    indexed = micrograph_index.get(raw.stem)
    if indexed is not None:
        return indexed
    normalized_stem = strip_numeric_export_prefix(raw.stem)
    if normalized_stem != raw.stem:
        indexed = micrograph_index.get(normalized_stem)
        if indexed is not None:
            return indexed
    raise FileNotFoundError(
        f"Could not resolve micrograph {cryosparc_micrograph_path!r} under {micrographs_dir}"
    )


def strip_numeric_export_prefix(stem: str) -> str:
    """Drop CryoSPARC export prefixes like 003884..._ before FoilHole names."""
    match = CRYOSPARC_EXPORT_PREFIX_RE.match(stem.lstrip(">"))
    return match.group(1) if match else stem.lstrip(">")


def patch_csg_cs_path(
    source_csg: Path,
    output_csg: Path,
    original_cs_name: str,
    new_cs_path: Path,
) -> None:
    """Copy a `.csg` file, replacing exactly one referenced `.cs` path."""
    text = source_csg.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        matches = _find_text_cs_references(text, original_cs_name)
        count = len(matches)
        if count == 0:
            raise ValueError(f"No .cs reference matching {original_cs_name!r} found in {source_csg}")
        if len(set(matches)) > 1:
            raise ValueError(f"Ambiguous .cs references matching {original_cs_name!r} found in {source_csg}")
        patched_text = text.replace(matches[0], str(new_cs_path))
        output_csg.parent.mkdir(parents=True, exist_ok=True)
        output_csg.write_text(patched_text)
        return
    patched, count = _replace_cs_reference(data, original_cs_name, str(new_cs_path))
    if count == 0:
        raise ValueError(f"No .cs reference matching {original_cs_name!r} found in {source_csg}")
    if count > 1:
        raise ValueError(f"Ambiguous .cs references matching {original_cs_name!r} found in {source_csg}")
    output_csg.parent.mkdir(parents=True, exist_ok=True)
    output_csg.write_text(json.dumps(patched, indent=2) + "\n")


def _find_text_cs_references(text: str, original_cs_name: str) -> list[str]:
    pattern = re.compile(rf">?[^\s'\"\]]*{re.escape(original_cs_name)}")
    return pattern.findall(text)


def _replace_cs_reference(value: Any, original_cs_name: str, new_cs_path: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        out = {}
        count = 0
        for key, item in value.items():
            patched, item_count = _replace_cs_reference(item, original_cs_name, new_cs_path)
            out[key] = patched
            count += item_count
        return out, count
    if isinstance(value, list):
        out = []
        count = 0
        for item in value:
            patched, item_count = _replace_cs_reference(item, original_cs_name, new_cs_path)
            out.append(patched)
            count += item_count
        return out, count
    if isinstance(value, str) and Path(value).name == original_cs_name:
        return new_cs_path, 1
    return copy.deepcopy(value), 0
