#!/usr/bin/env python3
"""Evaluate whether cryo-EM denoising residuals contain structured signal.

This script compares matched raw and denoised MRC micrographs by computing either:

    residual = raw - matched_denoised

or, with robust-zscore normalization:

    residual = zscore(raw) - zscore(denoised)

It writes residual MRCs, diagnostic PNG panels, optional coordinate-centered
patch montages, and a TSV of residual/patch metrics.

The test does not require membrane segmentation. If particle/candidate
coordinates are provided, the script compares residual energy at those
coordinates against nearby background patches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "Missing dependency: numpy. Activate the denoising environment or install it locally."
    ) from exc

try:
    import mrcfile
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "Missing dependency: mrcfile. Activate the denoising environment or install it locally."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "Missing dependency: matplotlib. Activate the denoising environment or install it locally."
    ) from exc


@dataclass(frozen=True)
class Pair:
    raw: Path
    denoised: Path
    method: str


@dataclass(frozen=True)
class Coord:
    micrograph_key: str
    x: float
    y: float
    label: str = ""


@dataclass
class Metrics:
    micrograph: str
    method: str
    residual_mrc: str
    raw_mean: float
    raw_std: float
    denoised_mean: float
    denoised_std: float
    residual_mean: float
    residual_std: float
    residual_energy: float
    particle_patch_count: int
    skipped_particle_patch_count: int
    background_patch_count: int
    particle_residual_energy_mean: float
    background_residual_energy_mean: float
    particle_background_energy_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    pairing = parser.add_mutually_exclusive_group(required=True)
    pairing.add_argument(
        "--pairs-tsv",
        type=Path,
        help="TSV with raw, denoised, and optional method columns. Header optional.",
    )
    pairing.add_argument(
        "--raw-dir",
        type=Path,
        help="Directory of raw MRCs. Requires --denoised-dir.",
    )

    parser.add_argument(
        "--denoised-dir",
        type=Path,
        help="Directory of denoised MRCs when using --raw-dir.",
    )
    parser.add_argument(
        "--denoised-suffix",
        default="",
        help=(
            "Suffix to append to raw stem when pairing raw-dir/denoised-dir files. "
            "Example: raw stem foo + suffix _denoised -> foo_denoised.mrc. "
            "Exact filename match is tried first."
        ),
    )
    parser.add_argument(
        "--method",
        default="denoised",
        help="Method name used for raw-dir pairing when no method column exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for residuals, panels, patches, metrics, and config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--normalization",
        choices=["linear-match", "robust-zscore"],
        default="linear-match",
        help="Residual normalization strategy.",
    )
    parser.add_argument(
        "--highpass-sigma",
        type=float,
        default=None,
        help=(
            "Optional Gaussian sigma in pixels for high-pass filtering residuals "
            "before metrics/panels/MRC output. Requires scipy."
        ),
    )
    parser.add_argument(
        "--coords-tsv",
        type=Path,
        help=(
            "Optional TSV with micrograph,x,y[,label] by default, or "
            "micrograph,y,x[,label] with --coord-order yx. Header optional."
        ),
    )
    parser.add_argument(
        "--coord-order",
        choices=["xy", "yx"],
        default="xy",
        help="Coordinate order in --coords-tsv after micrograph column.",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=128,
        help="Patch box size in pixels for coordinate-centered residual analysis.",
    )
    parser.add_argument(
        "--background-offset",
        type=int,
        default=None,
        help=(
            "Offset in pixels for nearby background patches. Default: 3 * --box-size."
        ),
    )
    parser.add_argument(
        "--max-panel-size",
        type=int,
        default=1024,
        help="Maximum displayed image side length for diagnostic PNGs.",
    )

    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contents = [entry for entry in path.iterdir() if entry.name != ".gitkeep"]
    if contents and not overwrite:
        raise SystemExit(
            f"Output directory is not empty: {path}. Use --overwrite to allow writing there."
        )

    (path / "residuals").mkdir(parents=True, exist_ok=True)
    (path / "panels").mkdir(parents=True, exist_ok=True)
    (path / "patches").mkdir(parents=True, exist_ok=True)


def read_mrc(path: Path) -> tuple[np.ndarray, object | None]:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        voxel_size = handle.voxel_size

    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError(
                f"{path} is 3D with shape {data.shape}; this residual script expects 2D micrographs."
            )
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2D MRC after squeezing; shape={data.shape}")
    return data, voxel_size


def write_mrc(path: Path, array: np.ndarray, voxel_size: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.asarray(array, dtype=np.float32))
        if voxel_size is not None:
            try:
                handle.voxel_size = voxel_size
            except Exception:
                pass


def finite_pixels(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def robust_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        raise ValueError("Cannot normalize array with no finite pixels")

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    if sigma < eps:
        sigma = float(np.std(values))
    if sigma < eps:
        sigma = eps
    return (array - med) / sigma


def linear_match_denoised_to_raw(raw: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    mask = finite_pixels(raw, denoised)
    x = denoised[mask].ravel()
    y = raw[mask].ravel()

    if x.size < 2:
        raise ValueError("Not enough finite pixels for linear matching")

    x_std = float(np.std(x))
    if x_std == 0:
        raise ValueError("Denoised image has zero finite-pixel variance")

    a, b = np.polyfit(x, y, deg=1)
    return (a * denoised + b).astype(np.float32)


def maybe_highpass(array: np.ndarray, sigma: float | None) -> np.ndarray:
    if sigma is None:
        return array
    if sigma <= 0:
        raise ValueError("--highpass-sigma must be positive")

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "--highpass-sigma requires scipy, but scipy is not available in this environment."
        ) from exc

    lowpass = gaussian_filter(array, sigma=sigma)
    return array - lowpass


def residual_for_pair(
    raw: np.ndarray,
    denoised: np.ndarray,
    normalization: str,
    highpass_sigma: float | None,
) -> np.ndarray:
    if raw.shape != denoised.shape:
        raise ValueError(f"Shape mismatch: raw {raw.shape}, denoised {denoised.shape}")

    if normalization == "robust-zscore":
        residual = robust_zscore(raw) - robust_zscore(denoised)
    elif normalization == "linear-match":
        matched = linear_match_denoised_to_raw(raw, denoised)
        residual = raw - matched
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    residual = maybe_highpass(residual, highpass_sigma)
    residual = np.where(np.isfinite(residual), residual, 0.0)
    return residual.astype(np.float32)


def image_stats(array: np.ndarray) -> tuple[float, float]:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return math.nan, math.nan
    return float(np.mean(values)), float(np.std(values))


def energy(array: np.ndarray) -> float:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return math.nan
    return float(np.mean(values**2))


def read_pairs_from_tsv(path: Path) -> list[Pair]:
    rows: list[list[str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            row = [cell.strip() for cell in row]
            if row and not row[0].startswith("#"):
                rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    first_lower = [cell.lower() for cell in rows[0]]
    has_header = "raw" in first_lower and "denoised" in first_lower

    pairs: list[Pair] = []
    if has_header:
        header = first_lower
        raw_idx = header.index("raw")
        den_idx = header.index("denoised")
        method_idx = header.index("method") if "method" in header else None
        data_rows = rows[1:]
        for row in data_rows:
            if len(row) <= max(raw_idx, den_idx):
                continue
            method = (
                row[method_idx]
                if method_idx is not None and len(row) > method_idx
                else "denoised"
            )
            pairs.append(
                Pair(
                    Path(row[raw_idx]).expanduser(),
                    Path(row[den_idx]).expanduser(),
                    method,
                )
            )
    else:
        for row in rows:
            if len(row) < 2:
                continue
            method = row[2] if len(row) >= 3 and row[2] else "denoised"
            pairs.append(
                Pair(Path(row[0]).expanduser(), Path(row[1]).expanduser(), method)
            )

    if not pairs:
        raise ValueError(f"No valid pairs found in {path}")
    return pairs


def read_pairs_from_dirs(
    raw_dir: Path,
    denoised_dir: Path | None,
    denoised_suffix: str,
    method: str,
) -> list[Pair]:
    if denoised_dir is None:
        raise SystemExit("--denoised-dir is required with --raw-dir")

    raw_files = sorted(raw_dir.expanduser().glob("*.mrc"))
    if not raw_files:
        raise FileNotFoundError(f"No .mrc files found in raw directory: {raw_dir}")

    pairs: list[Pair] = []
    missing: list[str] = []

    for raw_path in raw_files:
        exact = denoised_dir / raw_path.name
        suffixed = denoised_dir / f"{raw_path.stem}{denoised_suffix}.mrc"

        if exact.exists():
            denoised_path = exact
        elif denoised_suffix and suffixed.exists():
            denoised_path = suffixed
        else:
            missing.append(raw_path.name)
            continue

        pairs.append(Pair(raw_path, denoised_path, method))

    if missing:
        print(
            f"Warning: {len(missing)} raw files had no denoised match. "
            f"First few: {', '.join(missing[:5])}",
            file=sys.stderr,
        )

    if not pairs:
        raise FileNotFoundError("No raw/denoised pairs found")
    return pairs


def micrograph_keys(path: Path) -> set[str]:
    return {path.name, path.stem, str(path)}


def read_coords(path: Path, coord_order: str) -> dict[str, list[Coord]]:
    rows: list[list[str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            row = [cell.strip() for cell in row]
            if row and not row[0].startswith("#"):
                rows.append(row)

    if not rows:
        return {}

    first_lower = [cell.lower() for cell in rows[0]]
    has_header = "micrograph" in first_lower and (
        ("x" in first_lower and "y" in first_lower)
        or ("coord_x" in first_lower and "coord_y" in first_lower)
    )

    coords_by_key: dict[str, list[Coord]] = {}

    if has_header:
        header = first_lower
        mic_idx = header.index("micrograph")
        x_idx = header.index("x") if "x" in header else header.index("coord_x")
        y_idx = header.index("y") if "y" in header else header.index("coord_y")
        label_idx = header.index("label") if "label" in header else None
        data_rows = rows[1:]

        for row in data_rows:
            if len(row) <= max(mic_idx, x_idx, y_idx):
                continue
            label = (
                row[label_idx] if label_idx is not None and len(row) > label_idx else ""
            )
            coord = Coord(row[mic_idx], float(row[x_idx]), float(row[y_idx]), label)
            coords_by_key.setdefault(coord.micrograph_key, []).append(coord)
    else:
        for row in rows:
            if len(row) < 3:
                continue
            mic = row[0]
            first = float(row[1])
            second = float(row[2])
            label = row[3] if len(row) >= 4 else ""

            if coord_order == "xy":
                coord = Coord(mic, first, second, label)
            else:
                coord = Coord(mic, second, first, label)
            coords_by_key.setdefault(coord.micrograph_key, []).append(coord)

    return coords_by_key


def coords_for_micrograph(
    coords_by_key: dict[str, list[Coord]], raw_path: Path
) -> list[Coord]:
    out: list[Coord] = []
    for key in micrograph_keys(raw_path):
        out.extend(coords_by_key.get(key, []))
    return out


def crop_patch(
    array: np.ndarray, x: float, y: float, box_size: int
) -> np.ndarray | None:
    """Return centered patch, or None if the requested patch crosses image bounds."""
    half = box_size // 2
    cx = int(round(x))
    cy = int(round(y))
    y0 = cy - half
    y1 = y0 + box_size
    x0 = cx - half
    x1 = x0 + box_size

    if y0 < 0 or x0 < 0 or y1 > array.shape[0] or x1 > array.shape[1]:
        return None
    return array[y0:y1, x0:x1]


def background_coord(
    x: float,
    y: float,
    image_shape: tuple[int, int],
    box_size: int,
    offset: int,
) -> tuple[float, float] | None:
    """Choose a nearby bounded background coordinate using deterministic offsets."""
    candidates = [
        (x + offset, y),
        (x - offset, y),
        (x, y + offset),
        (x, y - offset),
        (x + offset, y + offset),
        (x - offset, y - offset),
    ]

    for bx, by in candidates:
        if (
            crop_patch(np.zeros(image_shape, dtype=np.float32), bx, by, box_size)
            is not None
        ):
            return bx, by
    return None


def percentile_limits(
    array: np.ndarray, low: float = 1.0, high: float = 99.0
) -> tuple[float, float]:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float(np.min(values)), float(np.max(values) + 1.0)
    return float(vmin), float(vmax)


def downsample_for_panel(array: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return array
    max_dim = max(array.shape)
    if max_dim <= max_side:
        return array

    step = int(math.ceil(max_dim / max_side))
    return array[::step, ::step]


def write_panel(
    path: Path,
    raw: np.ndarray,
    denoised: np.ndarray,
    residual: np.ndarray,
    title: str,
    max_panel_size: int,
) -> None:
    raw_show = downsample_for_panel(raw, max_panel_size)
    den_show = downsample_for_panel(denoised, max_panel_size)
    res_show = downsample_for_panel(residual, max_panel_size)
    abs_show = np.abs(res_show)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("raw", raw_show),
        ("denoised", den_show),
        ("residual", res_show),
        ("abs residual", abs_show),
    ]

    for ax, (name, img) in zip(axes, panels, strict=True):
        vmin, vmax = percentile_limits(img)
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_patch_montage(
    path: Path,
    particle_patches: list[np.ndarray],
    background_patches: list[np.ndarray],
    title: str,
    max_patches: int = 48,
) -> None:
    particle_limit = max_patches // 2
    background_limit = max_patches // 2
    selected = particle_patches[:particle_limit] + background_patches[:background_limit]
    labels = [
        f"particle {idx + 1}"
        for idx in range(min(len(particle_patches), particle_limit))
    ] + [
        f"background {idx + 1}"
        for idx in range(min(len(background_patches), background_limit))
    ]

    if not selected:
        return

    n = len(selected)
    cols = min(8, n)
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, patch, label in zip(axes_array, selected, labels, strict=False):
        vmin, vmax = percentile_limits(patch)
        ax.imshow(patch, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    for ax in axes_array[len(selected) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_patches(
    residual: np.ndarray,
    coords: list[Coord],
    box_size: int,
    background_offset: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    particle_patches: list[np.ndarray] = []
    background_patches: list[np.ndarray] = []
    skipped = 0

    for coord in coords:
        particle_patch = crop_patch(residual, coord.x, coord.y, box_size)
        if particle_patch is None:
            skipped += 1
            continue

        bg_xy = background_coord(
            coord.x,
            coord.y,
            residual.shape,
            box_size,
            background_offset,
        )
        if bg_xy is None:
            skipped += 1
            continue

        background_patch = crop_patch(residual, bg_xy[0], bg_xy[1], box_size)
        if background_patch is None:
            skipped += 1
            continue

        particle_patches.append(particle_patch)
        background_patches.append(background_patch)

    return particle_patches, background_patches, skipped


def safe_name(path: Path, method: str) -> str:
    method_clean = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in method)
    return f"{path.stem}__{method_clean}"


def write_metrics(path: Path, rows: list[Metrics]) -> None:
    fieldnames = (
        list(asdict(rows[0]).keys()) if rows else list(Metrics.__annotations__.keys())
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)

    if args.pairs_tsv:
        pairs = read_pairs_from_tsv(args.pairs_tsv)
    else:
        pairs = read_pairs_from_dirs(
            args.raw_dir,
            args.denoised_dir,
            args.denoised_suffix,
            args.method,
        )

    coords_by_key = (
        read_coords(args.coords_tsv, args.coord_order) if args.coords_tsv else {}
    )
    background_offset = args.background_offset or 3 * args.box_size

    config = vars(args).copy()
    config["pairs_count"] = len(pairs)
    config["background_offset_effective"] = background_offset
    config["pairs"] = [
        {"raw": str(pair.raw), "denoised": str(pair.denoised), "method": pair.method}
        for pair in pairs
    ]
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, default=str)

    metrics_rows: list[Metrics] = []

    for idx, pair in enumerate(pairs, start=1):
        print(
            f"[{idx}/{len(pairs)}] {pair.raw.name} vs {pair.denoised.name} ({pair.method})"
        )

        raw, voxel_size = read_mrc(pair.raw)
        denoised, _ = read_mrc(pair.denoised)
        residual = residual_for_pair(
            raw,
            denoised,
            args.normalization,
            args.highpass_sigma,
        )

        name = safe_name(pair.raw, pair.method)
        residual_path = args.output_dir / "residuals" / f"{name}_residual.mrc"
        panel_path = args.output_dir / "panels" / f"{name}_panel.png"
        patch_path = args.output_dir / "patches" / f"{name}_patches.png"

        write_mrc(residual_path, residual, voxel_size)
        write_panel(
            panel_path,
            raw,
            denoised,
            residual,
            title=f"{pair.raw.name} | {pair.method}",
            max_panel_size=args.max_panel_size,
        )

        coords = coords_for_micrograph(coords_by_key, pair.raw)
        particle_patches: list[np.ndarray] = []
        background_patches: list[np.ndarray] = []
        skipped = 0

        if coords:
            particle_patches, background_patches, skipped = analyze_patches(
                residual,
                coords,
                args.box_size,
                background_offset,
            )
            write_patch_montage(
                patch_path,
                particle_patches,
                background_patches,
                title=f"{pair.raw.name} | {pair.method} residual patches",
            )
            if skipped:
                print(
                    f"  skipped {skipped} coordinate/background patches outside bounds"
                )

        raw_mean, raw_std = image_stats(raw)
        den_mean, den_std = image_stats(denoised)
        res_mean, res_std = image_stats(residual)

        particle_energy_values = [energy(patch) for patch in particle_patches]
        background_energy_values = [energy(patch) for patch in background_patches]

        particle_mean = (
            float(np.mean(particle_energy_values))
            if particle_energy_values
            else math.nan
        )
        background_mean = (
            float(np.mean(background_energy_values))
            if background_energy_values
            else math.nan
        )
        ratio = (
            particle_mean / background_mean
            if np.isfinite(particle_mean)
            and np.isfinite(background_mean)
            and background_mean != 0
            else math.nan
        )

        metrics_rows.append(
            Metrics(
                micrograph=pair.raw.name,
                method=pair.method,
                residual_mrc=str(residual_path),
                raw_mean=raw_mean,
                raw_std=raw_std,
                denoised_mean=den_mean,
                denoised_std=den_std,
                residual_mean=res_mean,
                residual_std=res_std,
                residual_energy=energy(residual),
                particle_patch_count=len(particle_patches),
                skipped_particle_patch_count=skipped,
                background_patch_count=len(background_patches),
                particle_residual_energy_mean=particle_mean,
                background_residual_energy_mean=background_mean,
                particle_background_energy_ratio=ratio,
            )
        )

    metrics_path = args.output_dir / "residual_metrics.tsv"
    write_metrics(metrics_path, metrics_rows)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote residuals: {args.output_dir / 'residuals'}")
    print(f"Wrote panels: {args.output_dir / 'panels'}")
    if coords_by_key:
        print(f"Wrote patch montages: {args.output_dir / 'patches'}")


if __name__ == "__main__":
    main()
