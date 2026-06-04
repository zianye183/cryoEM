#!/usr/bin/env python3
"""Evaluate whether pretrained denoising helps Difference-of-Gaussians picking.

This is a lightweight, label-free screen before running a full downstream picker or
2D-classification comparison. It applies the same DoG detector to raw and denoised
micrographs, then asks whether denoised-assisted picks are stronger/more stable
when scored on the raw micrograph.

Outputs:
    dog_picks.tsv      per-pick coordinates and scores
    dog_metrics.tsv    per-micrograph/per-method summary metrics
    dog_summary.tsv    aggregate method-level summary metrics
    run_config.json    command provenance

Interpretation is comparative, not absolute. Without ground-truth Prestin labels,
use this to decide whether denoising is promising enough for downstream raw-data
extraction and 2D-class validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: numpy") from exc

try:
    import mrcfile
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: mrcfile") from exc

try:
    from scipy import ndimage
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency: scipy. Use the project DRACO venv or install scipy locally."
    ) from exc


@dataclass(frozen=True)
class MethodSpec:
    name: str
    directory: Path
    suffix: str = ""


@dataclass(frozen=True)
class ImageResult:
    coords_xy: np.ndarray
    detector_scores: np.ndarray
    raw_scores: np.ndarray
    score_threshold: float
    response_median: float
    response_mad_sigma: float


@dataclass(frozen=True)
class MetricsRow:
    micrograph: str
    method: str
    image_path: str
    raw_pick_count: int
    method_pick_count: int
    matched_to_raw_count: int
    matched_to_raw_fraction: float
    median_nearest_raw_distance_px: float
    method_score_median: float
    method_score_p90: float
    method_raw_score_median: float
    method_raw_score_p90: float
    raw_pick_raw_score_median: float
    raw_pick_raw_score_p90: float
    method_vs_raw_raw_score_median_delta: float
    method_vs_raw_raw_score_p90_delta: float
    score_threshold: float


def parse_method_spec(text: str) -> MethodSpec:
    """Parse name:path[:suffix], allowing colons only as separators."""
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "Method specs must be name:/path/to/denoised_dir[:suffix]"
        )
    name, directory = parts[0], Path(parts[1])
    suffix = parts[2] if len(parts) == 3 else ""
    if not name:
        raise argparse.ArgumentTypeError("Method name cannot be empty")
    return MethodSpec(name=name, directory=directory, suffix=suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Directory of raw MRCs"
    )
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method_spec,
        default=[],
        help=(
            "Denoised method as name:/path/to/dir[:suffix]. Exact raw filename is tried first; "
            "then raw_stem + suffix + .mrc is tried. Repeat for multiple pretrained methods."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow non-empty output dir"
    )
    parser.add_argument("--glob", default="*.mrc", help="Raw MRC glob within --raw-dir")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional max raw micrographs"
    )
    parser.add_argument(
        "--sigma-small", type=float, default=3.0, help="Small DoG sigma in px"
    )
    parser.add_argument(
        "--sigma-large", type=float, default=9.0, help="Large DoG sigma in px"
    )
    parser.add_argument(
        "--polarity",
        choices=["bright", "dark", "absolute"],
        default="absolute",
        help="Blob polarity after normalization. absolute is safest for uncertain contrast inversion.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Robust z-score threshold on DoG response",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=32,
        help="Minimum local-maximum spacing in pixels",
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=2000,
        help="Maximum retained picks per image after score sorting",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=48.0,
        help="Nearest-neighbor radius for matching denoised picks to raw picks",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contents = [entry for entry in path.iterdir() if entry.name != ".gitkeep"]
    if contents and not overwrite:
        raise SystemExit(f"Output directory is not empty: {path}. Use --overwrite.")


def read_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D MRC: {path}; got shape {data.shape}")
    return data


def robust_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image has no finite pixels")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if sigma < eps:
        sigma = float(np.std(finite))
    if sigma < eps:
        sigma = eps
    return (array - median) / sigma


def dog_response(
    image: np.ndarray, sigma_small: float, sigma_large: float, polarity: str
) -> np.ndarray:
    norm = robust_zscore(image)
    low = ndimage.gaussian_filter(norm, sigma=sigma_large, mode="nearest")
    high = ndimage.gaussian_filter(norm, sigma=sigma_small, mode="nearest")
    response = high - low
    if polarity == "dark":
        response = -response
    elif polarity == "absolute":
        response = np.abs(response)
    return np.asarray(response, dtype=np.float32)


def robust_location_scale(array: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad
    if sigma < eps:
        sigma = float(np.std(finite))
    if sigma < eps:
        sigma = eps
    return median, sigma


def detect_peaks(
    response: np.ndarray, threshold_sigma: float, min_distance: int, max_picks: int
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    median, mad_sigma = robust_location_scale(response)
    threshold = median + threshold_sigma * mad_sigma
    size = max(3, 2 * int(min_distance) + 1)
    max_filtered = ndimage.maximum_filter(response, size=size, mode="nearest")
    mask = (response == max_filtered) & (response >= threshold) & np.isfinite(response)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            threshold,
            median,
            mad_sigma,
        )
    scores = response[ys, xs]
    order = np.argsort(scores)[::-1]
    if max_picks > 0:
        order = order[:max_picks]
    coords_xy = np.column_stack([xs[order], ys[order]]).astype(np.float32)
    return coords_xy, scores[order].astype(np.float32), threshold, median, mad_sigma


def sample_response(response: np.ndarray, coords_xy: np.ndarray) -> np.ndarray:
    if coords_xy.size == 0:
        return np.empty((0,), dtype=np.float32)
    xs = np.clip(np.rint(coords_xy[:, 0]).astype(int), 0, response.shape[1] - 1)
    ys = np.clip(np.rint(coords_xy[:, 1]).astype(int), 0, response.shape[0] - 1)
    return response[ys, xs].astype(np.float32)


def run_detector(
    image: np.ndarray,
    raw_response: np.ndarray,
    sigma_small: float,
    sigma_large: float,
    polarity: str,
    threshold_sigma: float,
    min_distance: int,
    max_picks: int,
) -> ImageResult:
    response = dog_response(image, sigma_small, sigma_large, polarity)
    coords, scores, threshold, median, mad_sigma = detect_peaks(
        response, threshold_sigma, min_distance, max_picks
    )
    raw_scores = sample_response(raw_response, coords)
    return ImageResult(
        coords_xy=coords,
        detector_scores=scores,
        raw_scores=raw_scores,
        score_threshold=threshold,
        response_median=median,
        response_mad_sigma=mad_sigma,
    )


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def median_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.median(values))


def match_to_raw(
    method_coords: np.ndarray, raw_coords: np.ndarray, radius: float
) -> tuple[int, float, float]:
    if method_coords.size == 0 or raw_coords.size == 0:
        return 0, 0.0, math.nan
    tree = cKDTree(raw_coords)
    distances, _indices = tree.query(method_coords, k=1)
    matched = distances <= radius
    return int(np.sum(matched)), float(np.mean(matched)), median_or_nan(distances)


def find_denoised(raw_path: Path, spec: MethodSpec) -> Path | None:
    exact = spec.directory / raw_path.name
    if exact.exists():
        return exact
    suffixed = spec.directory / f"{raw_path.stem}{spec.suffix}{raw_path.suffix}"
    if suffixed.exists():
        return suffixed
    return None


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(metrics: list[MetricsRow]) -> list[dict]:
    by_method: dict[str, list[MetricsRow]] = {}
    for row in metrics:
        by_method.setdefault(row.method, []).append(row)

    summary = []
    for method, rows in sorted(by_method.items()):

        def vals(attr: str) -> np.ndarray:
            arr = np.asarray([getattr(row, attr) for row in rows], dtype=float)
            return arr[np.isfinite(arr)]

        summary.append(
            {
                "method": method,
                "micrograph_count": len(rows),
                "mean_pick_count": float(np.mean(vals("method_pick_count")))
                if rows
                else math.nan,
                "median_pick_count": float(np.median(vals("method_pick_count")))
                if rows
                else math.nan,
                "mean_matched_to_raw_fraction": float(
                    np.mean(vals("matched_to_raw_fraction"))
                )
                if vals("matched_to_raw_fraction").size
                else math.nan,
                "median_nearest_raw_distance_px": float(
                    np.median(vals("median_nearest_raw_distance_px"))
                )
                if vals("median_nearest_raw_distance_px").size
                else math.nan,
                "median_raw_score_delta": float(
                    np.median(vals("method_vs_raw_raw_score_median_delta"))
                )
                if vals("method_vs_raw_raw_score_median_delta").size
                else math.nan,
                "p90_raw_score_delta": float(
                    np.median(vals("method_vs_raw_raw_score_p90_delta"))
                )
                if vals("method_vs_raw_raw_score_p90_delta").size
                else math.nan,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)

    raw_paths = sorted(args.raw_dir.glob(args.glob))
    if args.limit is not None:
        raw_paths = raw_paths[: args.limit]
    if not raw_paths:
        raise SystemExit(f"No raw MRCs found in {args.raw_dir} with glob {args.glob}")

    methods = [MethodSpec("raw", args.raw_dir, "")] + args.method
    metric_rows: list[MetricsRow] = []
    pick_rows: list[dict] = []
    missing: list[dict] = []

    for raw_path in raw_paths:
        raw_image = read_mrc(raw_path)
        raw_response = dog_response(
            raw_image, args.sigma_small, args.sigma_large, args.polarity
        )
        raw_result = run_detector(
            raw_image,
            raw_response,
            args.sigma_small,
            args.sigma_large,
            args.polarity,
            args.threshold_sigma,
            args.min_distance,
            args.max_picks,
        )
        raw_median = median_or_nan(raw_result.raw_scores)
        raw_p90 = percentile_or_nan(raw_result.raw_scores, 90)

        results_by_method = {"raw": (raw_path, raw_result)}
        for spec in args.method:
            denoised_path = find_denoised(raw_path, spec)
            if denoised_path is None:
                missing.append(
                    {
                        "micrograph": raw_path.name,
                        "method": spec.name,
                        "directory": str(spec.directory),
                    }
                )
                continue
            denoised_image = read_mrc(denoised_path)
            if denoised_image.shape != raw_image.shape:
                missing.append(
                    {
                        "micrograph": raw_path.name,
                        "method": spec.name,
                        "directory": str(spec.directory),
                        "reason": f"shape mismatch raw={raw_image.shape} denoised={denoised_image.shape}",
                    }
                )
                continue
            results_by_method[spec.name] = (
                denoised_path,
                run_detector(
                    denoised_image,
                    raw_response,
                    args.sigma_small,
                    args.sigma_large,
                    args.polarity,
                    args.threshold_sigma,
                    args.min_distance,
                    args.max_picks,
                ),
            )

        for method_name, (image_path, result) in results_by_method.items():
            matched_count, matched_fraction, nn_median = match_to_raw(
                result.coords_xy, raw_result.coords_xy, args.match_radius
            )
            method_median = median_or_nan(result.detector_scores)
            method_p90 = percentile_or_nan(result.detector_scores, 90)
            method_raw_median = median_or_nan(result.raw_scores)
            method_raw_p90 = percentile_or_nan(result.raw_scores, 90)
            metric_rows.append(
                MetricsRow(
                    micrograph=raw_path.name,
                    method=method_name,
                    image_path=str(image_path),
                    raw_pick_count=int(raw_result.coords_xy.shape[0]),
                    method_pick_count=int(result.coords_xy.shape[0]),
                    matched_to_raw_count=matched_count,
                    matched_to_raw_fraction=matched_fraction,
                    median_nearest_raw_distance_px=nn_median,
                    method_score_median=method_median,
                    method_score_p90=method_p90,
                    method_raw_score_median=method_raw_median,
                    method_raw_score_p90=method_raw_p90,
                    raw_pick_raw_score_median=raw_median,
                    raw_pick_raw_score_p90=raw_p90,
                    method_vs_raw_raw_score_median_delta=method_raw_median - raw_median,
                    method_vs_raw_raw_score_p90_delta=method_raw_p90 - raw_p90,
                    score_threshold=float(result.score_threshold),
                )
            )
            for rank, ((x, y), detector_score, raw_score) in enumerate(
                zip(result.coords_xy, result.detector_scores, result.raw_scores),
                start=1,
            ):
                pick_rows.append(
                    {
                        "micrograph": raw_path.name,
                        "method": method_name,
                        "rank": rank,
                        "x": float(x),
                        "y": float(y),
                        "detector_score": float(detector_score),
                        "raw_dog_score_at_pick": float(raw_score),
                    }
                )

    write_tsv(
        args.output_dir / "dog_metrics.tsv",
        [asdict(row) for row in metric_rows],
        list(MetricsRow.__dataclass_fields__.keys()),
    )
    write_tsv(
        args.output_dir / "dog_picks.tsv",
        pick_rows,
        [
            "micrograph",
            "method",
            "rank",
            "x",
            "y",
            "detector_score",
            "raw_dog_score_at_pick",
        ],
    )
    write_tsv(
        args.output_dir / "dog_summary.tsv",
        summarize(metric_rows),
        [
            "method",
            "micrograph_count",
            "mean_pick_count",
            "median_pick_count",
            "mean_matched_to_raw_fraction",
            "median_nearest_raw_distance_px",
            "median_raw_score_delta",
            "p90_raw_score_delta",
        ],
    )
    if missing:
        write_tsv(
            args.output_dir / "missing_inputs.tsv",
            missing,
            sorted({key for row in missing for key in row.keys()}),
        )

    config = {
        **vars(args),
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "method": [
            asdict(spec) | {"directory": str(spec.directory)} for spec in args.method
        ],
        "raw_count": len(raw_paths),
        "metrics_count": len(metric_rows),
        "missing_count": len(missing),
    }
    with (args.output_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(f"Wrote {args.output_dir / 'dog_summary.tsv'}")
    print(f"Wrote {args.output_dir / 'dog_metrics.tsv'}")
    print(f"Wrote {args.output_dir / 'dog_picks.tsv'}")
    if missing:
        print(
            f"Warning: {len(missing)} missing/skipped method inputs; see missing_inputs.tsv"
        )


if __name__ == "__main__":
    main()
