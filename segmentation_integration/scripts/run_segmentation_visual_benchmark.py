#!/usr/bin/env python3
"""Run a visual CryoVIA-vs-SAM segmentation benchmark."""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation_integration.cryovia_runner import DEFAULT_CRYOVIA_MODEL, default_cryovia_home, run_cryovia_segmentation
from segmentation_integration.scripts.run_local_vesicle_picker import run_local_vesicle_picker
from segmentation_integration.segmentation_preview import (
    load_cryovia_label_mask,
    load_vesicle_picker_union_mask,
    render_segmentation_comparison,
    render_segmentation_overlay,
    write_review_index,
)


UID_PREFIX_RE = re.compile(r"^0*(\d+)_")


@dataclass(frozen=True)
class BenchmarkRecord:
    uid: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vesicle-picker-dir", type=Path, required=True)
    parser.add_argument("--sam-find-parameters", type=Path, required=True)
    parser.add_argument("--sam-filter-parameters", type=Path, required=True)
    parser.add_argument("--cryovia-model", default=DEFAULT_CRYOVIA_MODEL)
    parser.add_argument(
        "--cryovia-home",
        type=Path,
        default=None,
        help=(
            "Runtime home for CryoVIA .cryovia files. Defaults to CRYOVIA_HOME, "
            "or /nfs/bartesaghilab2/zy223/cryovia_runtime_home when available."
        ),
    )
    parser.add_argument("--pixel-size-a", type=float, default=7.0)
    parser.add_argument("--sam-downsample", type=int, default=4)
    parser.add_argument("--cryovia-downsample", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--njobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sam-run", action="store_true", help="Reuse existing sam_found/sam_filtered outputs.")
    parser.add_argument("--skip-cryovia-run", action="store_true", help="Reuse existing cryovia_labels outputs.")
    return parser.parse_args()


def parse_manifest(manifest: Path) -> list[BenchmarkRecord]:
    records = []
    for raw_path in _manifest_paths(manifest):
        records.append(BenchmarkRecord(uid=_uid_from_filename(raw_path), path=raw_path))
    seen: dict[int, Path] = {}
    for record in records:
        existing = seen.get(record.uid)
        if existing is not None and existing != record.path:
            raise ValueError(f"Manifest maps UID {record.uid} to multiple paths: {existing} and {record.path}")
        seen[record.uid] = record.path
    return sorted(records, key=lambda record: (record.uid, record.path.name))


def run_visual_benchmark(
    *,
    manifest: Path,
    output_dir: Path,
    vesicle_picker_dir: Path,
    sam_find_parameters: Path,
    sam_filter_parameters: Path,
    cryovia_model: str = DEFAULT_CRYOVIA_MODEL,
    cryovia_home: Path | None = None,
    pixel_size_a: float = 7.0,
    sam_downsample: int = 4,
    cryovia_downsample: int = 1,
    limit: int = 0,
    skip_existing: bool = False,
    gpus: str = "",
    njobs: int = 2,
    dry_run: bool = False,
    skip_sam_run: bool = False,
    skip_cryovia_run: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    cryovia_labels = output_dir / "cryovia_labels"
    sam_found = output_dir / "sam_found"
    sam_filtered = output_dir / "sam_filtered"
    overlays = output_dir / "overlays"
    for directory in (cryovia_labels, sam_found, sam_filtered, overlays):
        directory.mkdir(parents=True, exist_ok=True)

    records = parse_manifest(manifest)
    if limit > 0:
        records = records[:limit]
    _write_manifest(output_dir / "manifest.tsv", records)
    resolved_cryovia_home = default_cryovia_home() if cryovia_home is None else Path(cryovia_home).expanduser()
    config = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "vesicle_picker_dir": str(vesicle_picker_dir),
        "sam_find_parameters": str(sam_find_parameters),
        "sam_filter_parameters": str(sam_filter_parameters),
        "cryovia_model": cryovia_model,
        "cryovia_home": str(resolved_cryovia_home),
        "pixel_size_a": pixel_size_a,
        "sam_downsample": sam_downsample,
        "cryovia_downsample": cryovia_downsample,
        "limit": limit,
        "skip_existing": skip_existing,
        "gpus": gpus,
        "njobs": njobs,
        "dry_run": dry_run,
        "skip_sam_run": skip_sam_run,
        "skip_cryovia_run": skip_cryovia_run,
    }
    _write_json(output_dir / "config.json", config)

    sam_summary = {"mode": "dry_run"}
    cryovia_summary = {"mode": "dry_run"}
    if not dry_run:
        if skip_cryovia_run:
            cryovia_summary = {"mode": "skipped", "label_files": [str(path) for path in sorted(cryovia_labels.glob("*_labels.npz"))]}
        else:
            cryovia_result = run_cryovia_segmentation(
                micrograph_paths=[record.path for record in records],
                output_dir=cryovia_labels,
                cryovia_model=cryovia_model,
                cryovia_home=resolved_cryovia_home,
                pixel_size_a=pixel_size_a,
                gpus=gpus,
                njobs=njobs,
                skip_existing=skip_existing,
            )
            cryovia_summary = {
                "model": cryovia_result.model,
                "micrographs_seen": cryovia_result.micrographs_seen,
                "micrographs_processed": cryovia_result.micrographs_processed,
                "label_files": [str(path) for path in cryovia_result.label_files],
            }
        if skip_sam_run:
            sam_summary = {
                "mode": "skipped",
                "found_mask_files": len(list(sam_found.glob("*_vesicles.pkl"))),
                "filtered_mask_files": len(list(sam_filtered.glob("*_vesicles_filtered.pkl"))),
            }
        else:
            staged_find_parameters = output_dir / "parameters" / "find_vesicles.ini"
            staged_filter_parameters = output_dir / "parameters" / "filter_vesicles.ini"
            _stage_sam_parameters(
                find_parameters=sam_find_parameters,
                filter_parameters=sam_filter_parameters,
                staged_find_parameters=staged_find_parameters,
                staged_filter_parameters=staged_filter_parameters,
                sam_found=sam_found,
                sam_filtered=sam_filtered,
            )
            sam_summary = run_local_vesicle_picker(
                vesicle_picker_dir=vesicle_picker_dir,
                manifest=output_dir / "manifest.tsv",
                find_parameters=staged_find_parameters,
                filter_parameters=staged_filter_parameters,
                skip_existing=skip_existing,
                progress_every=1,
            )

    review_rows = []
    if not dry_run:
        review_rows = _render_overlays(
            records=records,
            output_dir=output_dir,
            cryovia_labels=cryovia_labels,
            sam_found=sam_found,
            sam_filtered=sam_filtered,
            overlays=overlays,
            cryovia_model=cryovia_model,
            cryovia_downsample=cryovia_downsample,
            sam_downsample=sam_downsample,
        )
    write_review_index(
        output_path=output_dir / "index.html",
        rows=review_rows,
        title=f"CryoVIA {cryovia_model} vs SAM visual benchmark",
    )
    summary = {
        "mode": "dry_run" if dry_run else "run",
        "micrographs": len(records),
        "cryovia": cryovia_summary,
        "sam": sam_summary,
        "overlays": len(review_rows),
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _manifest_paths(manifest: Path) -> list[Path]:
    manifest = Path(manifest).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    paths = []
    with manifest.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split("\t", 1)[0].strip()
            if line_number == 1 and first.lower() in {"path", "mrc", "micrograph", "micrograph_path"}:
                continue
            path = Path(first).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Manifest path does not exist: {path}")
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No micrograph paths found in manifest: {manifest}")
    return paths


def _uid_from_filename(path: Path) -> int:
    match = UID_PREFIX_RE.match(path.name)
    if match is None:
        raise ValueError(f"Manifest path must start with a leading CryoSPARC UID prefix: {path}")
    return int(match.group(1))


def _write_manifest(path: Path, records: list[BenchmarkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["micrograph_path\tuid"]
    lines.extend(f"{record.path}\t{record.uid}" for record in records)
    path.write_text("\n".join(lines) + "\n")


def _render_overlays(
    *,
    records: list[BenchmarkRecord],
    output_dir: Path,
    cryovia_labels: Path,
    sam_found: Path,
    sam_filtered: Path,
    overlays: Path,
    cryovia_model: str,
    cryovia_downsample: int,
    sam_downsample: int,
) -> list[dict[str, str]]:
    rows = []
    for record in records:
        stem = record.path.stem
        cryovia_path = cryovia_labels / f"{stem}_labels.npz"
        found_path = sam_found / f"{record.uid}_vesicles.pkl"
        filtered_path = sam_filtered / f"{record.uid}_vesicles_filtered.pkl"
        cryovia_mask = load_cryovia_label_mask(cryovia_path) if cryovia_path.exists() else None
        found_mask = load_vesicle_picker_union_mask(found_path) if found_path.exists() else None
        filtered_mask = load_vesicle_picker_union_mask(filtered_path) if filtered_path.exists() else None
        cryovia_png = overlays / f"{stem}_cryovia_{_slug(cryovia_model)}.png"
        found_png = overlays / f"{stem}_sam_found.png"
        filtered_png = overlays / f"{stem}_sam_filtered.png"
        comparison_png = overlays / f"{stem}_comparison.png"
        if cryovia_mask is not None:
            render_segmentation_overlay(
                micrograph_path=record.path,
                output_path=cryovia_png,
                masks=[cryovia_mask],
                labels=[f"CryoVIA {cryovia_model}"],
                colors=["lime"],
                title=f"{stem} CryoVIA {cryovia_model}",
                mask_downsample=cryovia_downsample,
            )
        if found_mask is not None:
            render_segmentation_overlay(
                micrograph_path=record.path,
                output_path=found_png,
                masks=[found_mask],
                labels=["SAM found"],
                colors=["orange"],
                title=f"{stem} SAM found",
                mask_downsample=sam_downsample,
            )
        if filtered_mask is not None:
            render_segmentation_overlay(
                micrograph_path=record.path,
                output_path=filtered_png,
                masks=[filtered_mask],
                labels=["SAM filtered"],
                colors=["cyan"],
                title=f"{stem} SAM filtered",
                mask_downsample=sam_downsample,
            )
        render_segmentation_comparison(
            micrograph_path=record.path,
            output_path=comparison_png,
            cryovia_mask=cryovia_mask,
            sam_found_mask=found_mask,
            sam_filtered_mask=filtered_mask,
            mask_downsample=max(cryovia_downsample, sam_downsample),
            cryovia_mask_downsample=cryovia_downsample,
            sam_mask_downsample=sam_downsample,
            title=f"{stem} comparison",
        )
        rows.append(
            {
                "micrograph": record.path.name,
                "comparison": _relative(output_dir, comparison_png),
                "cryovia": _relative(output_dir, cryovia_png) if cryovia_png.exists() else "",
                "sam_found": _relative(output_dir, found_png) if found_png.exists() else "",
                "sam_filtered": _relative(output_dir, filtered_png) if filtered_png.exists() else "",
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stage_sam_parameters(
    *,
    find_parameters: Path,
    filter_parameters: Path,
    staged_find_parameters: Path,
    staged_filter_parameters: Path,
    sam_found: Path,
    sam_filtered: Path,
) -> None:
    staged_find_parameters.parent.mkdir(parents=True, exist_ok=True)
    find_config = _read_config(find_parameters)
    filter_config = _read_config(filter_parameters)
    _require_section(find_config, "output", find_parameters)
    _require_section(filter_config, "input", filter_parameters)
    _require_section(filter_config, "output", filter_parameters)
    find_config["output"]["directory"] = _ensure_trailing_slash(sam_found)
    filter_config["input"]["directory"] = _ensure_trailing_slash(sam_found)
    filter_config["output"]["directory"] = _ensure_trailing_slash(sam_filtered)
    with staged_find_parameters.open("w") as handle:
        find_config.write(handle)
    with staged_filter_parameters.open("w") as handle:
        filter_config.write(handle)


def _read_config(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    with Path(path).open() as handle:
        parser.read_file(handle)
    return parser


def _require_section(parser: configparser.ConfigParser, section: str, source: Path) -> None:
    if not parser.has_section(section):
        raise ValueError(f"{source} is missing required [{section}] section")


def _ensure_trailing_slash(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    args = parse_args()
    summary = run_visual_benchmark(
        manifest=args.manifest,
        output_dir=args.output_dir,
        vesicle_picker_dir=args.vesicle_picker_dir,
        sam_find_parameters=args.sam_find_parameters,
        sam_filter_parameters=args.sam_filter_parameters,
        cryovia_model=args.cryovia_model,
        cryovia_home=args.cryovia_home,
        pixel_size_a=args.pixel_size_a,
        sam_downsample=args.sam_downsample,
        cryovia_downsample=args.cryovia_downsample,
        limit=args.limit,
        skip_existing=args.skip_existing,
        gpus=args.gpus,
        njobs=args.njobs,
        dry_run=args.dry_run,
        skip_sam_run=args.skip_sam_run,
        skip_cryovia_run=args.skip_cryovia_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
