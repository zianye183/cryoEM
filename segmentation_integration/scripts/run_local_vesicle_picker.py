#!/usr/bin/env python3
"""Generate Vesicle Picker masks from local MRC micrographs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from segmentation_integration.cryosparc_io import (
    build_micrograph_index,
    decode_path,
    load_particles_cs,
    resolve_micrograph_path,
)
from segmentation_integration.mrc_io import read_mode2_mrc


@dataclass(frozen=True)
class MicrographRecord:
    uid: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vesicle-picker-dir", type=Path, required=True)
    parser.add_argument("--particles-cs", type=Path, required=True)
    parser.add_argument("--micrographs-dir", type=Path, required=True)
    parser.add_argument("--find-parameters", type=Path, required=True)
    parser.add_argument("--filter-parameters", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Maximum micrographs to process; 0 means all.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Write progress every N micrographs; 0 disables progress lines.",
    )
    return parser.parse_args()


def micrograph_records_from_particles(particles: np.ndarray, micrographs_dir: Path) -> list[MicrographRecord]:
    micrograph_index = build_micrograph_index(micrographs_dir)
    uid_to_path: dict[int, Path] = {}
    uid_to_source: dict[int, str] = {}
    for particle in particles:
        uid = int(particle["location/micrograph_uid"])
        source_path = decode_path(particle["location/micrograph_path"])
        existing_source = uid_to_source.get(uid)
        if existing_source is not None and Path(existing_source).stem != Path(source_path).stem:
            raise ValueError(f"UID {uid} maps to multiple micrograph paths: {existing_source!r}, {source_path!r}")
        uid_to_source[uid] = source_path
        uid_to_path.setdefault(uid, resolve_micrograph_path(source_path, micrographs_dir, micrograph_index))
    return [MicrographRecord(uid=uid, path=uid_to_path[uid]) for uid in sorted(uid_to_path)]


def run_local_vesicle_picker(
    *,
    vesicle_picker_dir: Path,
    particles_cs: Path,
    micrographs_dir: Path,
    find_parameters: Path,
    filter_parameters: Path,
    limit: int = 0,
    progress_every: int = 1,
    progress_stream=sys.stderr,
) -> dict[str, int]:
    sys.path.insert(0, str(vesicle_picker_dir))
    from vesicle_picker import external_export, generate_masks, helpers, postprocess, preprocess

    particles = load_particles_cs(particles_cs)
    records = micrograph_records_from_particles(particles, micrographs_dir)
    if limit > 0:
        records = records[:limit]

    find_config = helpers.read_config(str(find_parameters))
    filter_config = helpers.read_config(str(filter_parameters))
    found_dir = Path(find_config.get("output", "directory"))
    filtered_dir = Path(filter_config.get("output", "directory"))
    found_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    model = generate_masks.initialize_model(
        model_weights_path=find_config.get("segmentation", "model_weights_path"),
        model_type=find_config.get("segmentation", "model_type"),
        device=find_config.get("segmentation", "device"),
    )

    processed = 0
    found_written = 0
    filtered_written = 0
    skipped_empty = 0
    total = len(records)
    for index, record in enumerate(records, start=1):
        if should_emit_progress(index, progress_every):
            emit_progress(progress_stream, index=index, total=total, record=record, status="start")
        image_fullres = read_mode2_mrc(record.path)
        preprocessed_micrograph = preprocess.preprocess_micrograph(
            image_fullres,
            downsample=find_config.getint("general", "downsample"),
            lowpass_mode=find_config.get("preprocessing", "lowpass_mode"),
            d=find_config.getint("preprocessing", "d"),
            sigmaColor=find_config.getint("preprocessing", "sigmaColor"),
            sigmaSpace=find_config.getint("preprocessing", "sigmaSpace"),
        )
        masks = generate_masks.generate_masks(
            preprocessed_micrograph,
            model,
            psize=find_config.getfloat("general", "psize"),
            downsample=find_config.getint("general", "downsample"),
            points_per_side=find_config.getint("segmentation", "points_per_side"),
            points_per_batch=find_config.getint("segmentation", "points_per_batch"),
            pred_iou_thresh=find_config.getfloat("segmentation", "pred_iou_thresh"),
            stability_score_thresh=find_config.getfloat("segmentation", "stability_score_thresh"),
            crop_n_layers=find_config.getint("segmentation", "crop_n_layers"),
            crop_n_points_downscale_factor=find_config.getint("segmentation", "crop_n_points_downscale_factor"),
            crop_nms_thresh=find_config.getfloat("segmentation", "crop_nms_thresh"),
            min_mask_region_area=find_config.getint("segmentation", "min_mask_region_area"),
        )
        processed += 1
        if not masks:
            skipped_empty += 1
            if should_emit_progress(index, progress_every):
                emit_progress(
                    progress_stream,
                    index=index,
                    total=total,
                    record=record,
                    found_masks=0,
                    filtered_masks=0,
                    status="empty",
                )
            continue

        postprocessed_masks = []
        for mask in masks:
            postprocessed = postprocess.find_mask_intensity(mask, preprocessed_micrograph)
            postprocessed = postprocess.find_contour(postprocessed)
            if postprocessed is None:
                continue
            postprocessed = postprocess.find_roundness(postprocessed)
            if postprocessed is None:
                continue
            postprocessed = postprocess.fit_ellipse(postprocessed)
            if postprocessed is not None:
                postprocessed_masks.append(postprocessed)
        for mask in postprocessed_masks:
            if "average_radius" in mask:
                psize = find_config.getfloat("general", "psize")
                downsample = find_config.getint("general", "downsample")
                mask["average_radius_A"] = mask["average_radius"] * psize * downsample
                mask["semi_minor_A"] = mask["semi_minor"] * psize * downsample
                mask["semi_major_A"] = mask["semi_major"] * psize * downsample

        found_path = found_dir / f"{record.uid}_vesicles.pkl"
        external_export.export_masks_to_disk(
            postprocessed_masks,
            str(found_path),
            micrograph_uid=record.uid,
            compression="uint64",
        )
        found_written += 1

        filtered_masks = postprocess.apply_filters(postprocessed_masks, str(filter_parameters))
        filtered_masks = [mask for mask in filtered_masks if mask is not None]
        if not filtered_masks:
            if should_emit_progress(index, progress_every):
                emit_progress(
                    progress_stream,
                    index=index,
                    total=total,
                    record=record,
                    found_masks=len(postprocessed_masks),
                    filtered_masks=0,
                    status="filtered_empty",
                )
            continue
        filtered_path = filtered_dir / f"{record.uid}_vesicles_filtered.pkl"
        external_export.export_masks_to_disk(
            filtered_masks,
            str(filtered_path),
            micrograph_uid=record.uid,
            compression="uint64",
        )
        filtered_written += 1
        if should_emit_progress(index, progress_every):
            emit_progress(
                progress_stream,
                index=index,
                total=total,
                record=record,
                found_masks=len(postprocessed_masks),
                filtered_masks=len(filtered_masks),
                status="done",
            )

    return {
        "micrographs_seen": len(records),
        "micrographs_processed": processed,
        "found_mask_files": found_written,
        "filtered_mask_files": filtered_written,
        "empty_micrographs": skipped_empty,
    }


def should_emit_progress(index: int, progress_every: int) -> bool:
    return progress_every > 0 and index % progress_every == 0


def emit_progress(
    stream,
    *,
    index: int,
    total: int,
    record: MicrographRecord,
    status: str,
    found_masks: int | None = None,
    filtered_masks: int | None = None,
) -> None:
    parts = [
        "[local vesicle picker]",
        f"{index}/{total}",
        f"uid={record.uid}",
        f"status={status}",
    ]
    if found_masks is not None:
        parts.append(f"found={found_masks}")
    if filtered_masks is not None:
        parts.append(f"filtered={filtered_masks}")
    parts.append(f"mic={record.path.name}")
    print(" ".join(parts), file=stream, flush=True)


def main() -> None:
    args = parse_args()
    summary = run_local_vesicle_picker(
        vesicle_picker_dir=args.vesicle_picker_dir,
        particles_cs=args.particles_cs,
        micrographs_dir=args.micrographs_dir,
        find_parameters=args.find_parameters,
        filter_parameters=args.filter_parameters,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
