#!/usr/bin/env python3
"""Prepare Vesicle Picker configs and an end-to-end mask-filter run script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation_integration.dataset_presets import dataset_choices
from segmentation_integration.vesicle_picker_invocation import (
    prepare_vesicle_picker_run,
    prepare_vesicle_picker_run_for_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), help="Use standard Prestin paths for raw or draco.")
    parser.add_argument(
        "--local-micrographs",
        action="store_true",
        help="Generate Vesicle Picker masks from preset/local MRC files instead of a CryoSPARC job.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vesicle-picker-dir", type=Path, help="Override Vesicle Picker checkout root.")
    parser.add_argument("--csparc-login", type=Path)
    parser.add_argument("--project-id")
    parser.add_argument("--workspace-id")
    parser.add_argument("--job-id")
    parser.add_argument("--job-type", choices=["curate", "patch_ctf"], default="curate")
    parser.add_argument("--model-weights-path", type=Path, required=True)
    parser.add_argument("--particle-export-dir", type=Path, help="Override the preset particle export directory.")
    parser.add_argument("--particles-cs", type=Path, help="Override the preset particle .cs file.")
    parser.add_argument("--particles-csg", type=Path, help="Override the preset particle .csg file.")
    parser.add_argument("--micrographs-dir", type=Path, help="Override the preset micrograph directory.")
    parser.add_argument("--filter-output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset is None:
        missing = [
            name
            for name in ("vesicle_picker_dir", "particles_cs", "particles_csg", "micrographs_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("--dataset is required unless --vesicle-picker-dir, --particles-cs, --particles-csg, and --micrographs-dir are all provided")
    if not args.local_micrographs:
        missing_cryosparc = [
            name
            for name in ("csparc_login", "project_id", "workspace_id", "job_id")
            if getattr(args, name) is None
        ]
        if missing_cryosparc:
            parser.error("--local-micrographs is required unless --csparc-login, --project-id, --workspace-id, and --job-id are all provided")
    return args


def main() -> None:
    args = parse_args()
    if args.dataset is not None:
        summary = prepare_vesicle_picker_run_for_dataset(
            dataset=args.dataset,
            output_dir=args.output_dir,
            particle_export_dir=args.particle_export_dir,
            csparc_login=args.csparc_login,
            project_id=args.project_id,
            workspace_id=args.workspace_id,
            job_id=args.job_id,
            job_type=args.job_type,
            model_weights_path=args.model_weights_path,
            particles_cs=args.particles_cs,
            particles_csg=args.particles_csg,
            micrographs_dir=args.micrographs_dir,
            vesicle_picker_dir=args.vesicle_picker_dir,
            filter_output_dir=args.filter_output_dir,
            local_micrographs=args.local_micrographs,
        )
    else:
        summary = prepare_vesicle_picker_run(
            output_dir=args.output_dir,
            vesicle_picker_dir=args.vesicle_picker_dir,
            csparc_login=args.csparc_login,
            project_id=args.project_id,
            workspace_id=args.workspace_id,
            job_id=args.job_id,
            job_type=args.job_type,
            model_weights_path=args.model_weights_path,
            particles_cs=args.particles_cs,
            particles_csg=args.particles_csg,
            micrographs_dir=args.micrographs_dir,
            filter_output_dir=args.filter_output_dir,
            local_micrographs=args.local_micrographs,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
