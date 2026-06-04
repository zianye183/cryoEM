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

from segmentation_integration.vesicle_picker_invocation import prepare_vesicle_picker_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vesicle-picker-dir", type=Path, required=True)
    parser.add_argument("--csparc-login", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-type", choices=["curate", "patch_ctf"], default="curate")
    parser.add_argument("--model-weights-path", type=Path, required=True)
    parser.add_argument("--particles-cs", type=Path, required=True)
    parser.add_argument("--particles-csg", type=Path, required=True)
    parser.add_argument("--micrographs-dir", type=Path, required=True)
    parser.add_argument("--filter-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
