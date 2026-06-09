#!/usr/bin/env python3
"""Generate offline CryoSPARC result groups from Vesicle Picker membrane masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation_integration.dataset_presets import dataset_choices, resolve_standard_paths
from segmentation_integration.vesicle_membrane_picks import export_vesicle_membrane_picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=dataset_choices(), help="Use standard Prestin paths for raw or draco.")
    parser.add_argument("--particle-export-dir", type=Path, help="Override the preset particle export directory.")
    parser.add_argument("--metadata-cs", type=Path, help="Exported particle or exposure .cs metadata/template source.")
    parser.add_argument("--metadata-csg", type=Path, help="Exported particle or exposure .csg wrapper template.")
    parser.add_argument("--exposures-cs", type=Path, help="Alias for --metadata-cs when using an exposures export.")
    parser.add_argument("--exposures-csg", type=Path, help="Alias for --metadata-csg when using an exposures export.")
    parser.add_argument("--particles-cs", type=Path, help="Deprecated alias for --metadata-cs.")
    parser.add_argument("--particles-csg", type=Path, help="Deprecated alias for --metadata-csg.")
    parser.add_argument("--vesicle-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psize-a", type=float, default=1.068)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--box-size-a", type=float, default=100.0)
    parser.add_argument("--dilation-radius-a", type=float, default=0.0)
    parser.add_argument("--mode", choices=("edge", "surface"), default="edge")
    args = parser.parse_args()
    explicit_cs = [value for value in (args.metadata_cs, args.exposures_cs, args.particles_cs) if value is not None]
    explicit_csg = [value for value in (args.metadata_csg, args.exposures_csg, args.particles_csg) if value is not None]
    if len(explicit_cs) > 1 or len(explicit_csg) > 1:
        parser.error("provide only one metadata .cs/.csg pair")
    if args.dataset is None and (not explicit_cs or not explicit_csg):
        parser.error("--dataset is required unless a metadata .cs/.csg pair is provided")
    args.metadata_cs = explicit_cs[0] if explicit_cs else None
    args.metadata_csg = explicit_csg[0] if explicit_csg else None
    return args


def main() -> None:
    args = parse_args()
    if args.dataset is not None:
        paths = resolve_standard_paths(
            args.dataset,
            particle_export_dir=args.particle_export_dir,
            particles_cs=args.metadata_cs,
            particles_csg=args.metadata_csg,
        )
        args.metadata_cs = paths.particles_cs
        args.metadata_csg = paths.particles_csg

    outputs = export_vesicle_membrane_picks(
        metadata_cs=args.metadata_cs,
        metadata_csg=args.metadata_csg,
        vesicle_mask_dir=args.vesicle_mask_dir,
        output_dir=args.output_dir,
        psize_a=args.psize_a,
        downsample=args.downsample,
        box_size_a=args.box_size_a,
        dilation_radius_a=args.dilation_radius_a,
        mode=args.mode,
    )
    print(f"Wrote vesicle membrane picks: {outputs.picks_cs}")
    print(f"Import result group in CryoSPARC with: {outputs.picks_csg}")
    print(f"Summary: {outputs.summary_json}")


if __name__ == "__main__":
    main()
