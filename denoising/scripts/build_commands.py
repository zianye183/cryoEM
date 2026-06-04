#!/usr/bin/env python3
"""Print command templates for the first denoising tests.

The printed commands are intentionally conservative: they do not execute Topaz or DRACO.
Review `topaz denoise --help` and the local DRACO notebook/API once environments are installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", required=True, help="Subset directory containing .mrc files"
    )
    parser.add_argument("--results", required=True, help="Results root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = Path(args.subset).resolve()
    results = Path(args.results).resolve()
    files = sorted(subset.glob("*.mrc"))

    if not files:
        raise SystemExit(f"No .mrc files found in subset: {subset}")

    file_list = " ".join(str(path) for path in files)
    topaz_out = results / "topaz_pretrained"
    draco_out = results / "draco_pretrained"

    print("# Topaz-Denoise pretrained command template")
    print("# Verify flags with: topaz denoise --help")
    print(f"mkdir -p {topaz_out}")
    print(
        "topaz denoise "
        f"--model unet --device 0 --patch-size 4000 --patch-padding 500 "
        f"--suffix _topaz_denoised --output {topaz_out} {file_list}"
    )

    print("\n# Topaz-Denoise retraining note")
    print(
        "# Topaz retraining requires paired noisy observations (typically even/odd frame sums)."
    )
    print(
        "# First confirm whether *_fractions.mrc files are frame stacks and prepare paired inputs."
    )
    print(
        "# Then inspect the installed command help, e.g.: topaz denoise --help and topaz denoise-train --help"
    )

    print("\n# DRACO Denoise/B pretrained command template")
    print("# First set up the DRACO env/repo/checkpoint, then dry-run inference:")
    print("python /nfs/bartesaghilab2/zy223/denoising/scripts/setup_draco.py --dry-run")
    print(f"mkdir -p {draco_out}")
    print(
        "python /nfs/bartesaghilab2/zy223/denoising/scripts/run_draco_pretrained.py "
        f"{subset} "
        "--repo /nfs/bartesaghilab2/zy223/denoising/external/draco "
        "--weights /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt "
        f"--output-dir {draco_out} "
        "--dry-run"
    )


if __name__ == "__main__":
    main()
