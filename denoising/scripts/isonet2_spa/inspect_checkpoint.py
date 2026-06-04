#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Inspect IsoNet2 checkpoint metadata needed for SPA smoke-test prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_MODEL, DEFAULT_WORK_DIR, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="IsoNet2 .pt checkpoint path."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_WORK_DIR / "checkpoint_metadata.json"),
        help="JSON file to write checkpoint metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"IsoNet2 checkpoint does not exist: {model_path}")

    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = {
        "model_path": str(model_path),
        "method": checkpoint.get("method"),
        "arch": checkpoint.get("arch"),
        "cube_size": checkpoint.get("cube_size"),
        "CTF_mode": checkpoint.get("CTF_mode"),
        "do_phaseflip_input": checkpoint.get("do_phaseflip_input"),
        "metrics_keys": sorted(list(checkpoint.get("metrics", {}).keys())),
        "checkpoint_keys": sorted(list(checkpoint.keys())),
    }
    write_json(Path(args.out).expanduser().resolve(), metadata)
    print(metadata)


if __name__ == "__main__":
    main()
