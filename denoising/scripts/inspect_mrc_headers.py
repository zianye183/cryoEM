#!/usr/bin/env python3
"""Inspect basic MRC header dimensions without external dependencies."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="MRC files or directories containing .mrc files"
    )
    return parser.parse_args()


def iter_mrcs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            out.extend(sorted(path.glob("*.mrc")))
        else:
            out.append(path)
    return out


def read_header(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) < 16:
        raise ValueError("file too small for MRC header")
    nx, ny, nz, mode = struct.unpack("<4i", header)
    return nx, ny, nz, mode


def main() -> None:
    args = parse_args()
    files = iter_mrcs(args.paths)
    if not files:
        raise SystemExit("No MRC files found")

    print("name\tnx\tny\tnz\tmode\tinterpretation")
    for path in files:
        try:
            nx, ny, nz, mode = read_header(path)
            interpretation = "2D micrograph" if nz == 1 else "3D stack/volume"
            print(f"{path.name}\t{nx}\t{ny}\t{nz}\t{mode}\t{interpretation}")
        except Exception as exc:  # noqa: BLE001 - report all parse failures for inspection utility
            print(f"{path.name}\tERROR\t{exc}")


if __name__ == "__main__":
    main()
