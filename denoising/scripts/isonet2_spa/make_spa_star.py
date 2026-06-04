#!/usr/bin/env python3
# pyright: reportImplicitRelativeImport=false
"""Create an IsoNet2-compatible STAR file for SPA pseudo-volumes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import DEFAULT_WORK_DIR, infer_pixel_size_from_mrc, write_star

STAR_COLUMNS = [
    "rlnIndex",
    "rlnTomoName",
    "rlnTomoReconstructedTomogramHalf1",
    "rlnTomoReconstructedTomogramHalf2",
    "rlnPixelSize",
    "rlnDefocus",
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnDeconvTomoName",
    "rlnMaskBoundary",
    "rlnMaskName",
    "rlnTiltMin",
    "rlnTiltMax",
    "rlnBoxFile",
    "rlnNumberSubtomo",
    "rlnCorrectedTomoName",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pseudo_volumes",
        nargs="*",
        default=[str(DEFAULT_WORK_DIR / "pseudo_volumes")],
        help="Pseudo-volume .mrc files or directories. Defaults to experiment pseudo_volumes.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_WORK_DIR / "stars" / "spa_pseudo.star"),
        help="Output STAR path.",
    )
    parser.add_argument(
        "--pixel-size",
        default="auto",
        help="Pixel size in Å, or 'auto' to read MRC header and fall back to --fallback-pixel-size.",
    )
    parser.add_argument(
        "--fallback-pixel-size",
        type=float,
        default=1.0,
        help="Pixel size if header is missing/zero.",
    )
    parser.add_argument(
        "--defocus-a", type=float, default=10000.0, help="Defocus placeholder in Å."
    )
    parser.add_argument(
        "--voltage-kv", type=float, default=300.0, help="Acceleration voltage in kV."
    )
    parser.add_argument(
        "--cs-mm", type=float, default=2.7, help="Spherical aberration in mm."
    )
    parser.add_argument(
        "--amplitude-contrast", type=float, default=0.1, help="Amplitude contrast."
    )
    parser.add_argument(
        "--tilt-min",
        type=float,
        default=-90.0,
        help="Placeholder tilt min; ignored when apply_mw_x1=False.",
    )
    parser.add_argument(
        "--tilt-max",
        type=float,
        default=90.0,
        help="Placeholder tilt max; ignored when apply_mw_x1=False.",
    )
    return parser.parse_args()


def collect_pseudo_volumes(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            manifest = path / "manifest.csv"
            if manifest.is_file():
                with manifest.open("r", encoding="utf-8", newline="") as handle:
                    out.extend(
                        Path(row["pseudo_volume_path"])
                        for row in csv.DictReader(handle)
                    )
            else:
                out.extend(sorted(path.glob("*.mrc")))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(f"Pseudo-volume path does not exist: {path}")
    out = sorted(dict.fromkeys(p.resolve() for p in out))
    if not out:
        raise FileNotFoundError("No pseudo-volume .mrc files found")
    return out


def load_manifest_map(pseudo_dir: Path) -> dict[str, str]:
    manifest = pseudo_dir / "manifest.csv"
    if not manifest.is_file():
        return {}
    mapping: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["pseudo_volume_path"]] = row["input_path"]
    return mapping


def main() -> None:
    args = parse_args()
    pseudo_paths = collect_pseudo_volumes(args.pseudo_volumes)
    manifest_map = load_manifest_map(pseudo_paths[0].parent)

    rows: list[dict[str, object]] = []
    for index, pseudo_path in enumerate(pseudo_paths, start=1):
        if args.pixel_size == "auto":
            pixel_size = (
                infer_pixel_size_from_mrc(pseudo_path) or args.fallback_pixel_size
            )
        else:
            pixel_size = float(args.pixel_size)
        pseudo = str(pseudo_path)
        rows.append(
            {
                "rlnIndex": index,
                "rlnTomoName": pseudo,
                "rlnTomoReconstructedTomogramHalf1": pseudo,
                "rlnTomoReconstructedTomogramHalf2": pseudo,
                "rlnPixelSize": pixel_size,
                "rlnDefocus": args.defocus_a,
                "rlnVoltage": args.voltage_kv,
                "rlnSphericalAberration": args.cs_mm,
                "rlnAmplitudeContrast": args.amplitude_contrast,
                "rlnDeconvTomoName": "None",
                "rlnMaskBoundary": "None",
                "rlnMaskName": "None",
                "rlnTiltMin": args.tilt_min,
                "rlnTiltMax": args.tilt_max,
                "rlnBoxFile": "None",
                "rlnNumberSubtomo": 1,
                "rlnCorrectedTomoName": "None",
                "input_micrograph": manifest_map.get(pseudo, "None"),
            }
        )

    out = Path(args.out).expanduser().resolve()
    write_star(out, rows, STAR_COLUMNS)
    print(f"Wrote {len(rows)} row(s) to {out}")


if __name__ == "__main__":
    main()
