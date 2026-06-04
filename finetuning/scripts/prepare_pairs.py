#!/usr/bin/env python3
"""Prepare a validated even/odd pair manifest for Topaz fine-tuning.

The script scans an external read-only dataset directory for files ending in
`_half1.mrc` and pairs them with matching `_half2.mrc` files. It does not copy
or modify MRC data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_SAMPLE_DATASET,
    PAIR_COLUMNS,
    get_mrc_metadata,
    manifest_summary,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SAMPLE_DATASET),
        help="Directory containing complete micrographs plus half-split MRC files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MANIFEST_DIR / "all_pairs.tsv"),
        help="Output TSV manifest path.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--even-suffix", default="_half1.mrc")
    parser.add_argument("--odd-suffix", default="_half2.mrc")
    parser.add_argument(
        "--complete-suffix",
        default=".mrc",
        help=(
            "Suffix for complete micrograph inferred from pair stem. With default half suffixes, "
            "`foo_half1.mrc` maps to `foo.mrc`."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Optional mask directory. Mask matching is by micrograph_id + --mask-suffix.",
    )
    parser.add_argument("--mask-suffix", default="_mask.mrc")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pairs to write after sorting.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip MRC header reads even if mrcfile is installed.",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Warn instead of failing when readable even/odd metadata disagree.",
    )
    return parser.parse_args()


def strip_suffix(name: str, suffix: str) -> str:
    if not name.endswith(suffix):
        raise ValueError(
            f"Filename does not end with expected suffix {suffix!r}: {name}"
        )
    return name[: -len(suffix)]


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None
    if mask_dir is not None and not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    rows: list[dict[str, str]] = []
    even_paths = sorted(source_dir.glob(f"*{args.even_suffix}"))
    if args.limit is not None:
        even_paths = even_paths[: args.limit]

    for even_path in even_paths:
        micrograph_id = strip_suffix(even_path.name, args.even_suffix)
        odd_path = source_dir / f"{micrograph_id}{args.odd_suffix}"
        if not odd_path.is_file():
            raise FileNotFoundError(
                f"Missing odd split for {even_path.name}: {odd_path}"
            )

        complete_path = source_dir / f"{micrograph_id}{args.complete_suffix}"
        complete_text = str(complete_path) if complete_path.is_file() else ""

        mask_text = ""
        if mask_dir is not None:
            mask_path = mask_dir / f"{micrograph_id}{args.mask_suffix}"
            if mask_path.is_file():
                mask_text = str(mask_path)

        shape_y = ""
        shape_x = ""
        pixel_size = ""
        notes: list[str] = []
        if not args.skip_metadata:
            even_y, even_x, even_pixel = get_mrc_metadata(even_path)
            odd_y, odd_x, odd_pixel = get_mrc_metadata(odd_path)
            shape_y, shape_x, pixel_size = even_y, even_x, even_pixel
            comparable = all([even_y, even_x, odd_y, odd_x])
            if comparable and (even_y != odd_y or even_x != odd_x):
                message = (
                    f"Metadata mismatch for {micrograph_id}: even={even_y}x{even_x}, "
                    f"odd={odd_y}x{odd_x}"
                )
                if args.allow_metadata_mismatch:
                    notes.append(message)
                else:
                    raise ValueError(message)
            if even_pixel and odd_pixel and even_pixel != odd_pixel:
                message = (
                    f"Pixel-size mismatch for {micrograph_id}: "
                    f"even={even_pixel}, odd={odd_pixel}"
                )
                if args.allow_metadata_mismatch:
                    notes.append(message)
                else:
                    raise ValueError(message)

        rows.append(
            {
                "micrograph_id": micrograph_id,
                "even_path": str(even_path),
                "odd_path": str(odd_path),
                "complete_path": complete_text,
                "mask_path": mask_text,
                "session_id": "",
                "defocus_group": "",
                "pixel_size_a": pixel_size,
                "shape_y": shape_y,
                "shape_x": shape_x,
                "notes": "; ".join(notes),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No files ending with {args.even_suffix!r} found in {source_dir}"
        )
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output.with_suffix(output.suffix + ".summary.json")
    )

    rows = build_rows(args)
    write_tsv(output, rows, PAIR_COLUMNS)
    summary = manifest_summary(rows)
    summary.update(
        {
            "source_dir": str(Path(args.source_dir).expanduser().resolve()),
            "even_suffix": args.even_suffix,
            "odd_suffix": args.odd_suffix,
            "output": str(output),
        }
    )
    write_json(summary_json, summary)

    print(f"Wrote {len(rows)} pair(s): {output}")
    print(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
