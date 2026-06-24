#!/usr/bin/env python3
"""Prepare a validated even/odd pair manifest for Topaz fine-tuning.

The script scans an external read-only dataset directory for files ending in
`_half1.mrc` and pairs them with matching `_half2.mrc` files. It does not copy
or modify MRC data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from finetuning_common import (
        DEFAULT_ACCEPTED_EXPOSURES_CS,
        DEFAULT_MANIFEST_DIR,
        DEFAULT_SAMPLE_DATASET,
        PAIR_COLUMNS,
        get_mrc_metadata,
        manifest_summary,
        write_json,
        write_tsv,
    )
except ImportError:
    from .finetuning_common import (
        DEFAULT_ACCEPTED_EXPOSURES_CS,
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
        "--accepted-cs",
        default=str(DEFAULT_ACCEPTED_EXPOSURES_CS),
        help=(
            "Optional CryoSPARC .cs file containing accepted exposures. Use an empty "
            "string to include all discovered half pairs."
        ),
    )
    parser.add_argument(
        "--accepted-path-column",
        default="micrograph_blob/path",
        help="CryoSPARC .cs column whose basename should match half-split stems.",
    )
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


def decode_cs_value(value: object) -> str:
    """Decode a CryoSPARC scalar value into stable text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def normalize_micrograph_id(name_or_path: str) -> str:
    """Normalize CryoSPARC and half-split names to `FoilHole_..._fractions`."""
    name = Path(name_or_path.lstrip(">")).name
    match = re.search(r"(FoilHole_.+?_fractions)", name)
    if match:
        return match.group(1)

    for suffix in [
        "_half1.mrc",
        "_half2.mrc",
        "_patch_aligned_doseweighted.mrc",
        "_patch_aligned.mrc",
        ".mrc",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_accepted_exposures(
    cs_path: Path, *, path_column: str
) -> dict[str, dict[str, str]]:
    """Return accepted exposure metadata keyed by normalized micrograph ID."""
    if not cs_path.is_file():
        raise FileNotFoundError(f"Accepted exposures .cs does not exist: {cs_path}")

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading CryoSPARC .cs files requires numpy. Run on the HPC Python "
            "environment or activate an environment with numpy installed."
        ) from exc

    exposures = np.load(cs_path, allow_pickle=True)
    if exposures.dtype.names is None:
        raise ValueError(f"CryoSPARC .cs file has no named columns: {cs_path}")
    if path_column not in exposures.dtype.names:
        raise ValueError(
            f"Column {path_column!r} not found in {cs_path}. Available columns include: "
            f"{', '.join(exposures.dtype.names[:20])}"
        )

    accepted: dict[str, dict[str, str]] = {}
    for row in exposures:
        source_path = decode_cs_value(row[path_column])
        micrograph_id = normalize_micrograph_id(source_path)
        metadata = {
            "source_micrograph_path": source_path,
            "cryosparc_uid": str(int(row["uid"])) if "uid" in exposures.dtype.names else "",
            "pixel_size_a": (
                f"{float(row['micrograph_blob/psize_A']):.6g}"
                if "micrograph_blob/psize_A" in exposures.dtype.names
                else ""
            ),
            "shape_y": (
                str(int(row["micrograph_blob/shape"][0]))
                if "micrograph_blob/shape" in exposures.dtype.names
                else ""
            ),
            "shape_x": (
                str(int(row["micrograph_blob/shape"][1]))
                if "micrograph_blob/shape" in exposures.dtype.names
                else ""
            ),
            "ctf_fit_to_a": (
                f"{float(row['ctf_stats/ctf_fit_to_A']):.6g}"
                if "ctf_stats/ctf_fit_to_A" in exposures.dtype.names
                else ""
            ),
            "df1_a": (
                f"{float(row['ctf/df1_A']):.6g}"
                if "ctf/df1_A" in exposures.dtype.names
                else ""
            ),
            "df2_a": (
                f"{float(row['ctf/df2_A']):.6g}"
                if "ctf/df2_A" in exposures.dtype.names
                else ""
            ),
            "ice_thickness_rel": (
                f"{float(row['ctf_stats/ice_thickness_rel']):.6g}"
                if "ctf_stats/ice_thickness_rel" in exposures.dtype.names
                else ""
            ),
        }
        if micrograph_id in accepted:
            raise ValueError(f"Duplicate accepted micrograph ID in .cs: {micrograph_id}")
        accepted[micrograph_id] = metadata
    if not accepted:
        raise ValueError(f"No accepted exposures found in {cs_path}")
    return accepted


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
    accepted = None
    if args.accepted_cs:
        accepted = load_accepted_exposures(
            Path(args.accepted_cs).expanduser().resolve(),
            path_column=args.accepted_path_column,
        )

    even_paths = sorted(source_dir.glob(f"*{args.even_suffix}"))
    if args.limit is not None:
        even_paths = even_paths[: args.limit]

    for even_path in even_paths:
        micrograph_id = normalize_micrograph_id(strip_suffix(even_path.name, args.even_suffix))
        if accepted is not None and micrograph_id not in accepted:
            continue
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

        accepted_metadata = accepted[micrograph_id] if accepted is not None else {}
        rows.append(
            {
                "micrograph_id": micrograph_id,
                "even_path": str(even_path),
                "odd_path": str(odd_path),
                "complete_path": complete_text,
                "mask_path": mask_text,
                **accepted_metadata,
                "session_id": "",
                "defocus_group": "",
                "pixel_size_a": pixel_size or accepted_metadata.get("pixel_size_a", ""),
                "shape_y": shape_y or accepted_metadata.get("shape_y", ""),
                "shape_x": shape_x or accepted_metadata.get("shape_x", ""),
                "notes": "; ".join(notes),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No matched files ending with {args.even_suffix!r} found in {source_dir}"
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
    if args.accepted_cs:
        accepted = load_accepted_exposures(
            Path(args.accepted_cs).expanduser().resolve(),
            path_column=args.accepted_path_column,
        )
        matched_ids = {row["micrograph_id"] for row in rows}
        summary["accepted_exposures"] = len(accepted)
        summary["accepted_with_half_pairs"] = len(matched_ids)
        summary["accepted_missing_half_pairs"] = sorted(set(accepted) - matched_ids)
    summary.update(
        {
            "source_dir": str(Path(args.source_dir).expanduser().resolve()),
            "accepted_cs": str(Path(args.accepted_cs).expanduser().resolve())
            if args.accepted_cs
            else "",
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
