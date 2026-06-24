from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from finetuning.scripts.prepare_pairs import (
    build_rows,
    load_accepted_exposures,
    normalize_micrograph_id,
)


def test_normalize_micrograph_id_handles_cryosparc_and_half_names() -> None:
    assert (
        normalize_micrograph_id(
            ">J3/motioncorrected/011_FoilHole_6367907_Data_5347108_0_20260207_032126_fractions_patch_aligned_doseweighted.mrc"
        )
        == "FoilHole_6367907_Data_5347108_0_20260207_032126_fractions"
    )
    assert (
        normalize_micrograph_id(
            "FoilHole_6367907_Data_5347108_0_20260207_032126_fractions_half1.mrc"
        )
        == "FoilHole_6367907_Data_5347108_0_20260207_032126_fractions"
    )


def test_load_accepted_exposures_reads_relevant_metadata(tmp_path: Path) -> None:
    cs_path = tmp_path / "accepted.cs"
    dtype = [
        ("uid", "<u8"),
        ("micrograph_blob/path", "S160"),
        ("ctf_stats/ctf_fit_to_A", "<f4"),
        ("ctf/df1_A", "<f4"),
        ("ctf/df2_A", "<f4"),
        ("ctf_stats/ice_thickness_rel", "<f4"),
    ]
    data = np.array(
        [
            (
                123,
                b">J3/motioncorrected/000123_FoilHole_1_Data_2_3_20260207_000000_fractions_patch_aligned_doseweighted.mrc",
                4.5,
                10000.0,
                11000.0,
                1.25,
            )
        ],
        dtype=dtype,
    )
    np.save(cs_path, data)
    saved_path = cs_path.with_suffix(".cs.npy")
    saved_path.rename(cs_path)

    accepted = load_accepted_exposures(cs_path, path_column="micrograph_blob/path")

    assert list(accepted) == ["FoilHole_1_Data_2_3_20260207_000000_fractions"]
    row = accepted["FoilHole_1_Data_2_3_20260207_000000_fractions"]
    assert row["cryosparc_uid"] == "123"
    assert row["ctf_fit_to_a"] == "4.5"
    assert row["df1_a"] == "10000"
    assert row["df2_a"] == "11000"
    assert row["ice_thickness_rel"] == "1.25"


def test_build_rows_filters_to_accepted_pairs(tmp_path: Path) -> None:
    source_dir = tmp_path / "mrc"
    source_dir.mkdir()
    accepted_id = "FoilHole_1_Data_2_3_20260207_000000_fractions"
    rejected_id = "FoilHole_2_Data_2_3_20260207_000001_fractions"
    for micrograph_id in [accepted_id, rejected_id]:
        (source_dir / f"{micrograph_id}_half1.mrc").write_bytes(b"fake")
        (source_dir / f"{micrograph_id}_half2.mrc").write_bytes(b"fake")
        (source_dir / f"{micrograph_id}.mrc").write_bytes(b"fake")

    cs_path = tmp_path / "accepted.cs"
    data = np.array(
        [
            (
                123,
                f">J3/motioncorrected/000123_{accepted_id}_patch_aligned_doseweighted.mrc".encode(),
            )
        ],
        dtype=[("uid", "<u8"), ("micrograph_blob/path", "S180")],
    )
    np.save(cs_path, data)
    cs_path.with_suffix(".cs.npy").rename(cs_path)

    args = argparse.Namespace(
        source_dir=str(source_dir),
        output=str(tmp_path / "all_pairs.tsv"),
        summary_json=None,
        even_suffix="_half1.mrc",
        odd_suffix="_half2.mrc",
        accepted_cs=str(cs_path),
        accepted_path_column="micrograph_blob/path",
        complete_suffix=".mrc",
        mask_dir=None,
        mask_suffix="_mask.mrc",
        limit=None,
        skip_metadata=True,
        allow_metadata_mismatch=False,
    )

    rows = build_rows(args)

    assert len(rows) == 1
    assert rows[0]["micrograph_id"] == accepted_id
    assert rows[0]["cryosparc_uid"] == "123"
    assert rows[0]["even_path"].endswith(f"{accepted_id}_half1.mrc")
    assert rows[0]["odd_path"].endswith(f"{accepted_id}_half2.mrc")
