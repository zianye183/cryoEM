import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.vesicle_membrane_picks import (
    export_vesicle_membrane_picks,
    generate_pick_indices,
    pick_fractional_coordinates,
)


def make_source_particles() -> np.ndarray:
    dtype = [
        ("uid", "<u8"),
        ("location/micrograph_uid", "<u8"),
        ("location/exp_group_id", "<u4"),
        ("location/micrograph_path", "S128"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
        ("location/micrograph_psize_A", "<f4"),
    ]
    arr = np.zeros(3, dtype=dtype)
    arr["uid"] = [10, 11, 12]
    arr["location/micrograph_uid"] = [7, 7, 8]
    arr["location/exp_group_id"] = [3, 3, 4]
    arr["location/micrograph_path"] = [
        b"J1/imported/mic_a.mrc",
        b"J1/imported/mic_a.mrc",
        b"J1/imported/mic_b.mrc",
    ]
    arr["location/micrograph_shape"] = [(20, 20), (20, 20), (20, 20)]
    arr["location/micrograph_psize_A"] = [1.0, 1.0, 1.0]
    return arr


def make_source_exposures() -> np.ndarray:
    dtype = [
        ("uid", "<u8"),
        ("micrograph_blob/path", "S128"),
        ("micrograph_blob/shape", "<u4", (2,)),
        ("micrograph_blob/psize_A", "<f4"),
        ("ctf/exp_group_id", "<u4"),
    ]
    arr = np.zeros(2, dtype=dtype)
    arr["uid"] = [7, 8]
    arr["micrograph_blob/path"] = [b"J3/motioncorrected/mic_a.mrc", b"J3/motioncorrected/mic_b.mrc"]
    arr["micrograph_blob/shape"] = [(20, 20), (20, 20)]
    arr["micrograph_blob/psize_A"] = [1.0, 1.0]
    arr["ctf/exp_group_id"] = [3, 4]
    return arr


def square_mask() -> dict[str, np.ndarray]:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[2:8, 2:8] = True
    return {"segmentation": segmentation}


def write_prime_mask(path: Path) -> None:
    segmentation = square_mask()["segmentation"]
    composite = np.ones(segmentation.shape, dtype=np.uint16)
    composite[segmentation] = 2
    payload = {
        "masks": [{"area": int(segmentation.sum()), "prime_key": 2}],
        "composite_mask": composite,
        "uid": 7,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


class VesicleMembranePickTests(unittest.TestCase):
    def test_generate_pick_indices_uses_edges_or_surface(self) -> None:
        edge_picks = generate_pick_indices(
            [square_mask()],
            psize_a=1.0,
            downsample=1,
            box_size_a=3.0,
            dilation_radius_a=0.0,
            mode="edge",
        )
        surface_picks = generate_pick_indices(
            [square_mask()],
            psize_a=1.0,
            downsample=1,
            box_size_a=3.0,
            dilation_radius_a=0.0,
            mode="surface",
        )

        self.assertGreater(len(edge_picks), 0)
        self.assertGreaterEqual(len(surface_picks), len(edge_picks))
        self.assertTrue(all(y in {2, 7} or x in {2, 7} for y, x in edge_picks))
        self.assertTrue(any(2 < y < 7 and 2 < x < 7 for y, x in surface_picks))

    def test_larger_box_size_produces_fewer_or_equal_picks(self) -> None:
        dense = generate_pick_indices(
            [square_mask()],
            psize_a=1.0,
            downsample=1,
            box_size_a=2.0,
            dilation_radius_a=0.0,
            mode="edge",
        )
        sparse = generate_pick_indices(
            [square_mask()],
            psize_a=1.0,
            downsample=1,
            box_size_a=5.0,
            dilation_radius_a=0.0,
            mode="edge",
        )

        self.assertLessEqual(len(sparse), len(dense))

    def test_pick_fractional_coordinates_maps_mask_to_fullres_fraction(self) -> None:
        x_frac, y_frac = pick_fractional_coordinates(
            y_mask=5,
            x_mask=2,
            micrograph_shape=(20, 20),
            downsample=2,
        )

        self.assertAlmostEqual(x_frac, 0.2)
        self.assertAlmostEqual(y_frac, 0.5)

    def test_export_vesicle_membrane_picks_writes_importable_result_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_cs = root / "particles.cs"
            source_csg = root / "particles.csg"
            mask_dir = root / "masks"
            output_dir = root / "out"
            mask_dir.mkdir()
            with source_cs.open("wb") as handle:
                np.save(handle, make_source_particles())
            source_csg.write_text(json.dumps({"files": [{"path": str(source_cs)}]}))
            write_prime_mask(mask_dir / "7_vesicles_filtered.pkl")

            outputs = export_vesicle_membrane_picks(
                particles_cs=source_cs,
                particles_csg=source_csg,
                vesicle_mask_dir=mask_dir,
                output_dir=output_dir,
                psize_a=1.0,
                downsample=2,
                box_size_a=4.0,
                dilation_radius_a=0.0,
                mode="edge",
            )

            generated = np.load(outputs.picks_cs, allow_pickle=False)
            patched_csg = json.loads(outputs.picks_csg.read_text())
            summary = json.loads(outputs.summary_json.read_text())

        self.assertGreater(len(generated), 0)
        self.assertEqual(set(generated["location/micrograph_uid"].tolist()), {7})
        self.assertEqual(generated["uid"].tolist(), list(range(len(generated))))
        self.assertEqual(patched_csg["files"][0]["path"], str(outputs.picks_cs))
        self.assertEqual(summary["total_picks"], len(generated))
        self.assertEqual(summary["missing_mask_count"], 1)
        self.assertEqual(summary["micrographs"][0]["micrograph_uid"], 7)
        self.assertEqual(summary["micrographs"][1]["micrograph_uid"], 8)
        self.assertEqual(summary["micrographs"][1]["status"], "missing_mask")

    def test_export_vesicle_membrane_picks_uses_exposure_metadata_without_particles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_cs = root / "exposures.cs"
            source_csg = root / "exposures.csg"
            mask_dir = root / "masks"
            output_dir = root / "out"
            mask_dir.mkdir()
            with source_cs.open("wb") as handle:
                np.save(handle, make_source_exposures())
            source_csg.write_text(json.dumps({"files": [{"path": str(source_cs)}]}))
            write_prime_mask(mask_dir / "7_vesicles_filtered.pkl")

            outputs = export_vesicle_membrane_picks(
                metadata_cs=source_cs,
                metadata_csg=source_csg,
                vesicle_mask_dir=mask_dir,
                output_dir=output_dir,
                psize_a=1.0,
                downsample=2,
                box_size_a=4.0,
                dilation_radius_a=0.0,
                mode="edge",
            )

            generated = np.load(outputs.picks_cs, allow_pickle=False)
            patched_csg = json.loads(outputs.picks_csg.read_text())
            summary = json.loads(outputs.summary_json.read_text())

        self.assertGreater(len(generated), 0)
        self.assertEqual(set(generated["location/micrograph_uid"].tolist()), {7})
        self.assertEqual(set(generated["location/micrograph_path"].tolist()), {b"J3/motioncorrected/mic_a.mrc"})
        self.assertEqual(set(generated["location/exp_group_id"].tolist()), {3})
        self.assertEqual(set(generated["location/micrograph_psize_A"].tolist()), {1.0})
        self.assertEqual(patched_csg["files"][0]["path"], str(outputs.picks_cs))
        self.assertEqual(summary["inputs"]["metadata_cs"], str(source_cs))
        self.assertEqual(summary["inputs"]["metadata_kind"], "exposure")
        self.assertEqual(summary["missing_mask_count"], 1)


if __name__ == "__main__":
    unittest.main()
