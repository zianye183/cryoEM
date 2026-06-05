import pickle
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import numpy as np

from segmentation_integration.filtering import filter_particles_by_masks, particle_mask_coordinate
from segmentation_integration.vesicle_masks import build_membrane_band, load_vesicle_masks


def make_particles() -> np.ndarray:
    dtype = [
        ("uid", "<u8"),
        ("location/micrograph_uid", "<u8"),
        ("location/micrograph_path", "S128"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
    ]
    arr = np.zeros(4, dtype=dtype)
    arr["uid"] = [1, 2, 3, 4]
    arr["location/micrograph_uid"] = [7, 7, 7, 8]
    arr["location/micrograph_path"] = [
        b"J1/imported/mic_a.mrc",
        b"J1/imported/mic_a.mrc",
        b"J1/imported/mic_a.mrc",
        b"J1/imported/mic_b.mrc",
    ]
    arr["location/micrograph_shape"] = [(100, 100), (100, 100), (100, 100), (100, 100)]
    arr["location/center_x_frac"] = [0.20, 0.50, 1.10, 0.20]
    arr["location/center_y_frac"] = [0.50, 0.50, 0.50, 0.50]
    return arr


def write_prime_mask(path: Path) -> None:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[2:8, 2:8] = True
    composite = np.ones((10, 10), dtype=np.uint16)
    composite[segmentation] = 2
    payload = {
        "masks": [{"area": int(segmentation.sum()), "prime_key": 2}],
        "composite_mask": composite,
        "uid": 7,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


class VesicleFilteringTests(unittest.TestCase):
    def test_load_vesicle_masks_reconstructs_prime_encoded_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "7_vesicles_filtered.pkl"
            write_prime_mask(path)

            masks = load_vesicle_masks(path)

        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0]["segmentation"][3, 3])
        self.assertFalse(masks[0]["segmentation"][0, 0])

    def test_build_membrane_band_uses_contour_not_filled_surface(self) -> None:
        segmentation = np.zeros((10, 10), dtype=bool)
        segmentation[2:8, 2:8] = True

        band = build_membrane_band([{"segmentation": segmentation}], radius_px=1)

        self.assertTrue(band[2, 5])
        self.assertTrue(band[5, 2])
        self.assertFalse(band[5, 5])
        self.assertFalse(band[0, 0])

    def test_particle_mask_coordinate_maps_fractional_fullres_to_downsampled_mask(self) -> None:
        particle = make_particles()[0]

        y_mask, x_mask = particle_mask_coordinate(particle, downsample=2)

        self.assertEqual((y_mask, x_mask), (25, 10))

    def test_filter_particles_by_masks_keeps_on_band_and_rejects_missing_or_out_of_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp)
            write_prime_mask(mask_dir / "7_vesicles_filtered.pkl")

            result = filter_particles_by_masks(
                make_particles(),
                vesicle_mask_dir=mask_dir,
                psize_a=1.0,
                downsample=10,
                membrane_band_a=10.0,
            )

        self.assertEqual([int(row["uid"]) for row in result.kept], [1])
        self.assertEqual([int(row["uid"]) for row in result.rejected], [2, 3, 4])
        self.assertEqual(result.reason_counts["outside_membrane_band"], 1)
        self.assertEqual(result.reason_counts["out_of_bounds"], 1)
        self.assertEqual(result.reason_counts["missing_mask"], 1)

    def test_filter_particles_by_masks_emits_per_micrograph_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp)
            write_prime_mask(mask_dir / "7_vesicles_filtered.pkl")
            stream = StringIO()

            filter_particles_by_masks(
                make_particles(),
                vesicle_mask_dir=mask_dir,
                psize_a=1.0,
                downsample=10,
                membrane_band_a=10.0,
                progress_stream=stream,
            )

        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(lines[0], "[particle filter] 1/2 uid=7 status=start mic=mic_a.mrc")
        self.assertRegex(
            lines[1],
            r"^\[particle filter\] 1/2 uid=7 status=done kept=1 rejected=2 elapsed=\d+\.\d{2}s mic=mic_a.mrc$",
        )
        self.assertEqual(lines[2], "[particle filter] 2/2 uid=8 status=start mic=mic_b.mrc")
        self.assertRegex(
            lines[3],
            r"^\[particle filter\] 2/2 uid=8 status=missing_mask kept=0 rejected=1 elapsed=\d+\.\d{2}s mic=mic_b.mrc$",
        )


if __name__ == "__main__":
    unittest.main()
