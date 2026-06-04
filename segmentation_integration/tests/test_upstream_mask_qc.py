import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.mrc_io import write_mode2_mrc_for_test
from segmentation_integration.scripts.visualize_vesicle_mask_filtering import render_mask_filter_qc


def write_particles_cs(path: Path) -> None:
    dtype = [
        ("location/micrograph_uid", "<u8"),
        ("location/micrograph_path", "S128"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
    ]
    arr = np.zeros(1, dtype=dtype)
    arr[0]["location/micrograph_uid"] = 7
    arr[0]["location/micrograph_path"] = b"mic_a.mrc"
    arr[0]["location/micrograph_shape"] = (100, 100)
    with path.open("wb") as handle:
        np.save(handle, arr)


def write_mask(path: Path, uid: int, area: int, roundness: float) -> None:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[2:8, 2:8] = True
    composite = np.ones((10, 10), dtype=np.uint16)
    composite[segmentation] = 2
    with path.open("wb") as handle:
        pickle.dump(
            {
                "masks": [{"prime_key": 2, "area": area, "roundness": roundness}],
                "composite_mask": composite,
                "uid": uid,
            },
            handle,
        )


class UpstreamMaskQcTests(unittest.TestCase):
    def test_render_mask_filter_qc_writes_overlay_comparison_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            found_dir = root / "found"
            filtered_dir = root / "filtered"
            micrographs_dir = root / "micrographs"
            output_dir = root / "qc"
            found_dir.mkdir()
            filtered_dir.mkdir()
            micrographs_dir.mkdir()
            particles_cs = root / "particles.cs"
            write_particles_cs(particles_cs)
            write_mode2_mrc_for_test(micrographs_dir / "mic_a.mrc", np.arange(10000, dtype=np.float32).reshape(100, 100))
            write_mask(found_dir / "7_vesicles.pkl", uid=7, area=36, roundness=0.5)
            write_mask(filtered_dir / "7_vesicles_filtered.pkl", uid=7, area=36, roundness=0.5)

            summary = render_mask_filter_qc(
                found_mask_dir=found_dir,
                filtered_mask_dir=filtered_dir,
                output_dir=output_dir,
                max_micrographs=0,
                particles_cs=particles_cs,
                micrographs_dir=micrographs_dir,
            )

            stats = (output_dir / "mask_filter_stats.tsv").read_text()
            comparison_size = (output_dir / "7_found_vs_filtered_overlay.png").stat().st_size

        self.assertEqual(summary["rendered_comparisons"], 1)
        self.assertGreater(comparison_size, 0)
        self.assertIn("uid\tstage\tmask_index\tarea\troundness", stats)
        self.assertIn("7\tfound\t0\t36\t0.5", stats)
        self.assertIn("7\tfiltered\t0\t36\t0.5", stats)


if __name__ == "__main__":
    unittest.main()
