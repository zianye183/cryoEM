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


def write_mask(path: Path, uid: int, area: int, roundness: float, offset: int = 2) -> None:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[offset : offset + 3, offset : offset + 3] = True
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


def write_two_mask_pickle(path: Path, uid: int) -> None:
    first = np.zeros((10, 10), dtype=bool)
    first[1:4, 1:4] = True
    second = np.zeros((10, 10), dtype=bool)
    second[5:9, 5:9] = True
    composite = np.ones((10, 10), dtype=np.uint16)
    composite[first] *= 2
    composite[second] *= 3
    with path.open("wb") as handle:
        pickle.dump(
            {
                "masks": [
                    {"prime_key": 2, "area": 9, "roundness": 0.9},
                    {"prime_key": 3, "area": 16, "roundness": 0.9},
                ],
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
            comparison_size = (output_dir / "7_found_filter_debug.png").stat().st_size
            metrics_size = (output_dir / "7_found_mask_metrics.png").stat().st_size

        self.assertEqual(summary["rendered_comparisons"], 1)
        self.assertGreater(comparison_size, 0)
        self.assertGreater(metrics_size, 0)
        self.assertIn("uid\tstage\tmask_index\tstatus\treject_reasons\tarea\troundness", stats)
        self.assertIn("7\tfound\t0\tkept\t\t36\t0.5", stats)
        self.assertIn("7\tfiltered\t0\toutput\t\t36\t0.5", stats)

    def test_render_mask_filter_qc_adds_density_metrics_when_micrographs_available(self) -> None:
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
            image = np.zeros((10, 10), dtype=np.float32)
            image[2:5, 2:5] = 10
            image[3:4, 3:4] = 20
            write_mode2_mrc_for_test(micrographs_dir / "mic_a.mrc", image)
            write_mask(found_dir / "7_vesicles.pkl", uid=7, area=9, roundness=0.9)
            write_mask(filtered_dir / "7_vesicles_filtered.pkl", uid=7, area=9, roundness=0.9)

            render_mask_filter_qc(
                found_mask_dir=found_dir,
                filtered_mask_dir=filtered_dir,
                output_dir=output_dir,
                max_micrographs=0,
                particles_cs=particles_cs,
                micrographs_dir=micrographs_dir,
                downsample=1,
            )

            stats = (output_dir / "mask_filter_stats.tsv").read_text()

        self.assertIn("interior_mean", stats)
        self.assertIn("membrane_mean", stats)
        self.assertIn("local_bg_mean", stats)
        self.assertIn("ring_contrast", stats)
        self.assertIn("interior_contrast", stats)
        self.assertIn("interior_std", stats)

    def test_render_mask_filter_qc_classifies_found_masks_with_reject_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            found_dir = root / "found"
            filtered_dir = root / "filtered"
            output_dir = root / "qc"
            found_dir.mkdir()
            filtered_dir.mkdir()
            filter_ini = root / "filter_vesicles.ini"
            filter_ini.write_text("[area]\nmin = 10\nmax = 20\n\n[roundness]\nmin = 0.75\nmax = 1\n")
            write_two_mask_pickle(found_dir / "7_vesicles.pkl", uid=7)
            write_mask(filtered_dir / "7_vesicles_filtered.pkl", uid=7, area=16, roundness=0.9, offset=5)

            summary = render_mask_filter_qc(
                found_mask_dir=found_dir,
                filtered_mask_dir=filtered_dir,
                output_dir=output_dir,
                max_micrographs=0,
                filter_parameters=filter_ini,
            )

            stats = (output_dir / "mask_filter_stats.tsv").read_text()
            debug_size = (output_dir / "7_found_filter_debug.png").stat().st_size
            metrics_size = (output_dir / "7_found_mask_metrics.png").stat().st_size

        self.assertEqual(summary["found_masks_kept"], 1)
        self.assertEqual(summary["found_masks_rejected"], 1)
        self.assertGreater(debug_size, 0)
        self.assertGreater(metrics_size, 0)
        self.assertIn("uid\tstage\tmask_index\tstatus\treject_reasons", stats)
        self.assertIn("7\tfound\t0\trejected\tarea_min", stats)
        self.assertIn("7\tfound\t1\tkept\t", stats)


if __name__ == "__main__":
    unittest.main()
