import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.scripts.visualize_vesicle_mask_filtering import render_mask_filter_qc


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
    def test_render_mask_filter_qc_writes_side_by_side_png_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            found_dir = root / "found"
            filtered_dir = root / "filtered"
            output_dir = root / "qc"
            found_dir.mkdir()
            filtered_dir.mkdir()
            write_mask(found_dir / "7_vesicles.pkl", uid=7, area=36, roundness=0.5)
            write_mask(filtered_dir / "7_vesicles_filtered.pkl", uid=7, area=36, roundness=0.5)

            summary = render_mask_filter_qc(
                found_mask_dir=found_dir,
                filtered_mask_dir=filtered_dir,
                output_dir=output_dir,
                max_micrographs=0,
            )

            stats = (output_dir / "mask_filter_stats.tsv").read_text()
            comparison_size = (output_dir / "7_found_vs_filtered_masks.png").stat().st_size

        self.assertEqual(summary["rendered_comparisons"], 1)
        self.assertGreater(comparison_size, 0)
        self.assertIn("uid\tstage\tmask_index\tarea\troundness", stats)
        self.assertIn("7\tfound\t0\t36\t0.5", stats)
        self.assertIn("7\tfiltered\t0\t36\t0.5", stats)


if __name__ == "__main__":
    unittest.main()
