import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.mrc_io import write_mode2_mrc_for_test
from segmentation_integration.segmentation_preview import (
    load_cryovia_label_mask,
    load_vesicle_picker_union_mask,
    render_segmentation_comparison,
    render_segmentation_overlay,
)


def write_vesicle_picker_mask(path: Path) -> None:
    segmentation = np.zeros((12, 12), dtype=bool)
    segmentation[3:9, 3:9] = True
    composite = np.ones((12, 12), dtype=np.uint16)
    composite[segmentation] = 2
    with path.open("wb") as handle:
        pickle.dump(
            {
                "masks": [{"prime_key": 2, "area": int(segmentation.sum())}],
                "composite_mask": composite,
                "uid": 7,
            },
            handle,
        )


class SegmentationPreviewTests(unittest.TestCase):
    def test_load_cryovia_label_mask_reads_sparse_label_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mic_labels.npz"
            labels = np.zeros((1, 8, 8), dtype=np.uint8)
            labels[0, 2:6, 2:6] = 1
            np.savez_compressed(path, data=labels)

            mask = load_cryovia_label_mask(path)

        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(int(mask.sum()), 16)

    def test_load_vesicle_picker_union_mask_reads_prime_encoded_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "7_vesicles_filtered.pkl"
            write_vesicle_picker_mask(path)

            mask = load_vesicle_picker_union_mask(path)

        self.assertEqual(mask.shape, (12, 12))
        self.assertEqual(int(mask.sum()), 36)

    def test_render_segmentation_overlay_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            micrograph_path = root / "mic.mrc"
            out_path = root / "cryovia.png"
            write_mode2_mrc_for_test(micrograph_path, np.arange(400, dtype=np.float32).reshape(20, 20))
            mask = np.zeros((10, 10), dtype=bool)
            mask[2:8, 2:8] = True

            render_segmentation_overlay(
                micrograph_path=micrograph_path,
                output_path=out_path,
                masks=[mask],
                labels=["CryoVIA"],
                colors=["lime"],
                title="mic CryoVIA",
                mask_downsample=2,
            )

            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_segmentation_comparison_writes_side_by_side_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            micrograph_path = root / "mic.mrc"
            out_path = root / "comparison.png"
            write_mode2_mrc_for_test(micrograph_path, np.arange(400, dtype=np.float32).reshape(20, 20))
            cryovia = np.zeros((10, 10), dtype=bool)
            sam_found = np.zeros((10, 10), dtype=bool)
            sam_filtered = np.zeros((10, 10), dtype=bool)
            cryovia[2:8, 2:8] = True
            sam_found[1:8, 1:8] = True
            sam_filtered[3:7, 3:7] = True

            render_segmentation_comparison(
                micrograph_path=micrograph_path,
                output_path=out_path,
                cryovia_mask=cryovia,
                sam_found_mask=sam_found,
                sam_filtered_mask=sam_filtered,
                mask_downsample=2,
                title="mic comparison",
            )

            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
