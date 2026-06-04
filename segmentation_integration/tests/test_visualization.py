import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.mrc_io import write_mode2_mrc_for_test
from segmentation_integration.visualization import (
    render_mask_png,
    render_overlay,
)


class VisualizationTests(unittest.TestCase):
    def test_render_overlay_writes_png_from_synthetic_mrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrc_path = root / "mic.mrc"
            out_path = root / "overlay.png"
            image = np.arange(100, dtype=np.float32).reshape(10, 10)
            write_mode2_mrc_for_test(mrc_path, image)
            band = np.zeros((5, 5), dtype=bool)
            band[1:4, 1:4] = True

            render_overlay(
                micrograph_path=mrc_path,
                output_path=out_path,
                membrane_band=band,
                downsample=2,
                original_xy=np.asarray([[2, 2], [8, 8]], dtype=float),
                kept_xy=np.asarray([[2, 2]], dtype=float),
                rejected_xy=np.asarray([[8, 8]], dtype=float),
            )

            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_mask_png_writes_standalone_membrane_band_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_path = root / "mask.png"
            band = np.zeros((5, 5), dtype=bool)
            band[1:4, 1:4] = True

            render_mask_png(
                output_path=out_path,
                membrane_band=band,
                title="mic_a membrane band",
            )

            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
