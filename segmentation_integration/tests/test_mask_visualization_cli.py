import pickle
import tempfile
import unittest
from io import StringIO
from pathlib import Path

import numpy as np

from segmentation_integration.scripts.visualize_vesicle_masks import render_mask_previews


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
    arr[0]["location/micrograph_path"] = b"J1/imported/mic_a.mrc"
    arr[0]["location/micrograph_shape"] = (100, 100)
    arr[0]["location/center_x_frac"] = 0.5
    arr[0]["location/center_y_frac"] = 0.5
    with path.open("wb") as handle:
        np.save(handle, arr)


def write_mask(path: Path) -> None:
    segmentation = np.zeros((10, 10), dtype=bool)
    segmentation[2:8, 2:8] = True
    composite = np.ones((10, 10), dtype=np.uint16)
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


class MaskVisualizationCliTests(unittest.TestCase):
    def test_render_mask_previews_writes_mask_png_after_vesicle_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_cs = root / "particles.cs"
            mask_dir = root / "masks"
            output_dir = root / "mask_previews"
            mask_dir.mkdir()
            write_particles_cs(particles_cs)
            write_mask(mask_dir / "7_vesicles_filtered.pkl")

            summary = render_mask_previews(
                particles_cs=particles_cs,
                vesicle_mask_dir=mask_dir,
                output_dir=output_dir,
                psize_a=1.0,
                downsample=10,
                membrane_band_a=10.0,
                max_masks=0,
            )

            out_path = output_dir / "mic_a_membrane_band_mask.png"

            self.assertEqual(summary["rendered_masks"], 1)
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_mask_previews_emits_per_micrograph_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_cs = root / "particles.cs"
            mask_dir = root / "masks"
            output_dir = root / "mask_previews"
            mask_dir.mkdir()
            write_particles_cs(particles_cs)
            write_mask(mask_dir / "7_vesicles_filtered.pkl")
            stream = StringIO()

            render_mask_previews(
                particles_cs=particles_cs,
                vesicle_mask_dir=mask_dir,
                output_dir=output_dir,
                psize_a=1.0,
                downsample=10,
                membrane_band_a=10.0,
                max_masks=0,
                progress_stream=stream,
            )

        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(lines[0], "[mask preview] 1/1 uid=7 status=start mic=mic_a.mrc")
        self.assertRegex(lines[1], r"^\[mask preview\] 1/1 uid=7 status=done masks=1 elapsed=\d+\.\d{2}s mic=mic_a.mrc$")


if __name__ == "__main__":
    unittest.main()
