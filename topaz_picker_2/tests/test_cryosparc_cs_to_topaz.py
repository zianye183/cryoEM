import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from topaz_picker.scripts.cryosparc_cs_to_topaz import convert_cryosparc_cs_to_topaz


def write_particles_cs(path: Path, rows: list[tuple[str, tuple[int, int], float, float]]) -> None:
    dtype = [
        ("uid", "<u8"),
        ("location/micrograph_path", "S256"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
    ]
    arr = np.zeros(len(rows), dtype=dtype)
    for idx, (micrograph_path, shape, x_frac, y_frac) in enumerate(rows):
        arr[idx]["uid"] = idx + 1
        arr[idx]["location/micrograph_path"] = micrograph_path.encode("utf-8")
        arr[idx]["location/micrograph_shape"] = shape
        arr[idx]["location/center_x_frac"] = x_frac
        arr[idx]["location/center_y_frac"] = y_frac
    with path.open("wb") as handle:
        np.save(handle, arr)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class CryosparcCsToTopazTests(unittest.TestCase):
    def test_converts_fractional_coordinates_and_splits_by_micrograph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_dir = root / "particles"
            micrographs_dir = root / "micrographs"
            output_dir = root / "metadata"
            particles_dir.mkdir()
            micrographs_dir.mkdir()

            (micrographs_dir / "mic_a.mrc").touch()
            (micrographs_dir / "mic_b.mrc").touch()

            write_particles_cs(
                particles_dir / "particles_accepted.cs",
                [
                    ("J42/imported/mic_a.mrc", (100, 200), 0.25, 0.50),
                    ("J42/imported/mic_a.mrc", (100, 200), 0.75, 0.10),
                    ("J42/imported/mic_b.mrc", (120, 300), 0.50, 0.25),
                ],
            )

            summary = convert_cryosparc_cs_to_topaz(
                particles_dir=particles_dir,
                micrographs_dir=micrographs_dir,
                output_dir=output_dir,
                downsample_factor=2.0,
                test_fraction=0.5,
                seed=7,
            )

            particles = read_tsv(output_dir / "particles_all.tsv")
            particles_txt = read_tsv(output_dir / "particles_all.txt")
            self.assertEqual(len(particles), 3)
            self.assertEqual(particles_txt, particles)
            self.assertEqual(list(particles[0].keys()), ["image_name", "x_coord", "y_coord"])
            by_image = {(row["image_name"], row["x_coord"], row["y_coord"]) for row in particles}
            self.assertIn(("mic_a", "25.000", "25.000"), by_image)
            self.assertIn(("mic_a", "75.000", "5.000"), by_image)
            self.assertIn(("mic_b", "75.000", "15.000"), by_image)

            images = read_tsv(output_dir / "images_all.tsv")
            self.assertNotIn(b"\r\n", (output_dir / "images_all.tsv").read_bytes())
            self.assertNotIn(b"\r\n", (output_dir / "train_particles.txt").read_bytes())
            self.assertEqual(
                {row["image_name"]: row["path"] for row in images},
                {
                    "mic_a": str(micrographs_dir / "mic_a.mrc"),
                    "mic_b": str(micrographs_dir / "mic_b.mrc"),
                },
            )

            train_particles = read_tsv(output_dir / "train_particles.tsv")
            test_particles = read_tsv(output_dir / "test_particles.tsv")
            self.assertEqual(read_tsv(output_dir / "train_particles.txt"), train_particles)
            self.assertEqual(read_tsv(output_dir / "test_particles.txt"), test_particles)
            self.assertEqual(
                {row["image_name"] for row in train_particles}.isdisjoint(
                    {row["image_name"] for row in test_particles}
                ),
                True,
            )

            saved_summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(saved_summary["num_particles"], 3)
            self.assertEqual(
                saved_summary["outputs"]["train_particles"]["txt"],
                str(output_dir / "train_particles.txt"),
            )
            self.assertEqual(summary["num_micrographs"], 2)

    def test_fails_when_micrograph_stems_are_ambiguous_for_topaz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_dir = root / "particles"
            micrographs_dir = root / "micrographs"
            output_dir = root / "metadata"
            particles_dir.mkdir()
            micrographs_dir.mkdir()

            (micrographs_dir / "mic_a.mrc").touch()
            (micrographs_dir / "mic_a.tif").touch()

            write_particles_cs(
                particles_dir / "particles_accepted.cs",
                [("J42/imported/mic_a.mrc", (100, 100), 0.5, 0.5)],
            )

            with self.assertRaisesRegex(ValueError, "duplicate Topaz image_name stems"):
                convert_cryosparc_cs_to_topaz(
                    particles_dir=particles_dir,
                    micrographs_dir=micrographs_dir,
                    output_dir=output_dir,
                )

    def test_rejects_coordinates_on_exclusive_image_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_dir = root / "particles"
            micrographs_dir = root / "micrographs"
            output_dir = root / "metadata"
            particles_dir.mkdir()
            micrographs_dir.mkdir()

            (micrographs_dir / "mic_a.mrc").touch()
            (micrographs_dir / "mic_b.mrc").touch()

            write_particles_cs(
                particles_dir / "particles_accepted.cs",
                [
                    ("J42/imported/mic_a.mrc", (100, 100), 1.0, 0.5),
                    ("J42/imported/mic_b.mrc", (100, 100), 0.5, 0.5),
                ],
            )

            with self.assertRaisesRegex(ValueError, "outside their source image bounds"):
                convert_cryosparc_cs_to_topaz(
                    particles_dir=particles_dir,
                    micrographs_dir=micrographs_dir,
                    output_dir=output_dir,
                )

    def test_fails_when_required_particle_fields_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particles_dir = root / "particles"
            micrographs_dir = root / "micrographs"
            output_dir = root / "metadata"
            particles_dir.mkdir()
            micrographs_dir.mkdir()

            arr = np.zeros(1, dtype=[("location/center_x_frac", "<f4")])
            with (particles_dir / "bad.cs").open("wb") as handle:
                np.save(handle, arr)

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                convert_cryosparc_cs_to_topaz(
                    particles_dir=particles_dir,
                    micrographs_dir=micrographs_dir,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
