import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.scripts.run_local_vesicle_picker import micrograph_records_from_particles


class LocalVesiclePickerTests(unittest.TestCase):
    def test_micrograph_records_from_particles_resolves_unique_uid_paths(self) -> None:
        dtype = [
            ("location/micrograph_uid", "<u8"),
            ("location/micrograph_path", "S128"),
            ("location/micrograph_shape", "<u4", (2,)),
            ("location/center_x_frac", "<f4"),
            ("location/center_y_frac", "<f4"),
        ]
        particles = np.zeros(3, dtype=dtype)
        particles["location/micrograph_uid"] = [7, 7, 8]
        particles["location/micrograph_path"] = [b"cryosparc/mic_a.mrc", b"cryosparc/mic_a.mrc", b"mic_b.mrc"]

        with tempfile.TemporaryDirectory() as tmp:
            micrographs_dir = Path(tmp)
            (micrographs_dir / "mic_a.mrc").write_bytes(b"")
            (micrographs_dir / "mic_b.mrc").write_bytes(b"")

            records = micrograph_records_from_particles(particles, micrographs_dir)

        self.assertEqual([record.uid for record in records], [7, 8])
        self.assertEqual([record.path.name for record in records], ["mic_a.mrc", "mic_b.mrc"])


if __name__ == "__main__":
    unittest.main()
