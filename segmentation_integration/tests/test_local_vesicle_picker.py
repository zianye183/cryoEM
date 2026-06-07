import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from io import StringIO
from copy import deepcopy

import numpy as np

from segmentation_integration.scripts.run_local_vesicle_picker import (
    MicrographRecord,
    emit_progress,
    export_masks_to_disk_without_mutating,
    micrograph_records_from_particles,
    records_for_shard,
)


class LocalVesiclePickerTests(unittest.TestCase):
    def test_local_runner_import_does_not_load_matplotlib(self) -> None:
        code = (
            "import sys; "
            "import segmentation_integration.scripts.run_local_vesicle_picker; "
            "print('matplotlib' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), "False")

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

    def test_emit_progress_writes_cluster_log_friendly_line(self) -> None:
        stream = StringIO()

        emit_progress(
            stream,
            index=2,
            total=5,
            record=MicrographRecord(uid=17, path=Path("/data/mic.mrc")),
            found_masks=4,
            filtered_masks=3,
            status="done",
            elapsed_seconds=1.234,
        )

        self.assertEqual(
            stream.getvalue().strip(),
            "[local vesicle picker] 2/5 uid=17 status=done found=4 filtered=3 elapsed=1.23s mic=mic.mrc",
        )

    def test_export_masks_to_disk_without_mutating_keeps_segmentation_for_filtering(self) -> None:
        masks = [{"segmentation": np.ones((2, 2), dtype=bool), "area": 4}]
        exported = []

        def fake_export(export_masks, filename, micrograph_uid, compression):
            exported.append((filename, micrograph_uid, compression))
            for mask in export_masks:
                mask.pop("segmentation", None)

        export_masks_to_disk_without_mutating(
            fake_export,
            masks,
            "out.pkl",
            micrograph_uid=7,
            compression="uint64",
        )

        self.assertIn("segmentation", masks[0])
        self.assertEqual(exported, [("out.pkl", 7, "uint64")])
        self.assertEqual(deepcopy(masks)[0]["area"], 4)

    def test_records_for_shard_uses_chunk_size(self) -> None:
        records = [
            MicrographRecord(uid=index, path=Path(f"/data/mic_{index}.mrc"))
            for index in range(9)
        ]

        selected = records_for_shard(records, shard_index=1, chunk_size=4)

        self.assertEqual([record.uid for record in selected], [4, 5, 6, 7])

    def test_records_for_shard_uses_balanced_shard_count_without_chunk_size(self) -> None:
        records = [
            MicrographRecord(uid=index, path=Path(f"/data/mic_{index}.mrc"))
            for index in range(10)
        ]

        selected = records_for_shard(records, shard_index=2, shard_count=3)

        self.assertEqual([record.uid for record in selected], [7, 8, 9])


if __name__ == "__main__":
    unittest.main()
