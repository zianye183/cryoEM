import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.cryosparc_io import (
    REQUIRED_LOCATION_FIELDS,
    decode_path,
    load_particles_cs,
    patch_csg_cs_path,
    resolve_micrograph_path,
    write_cs,
)


def make_particles() -> np.ndarray:
    dtype = [
        ("uid", "<u8"),
        ("location/micrograph_uid", "<u8"),
        ("location/micrograph_path", "S128"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
    ]
    arr = np.zeros(2, dtype=dtype)
    arr["uid"] = [10, 11]
    arr["location/micrograph_uid"] = [7, 7]
    arr["location/micrograph_path"] = [b"J1/imported/mic_a.mrc", b"J1/imported/mic_a.mrc"]
    arr["location/micrograph_shape"] = [(100, 200), (100, 200)]
    arr["location/center_x_frac"] = [0.25, 0.75]
    arr["location/center_y_frac"] = [0.50, 0.10]
    return arr


class CryosparcIoTests(unittest.TestCase):
    def test_load_particles_cs_requires_location_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particles.cs"
            with path.open("wb") as handle:
                np.save(handle, make_particles())

            loaded = load_particles_cs(path)

        self.assertEqual(set(REQUIRED_LOCATION_FIELDS) - set(loaded.dtype.names or ()), set())
        self.assertEqual(decode_path(loaded[0]["location/micrograph_path"]), "J1/imported/mic_a.mrc")

    def test_load_particles_cs_rejects_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.cs"
            with path.open("wb") as handle:
                np.save(handle, np.zeros(1, dtype=[("location/center_x_frac", "<f4")]))

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_particles_cs(path)

    def test_write_cs_preserves_dtype_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset.cs"
            original = make_particles()

            write_cs(path, original[[1]])
            loaded = np.load(path, allow_pickle=False)

        self.assertEqual(loaded.dtype, original.dtype)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(int(loaded[0]["uid"]), 11)

    def test_patch_csg_replaces_single_referenced_cs_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csg = root / "particles.csg"
            csg.write_text(json.dumps({"files": [{"path": "/old/particles.cs"}]}))

            patch_csg_cs_path(csg, root / "filtered.csg", "particles.cs", root / "filtered_particles.cs")
            patched = json.loads((root / "filtered.csg").read_text())

        self.assertEqual(patched["files"][0]["path"], str(root / "filtered_particles.cs"))

    def test_patch_csg_replaces_single_yaml_referenced_cs_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csg = root / "particles.csg"
            csg.write_text("files:\n  - path: >J71_particles_exported.cs\n")

            patch_csg_cs_path(csg, root / "filtered.csg", "J71_particles_exported.cs", root / "filtered_particles.cs")
            patched = (root / "filtered.csg").read_text()

        self.assertIn(f"path: {root / 'filtered_particles.cs'}", patched)
        self.assertNotIn(">J71_particles_exported.cs", patched)

    def test_patch_csg_replaces_repeated_same_yaml_referenced_cs_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csg = root / "particles.csg"
            csg.write_text(
                "files:\n"
                "  - path: >exports/groups/J71_particles/J71_particles_exported.cs\n"
                "metadata:\n"
                "  source: >exports/groups/J71_particles/J71_particles_exported.cs\n"
            )

            patch_csg_cs_path(csg, root / "filtered.csg", "J71_particles_exported.cs", root / "filtered_particles.cs")
            patched = (root / "filtered.csg").read_text()

        self.assertEqual(patched.count(str(root / "filtered_particles.cs")), 2)
        self.assertNotIn("J71_particles_exported.cs", patched)

    def test_patch_csg_rejects_different_yaml_referenced_cs_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csg = root / "particles.csg"
            csg.write_text(
                "files:\n"
                "  - path: >exports/a/J71_particles_exported.cs\n"
                "metadata:\n"
                "  source: >exports/b/J71_particles_exported.cs\n"
            )

            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                patch_csg_cs_path(csg, root / "filtered.csg", "J71_particles_exported.cs", root / "filtered_particles.cs")

    def test_patch_csg_rejects_ambiguous_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csg = root / "particles.csg"
            csg.write_text(json.dumps({"a": "particles.cs", "b": "/other/particles.cs"}))

            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                patch_csg_cs_path(csg, root / "filtered.csg", "particles.cs", root / "filtered_particles.cs")

    def test_resolve_micrograph_path_matches_local_file_after_numeric_export_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_name = "FoilHole_6367920_Data_5347108_17_20260207_032343_fractions.mrc"
            local_path = root / local_name
            local_path.write_bytes(b"")

            resolved = resolve_micrograph_path(
                ">J26/imported/003884161951494007383_" + local_name,
                root,
                {local_path.stem: local_path},
            )

        self.assertEqual(resolved, local_path)


if __name__ == "__main__":
    unittest.main()
