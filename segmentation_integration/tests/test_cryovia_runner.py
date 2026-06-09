import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from segmentation_integration.cryovia_runner import configure_cryovia_home, _runtime_model_directory


class CryoviaRunnerTests(unittest.TestCase):
    def test_configure_cryovia_home_sets_home_before_cryovia_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cryovia_home = Path(tmp) / "cryovia_runtime_home"
            with patch.dict(os.environ, {"HOME": str(Path(tmp) / "old_home")}, clear=False):
                resolved = configure_cryovia_home(cryovia_home)

                self.assertEqual(resolved, cryovia_home)
                self.assertEqual(os.environ["HOME"], str(cryovia_home))
                self.assertTrue(cryovia_home.is_dir())

    def test_runtime_model_directory_uses_packaged_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cryovia_home = Path(tmp) / ".cryovia"

            model_dir = _runtime_model_directory(model="Default_thin", cryovia_home=cryovia_home)

        self.assertEqual(model_dir, cryovia_home / "SegmentationModels" / "Default_thin")

    def test_runtime_model_directory_stages_explicit_model_under_cryovia_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_model = root / "custom_model"
            source_model.mkdir()
            (source_model / "Segmentator.pickle").touch()
            (source_model / "best.weights.h5").touch()
            cryovia_home = root / ".cryovia"

            model_dir = _runtime_model_directory(model=source_model, cryovia_home=cryovia_home)

            self.assertEqual(model_dir, cryovia_home / "SegmentationModels" / "custom_model")
            self.assertTrue((model_dir / "Segmentator.pickle").exists())
            self.assertTrue((model_dir / "best.weights.h5").exists())


if __name__ == "__main__":
    unittest.main()
