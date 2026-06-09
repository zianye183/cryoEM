import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py

from segmentation_integration.cryovia_runner import (
    _has_keras_v3_vars_group,
    _prepare_cryovia_weights,
    _runtime_model_directory,
    configure_cryovia_home,
)


class FakeBuiltKerasModel:
    def __init__(self) -> None:
        self.loaded_from = None
        self.saved_to = None

    def load_weights(self, path: Path) -> None:
        self.loaded_from = path

    def save_weights(self, path: Path) -> None:
        self.saved_to = path
        with h5py.File(path, "w") as handle:
            handle.create_group("vars")


class FakeCryoviaModel:
    def __init__(self) -> None:
        self.built_model = FakeBuiltKerasModel()

    def build_model(self) -> FakeBuiltKerasModel:
        return self.built_model


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

    def test_has_keras_v3_vars_group_detects_new_weight_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "best.weights.h5"
            with h5py.File(weights_path, "w") as handle:
                handle.create_group("vars")

            self.assertTrue(_has_keras_v3_vars_group(weights_path))

    def test_prepare_cryovia_weights_converts_legacy_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights_path = root / "best.weights.h5"
            legacy_weights_path = root / "best_weights.h5"
            with h5py.File(weights_path, "w") as handle:
                handle.create_group("model_weights")
            with h5py.File(legacy_weights_path, "w") as handle:
                handle.create_group("model_weights")
            model = FakeCryoviaModel()

            _prepare_cryovia_weights(model=model, weights_path=weights_path, legacy_weights_path=legacy_weights_path)

            self.assertEqual(model.built_model.loaded_from, legacy_weights_path)
            self.assertEqual(model.built_model.saved_to, weights_path)
            self.assertTrue(_has_keras_v3_vars_group(weights_path))


if __name__ == "__main__":
    unittest.main()
