import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from segmentation_integration.cryovia_runner import configure_cryovia_home


class CryoviaRunnerTests(unittest.TestCase):
    def test_configure_cryovia_home_sets_home_before_cryovia_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cryovia_home = Path(tmp) / "cryovia_runtime_home"
            with patch.dict(os.environ, {"HOME": str(Path(tmp) / "old_home")}, clear=False):
                resolved = configure_cryovia_home(cryovia_home)

                self.assertEqual(resolved, cryovia_home)
                self.assertEqual(os.environ["HOME"], str(cryovia_home))
                self.assertTrue(cryovia_home.is_dir())


if __name__ == "__main__":
    unittest.main()
