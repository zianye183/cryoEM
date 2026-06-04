import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from segmentation_integration.mrc_io import read_mrc


def write_mrc(path: Path, image: np.ndarray, mode: int, dtype: str) -> None:
    image = np.asarray(image, dtype=dtype)
    ny, nx = image.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nx, ny, 1, mode)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(image.tobytes(order="C"))


class MrcIoTests(unittest.TestCase):
    def test_read_mrc_supports_int16_mode_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mode1.mrc"
            write_mrc(path, np.array([[1, 2], [3, 4]]), 1, "<i2")

            image = read_mrc(path)

        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_array_equal(image, np.array([[1, 2], [3, 4]], dtype=np.float32))

    def test_read_mrc_supports_uint16_mode_6(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mode6.mrc"
            write_mrc(path, np.array([[100, 200]], dtype=np.uint16), 6, "<u2")

            image = read_mrc(path)

        np.testing.assert_array_equal(image, np.array([[100, 200]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
