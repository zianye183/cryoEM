import argparse
import tempfile
import unittest
from pathlib import Path

from denoising.scripts.denoising_run_common import (
    add_sharding_args,
    allow_existing_output_dir,
    files_for_shard,
    output_path_for_input,
    select_pending_files,
)


class DenoisingParallelTests(unittest.TestCase):
    def test_files_for_shard_uses_chunk_size_and_array_index(self) -> None:
        files = [Path(f"mic_{index:03d}.mrc") for index in range(12)]

        selected = files_for_shard(files, shard_index=2, chunk_size=5)

        self.assertEqual(
            selected,
            [Path("mic_010.mrc"), Path("mic_011.mrc")],
        )

    def test_files_for_shard_uses_balanced_shard_count_without_chunk_size(self) -> None:
        files = [Path(f"mic_{index:03d}.mrc") for index in range(10)]

        selected = files_for_shard(files, shard_index=1, shard_count=3)

        self.assertEqual(
            selected,
            [Path("mic_004.mrc"), Path("mic_005.mrc"), Path("mic_006.mrc")],
        )

    def test_select_pending_files_skips_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / "mic_a.mrc", root / "mic_b.mrc"]
            for path in files:
                path.write_bytes(b"")
            output_dir = root / "out"
            output_dir.mkdir()
            (output_dir / "mic_a_draco_denoised.mrc").write_bytes(b"done")

            pending = select_pending_files(
                files,
                output_dir=output_dir,
                suffix="_draco_denoised",
                skip_existing=True,
            )

        self.assertEqual([path.name for path in pending], ["mic_b.mrc"])

    def test_output_path_for_input_uses_suffix_before_mrc_extension(self) -> None:
        output = output_path_for_input(
            Path("/inputs/FoilHole_1_fractions.mrc"),
            Path("/outputs"),
            "_draco_denoised",
        )

        self.assertEqual(output, Path("/outputs/FoilHole_1_fractions_draco_denoised.mrc"))

    def test_add_sharding_args_accepts_slurm_array_style_options(self) -> None:
        parser = argparse.ArgumentParser()
        add_sharding_args(parser)

        args = parser.parse_args(
            [
                "--manifest",
                "micrographs.tsv",
                "--chunk-size",
                "10",
                "--shard-index",
                "7",
                "--skip-existing",
            ]
        )

        self.assertEqual(args.manifest, "micrographs.tsv")
        self.assertEqual(args.chunk_size, 10)
        self.assertEqual(args.shard_index, 7)
        self.assertTrue(args.skip_existing)

    def test_allow_existing_output_dir_is_enabled_for_resume_mode(self) -> None:
        self.assertTrue(allow_existing_output_dir(overwrite=False, skip_existing=True))
        self.assertTrue(allow_existing_output_dir(overwrite=True, skip_existing=False))
        self.assertFalse(allow_existing_output_dir(overwrite=False, skip_existing=False))


if __name__ == "__main__":
    unittest.main()
