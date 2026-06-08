import json
import tempfile
import unittest
from pathlib import Path

from segmentation_integration.scripts.run_segmentation_visual_benchmark import (
    DEFAULT_CRYOVIA_MODEL,
    parse_manifest,
    run_visual_benchmark,
)


class VisualBenchmarkCliTests(unittest.TestCase):
    def test_default_cryovia_model_is_default_thin(self) -> None:
        self.assertEqual(DEFAULT_CRYOVIA_MODEL, "Default_thin")

    def test_parse_manifest_accepts_newline_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "0007_first.mrc"
            second = root / "0008_second.mrc"
            first.touch()
            second.touch()
            manifest = root / "manifest.txt"
            manifest.write_text(f"{first}\n{second}\n")

            records = parse_manifest(manifest)

        self.assertEqual([record.path for record in records], [first, second])
        self.assertEqual([record.uid for record in records], [7, 8])

    def test_parse_manifest_accepts_tsv_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "0007_first.mrc"
            first.touch()
            manifest = root / "manifest.tsv"
            manifest.write_text(f"micrograph_path\tnote\n{first}\tkeep\n")

            records = parse_manifest(manifest)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].path, first)
        self.assertEqual(records[0].uid, 7)

    def test_run_visual_benchmark_dry_run_writes_review_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "0007_first.mrc"
            second = root / "0008_second.mrc"
            first.touch()
            second.touch()
            manifest = root / "manifest.tsv"
            manifest.write_text(f"path\n{first}\n{second}\n")
            output_dir = root / "benchmark"

            summary = run_visual_benchmark(
                manifest=manifest,
                output_dir=output_dir,
                vesicle_picker_dir=root / "vesicle-picker",
                sam_find_parameters=root / "find_vesicles.ini",
                sam_filter_parameters=root / "filter_vesicles.ini",
                dry_run=True,
                limit=1,
            )

            config = json.loads((output_dir / "config.json").read_text())
            run_summary = json.loads((output_dir / "run_summary.json").read_text())
            manifest_exists = (output_dir / "manifest.tsv").exists()
            index_exists = (output_dir / "index.html").exists()

        self.assertEqual(summary["micrographs"], 1)
        self.assertEqual(config["cryovia_model"], "Default_thin")
        self.assertEqual(run_summary["mode"], "dry_run")
        self.assertTrue(manifest_exists)
        self.assertTrue(index_exists)

    def test_run_visual_benchmark_records_split_env_skip_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "0007_first.mrc"
            first.touch()
            manifest = root / "manifest.tsv"
            manifest.write_text(f"path\n{first}\n")
            output_dir = root / "benchmark"

            run_visual_benchmark(
                manifest=manifest,
                output_dir=output_dir,
                vesicle_picker_dir=root / "vesicle-picker",
                sam_find_parameters=root / "find_vesicles.ini",
                sam_filter_parameters=root / "filter_vesicles.ini",
                dry_run=True,
                skip_sam_run=True,
                skip_cryovia_run=True,
            )

            config = json.loads((output_dir / "config.json").read_text())

        self.assertTrue(config["skip_sam_run"])
        self.assertTrue(config["skip_cryovia_run"])


if __name__ == "__main__":
    unittest.main()
