import tempfile
import unittest
from pathlib import Path

from segmentation_integration.vesicle_picker_invocation import (
    prepare_vesicle_picker_run,
    prepare_vesicle_picker_run_for_dataset,
)


class VesiclePickerInvocationTests(unittest.TestCase):
    def test_prepare_run_writes_configs_and_end_to_end_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vesicle_picker_dir = root / "vesicle-picker"
            vesicle_picker_dir.mkdir()
            output_dir = root / "run"

            summary = prepare_vesicle_picker_run(
                output_dir=output_dir,
                vesicle_picker_dir=vesicle_picker_dir,
                csparc_login=Path("/cluster/csparc_login.ini"),
                project_id="P1",
                workspace_id="W2",
                job_id="J3",
                job_type="curate",
                model_weights_path=Path("/cluster/sam_vit_l_0b3195.pth"),
                particles_cs=Path("/exports/particles.cs"),
                particles_csg=Path("/exports/particles.csg"),
                micrographs_dir=Path("/exports/micrographs"),
                filter_output_dir=Path("/outputs/filtered_particles"),
            )

            find_ini = (output_dir / "parameters" / "find_vesicles.ini").read_text()
            filter_ini = (output_dir / "parameters" / "filter_vesicles.ini").read_text()
            script = (output_dir / "run_end_to_end.sh").read_text()

        self.assertIn("PID = P1", find_ini)
        self.assertIn("JID = J3", find_ini)
        self.assertIn("WID", script)
        self.assertIn("model_type = vit_l", find_ini)
        self.assertIn("psize = 1.068", find_ini)
        self.assertIn("[area]", filter_ini)
        self.assertIn("min = 2.5e3", filter_ini)
        self.assertIn("find_vesicles.py", script)
        self.assertIn("filter_vesicles.py", script)
        self.assertLess(script.index("filter_vesicles.py"), script.index("visualize_vesicle_masks.py"))
        self.assertLess(script.index("visualize_vesicle_masks.py"), script.index("filter_cryosparc_picks_by_vesicle_masks.py"))
        self.assertIn("--skip-mask-pngs", script)
        self.assertIn("filter_cryosparc_picks_by_vesicle_masks.py", script)
        self.assertEqual(summary["find_parameters"], str(output_dir / "parameters" / "find_vesicles.ini"))

    def test_prepare_run_for_dataset_uses_standard_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_export_dir = root / "J70_particles"
            particle_export_dir.mkdir()
            (particle_export_dir / "J70_particles_exported.cs").write_bytes(b"")
            (particle_export_dir / "J70_particles_exported.csg").write_text("{}")

            summary = prepare_vesicle_picker_run_for_dataset(
                dataset="draco",
                output_dir=root / "run",
                particle_export_dir=particle_export_dir,
                csparc_login=Path("/cluster/csparc_login.ini"),
                project_id="P1",
                workspace_id="W2",
                job_id="J3",
                job_type="curate",
                model_weights_path=Path("/cluster/sam_vit_l_0b3195.pth"),
                filter_output_dir=root / "filtered",
            )

            script = Path(summary["run_script"]).read_text()

        self.assertIn("/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker", script)
        self.assertIn("J70_particles_exported.cs", script)
        self.assertIn("draco_pretrained_fuller_subset", script)


if __name__ == "__main__":
    unittest.main()
