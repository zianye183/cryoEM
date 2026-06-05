import tempfile
import unittest
from pathlib import Path

from segmentation_integration.vesicle_picker_invocation import (
    DEFAULT_MODEL_WEIGHTS_PATH,
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
        self.assertIn("model_type = vit_h", find_ini)
        self.assertIn("psize = 1.068", find_ini)
        self.assertIn("[area]", filter_ini)
        self.assertIn("min = 2.5e3", filter_ini)
        self.assertIn("find_vesicles.py", script)
        self.assertIn("filter_vesicles.py", script)
        self.assertIn("[segmentation integration] generating vesicle masks", script)
        self.assertIn("[segmentation integration] visualizing found vs filtered masks", script)
        self.assertIn("visualize_vesicle_mask_filtering.py", script)
        self.assertIn("[segmentation integration] visualizing masks", script)
        self.assertIn("[segmentation integration] filtering particle picks", script)
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

    def test_prepare_local_micrograph_run_does_not_require_cryosparc_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_export_dir = root / "J71_particles"
            particle_export_dir.mkdir()
            (particle_export_dir / "J71_particles_exported.cs").write_bytes(b"")
            (particle_export_dir / "J71_particles_exported.csg").write_text("{}")

            summary = prepare_vesicle_picker_run_for_dataset(
                dataset="raw",
                output_dir=root / "run",
                particle_export_dir=particle_export_dir,
                local_micrographs=True,
                filter_output_dir=root / "filtered",
            )

            find_ini = Path(summary["find_parameters"]).read_text()
            filter_ini = Path(summary["filter_parameters"]).read_text()
            script = Path(summary["run_script"]).read_text()

        self.assertNotIn("[csparc_input]", find_ini)
        self.assertNotIn("[csparc_input]", filter_ini)
        self.assertIn("run_local_vesicle_picker.py", script)
        self.assertIn("model_type = vit_h", find_ini)
        self.assertIn(f"model_weights_path = {DEFAULT_MODEL_WEIGHTS_PATH}", find_ini)
        self.assertNotIn("find_vesicles.py", script)
        self.assertNotIn("filter_vesicles.py", script)
        self.assertLess(script.index("run_local_vesicle_picker.py"), script.index("visualize_vesicle_masks.py"))
        self.assertLess(script.index("visualize_vesicle_masks.py"), script.index("filter_cryosparc_picks_by_vesicle_masks.py"))

    def test_prepare_run_can_use_recall_heavy_segmentation_preset(self) -> None:
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
                model_weights_path=Path("/cluster/sam_vit_h_4b8939.pth"),
                particles_cs=Path("/exports/particles.cs"),
                particles_csg=Path("/exports/particles.csg"),
                micrographs_dir=Path("/exports/micrographs"),
                filter_output_dir=Path("/outputs/filtered_particles"),
                segmentation_preset="recall_heavy",
            )

            find_ini = (output_dir / "parameters" / "find_vesicles.ini").read_text()

        self.assertEqual(summary["segmentation_preset"], "recall_heavy")
        self.assertIn("points_per_side = 64", find_ini)
        self.assertIn("pred_iou_thresh = 0.8", find_ini)
        self.assertIn("stability_score_thresh = 0.8", find_ini)
        self.assertIn("crop_n_layers = 2", find_ini)
        self.assertIn("crop_nms_thresh = 0.5", find_ini)


if __name__ == "__main__":
    unittest.main()
