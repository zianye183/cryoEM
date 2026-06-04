import unittest
from pathlib import Path

from segmentation_integration.dataset_presets import get_dataset_preset, resolve_particle_cs_path, resolve_standard_paths


class DatasetPresetTests(unittest.TestCase):
    def test_raw_preset_records_standard_hpc_paths(self) -> None:
        preset = get_dataset_preset("raw")

        self.assertEqual(
            preset.particle_export_dir,
            Path("/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J71_particles"),
        )
        self.assertEqual(
            preset.micrographs_dir,
            Path("/nfs/bartesaghilab2/zy223/denoising/data/fuller_subset"),
        )
        self.assertEqual(
            preset.vesicle_picker_package_dir,
            Path("/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker/vesicle_picker"),
        )
        self.assertEqual(
            preset.vesicle_picker_root_dir,
            Path("/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker"),
        )

    def test_draco_preset_records_standard_hpc_paths(self) -> None:
        preset = get_dataset_preset("draco")

        self.assertEqual(
            preset.particle_export_dir,
            Path("/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J70_particles"),
        )
        self.assertEqual(
            preset.micrographs_dir,
            Path("/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained_fuller_subset"),
        )

    def test_resolve_standard_paths_finds_particles_and_csg_in_export_dir(self) -> None:
        with self.subTest("unique exported names"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                export_dir = root / "exports" / "groups" / "J71_particles"
                export_dir.mkdir(parents=True)
                particles_cs = export_dir / "J71_particles_exported.cs"
                particles_csg = export_dir / "J71_particles_exported.csg"
                particles_cs.write_bytes(b"")
                particles_csg.write_text("{}")

                paths = resolve_standard_paths("raw", particle_export_dir=export_dir)

                self.assertEqual(paths.particles_cs, particles_cs)
                self.assertEqual(paths.particles_csg, particles_csg)

    def test_resolve_particle_cs_path_does_not_require_csg(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp)
            particles_cs = export_dir / "J71_particles_exported.cs"
            particles_cs.write_bytes(b"")

            resolved = resolve_particle_cs_path("raw", particle_export_dir=export_dir)

        self.assertEqual(resolved, particles_cs)


if __name__ == "__main__":
    unittest.main()
