import tempfile
import unittest
from pathlib import Path

from segmentation_integration.scripts.prepare_slurm_pipeline import write_slurm_scripts


class SlurmPipelineTests(unittest.TestCase):
    def test_write_slurm_scripts_caps_gpu_arrays_at_four(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = write_slurm_scripts(
                output_dir=root,
                manifest=Path("/runs/micrographs.tsv"),
                micrograph_count=12000,
                denoised_dir=Path("/runs/draco"),
                segmentation_run_dir=Path("/runs/segmentation"),
                filtered_output_dir=Path("/runs/filtered_particles"),
                particles_cs=Path("/exports/particles.cs"),
                particles_csg=Path("/exports/particles.csg"),
                micrographs_dir=Path("/runs/draco"),
                vesicle_picker_dir=Path("/software/vesicle-picker"),
                draco_repo=Path("/denoising/external/draco"),
                draco_weights=Path("/denoising/models/draco.ckpt"),
                draco_venv=Path("/venvs/draco311"),
                draco_chunk_size=10,
                sam_chunk_size=5,
                max_parallel_gpus=4,
            )

            draco_script = scripts["draco_array"].read_text()
            sam_script = scripts["sam_array"].read_text()
            filter_script = scripts["filter_merge"].read_text()

        self.assertIn("#SBATCH --array=0-1199%4", draco_script)
        self.assertIn("#SBATCH --array=0-2399%4", sam_script)
        self.assertIn("#SBATCH -p bartesaghilab-a6000", draco_script)
        self.assertIn("#SBATCH --gres=gpu:1", sam_script)
        self.assertIn('source "$(conda info --base)/etc/profile.d/conda.sh"', sam_script)
        self.assertIn("conda activate /nfs/bartesaghilab2/mm1229/conda-envs/vesicle", sam_script)
        self.assertIn("--chunk-size 10", draco_script)
        self.assertIn("--chunk-size 5", sam_script)
        self.assertIn("--skip-existing", draco_script)
        self.assertIn("--skip-existing", sam_script)
        self.assertIn("filter_cryosparc_picks_by_vesicle_masks.py", filter_script)
        self.assertIn("/runs/filtered_particles", filter_script)
        self.assertIn("--jobs 16", filter_script)

    def test_write_slurm_scripts_can_generate_draco_only_without_particles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = write_slurm_scripts(
                output_dir=root,
                manifest=Path("/runs/micrographs.tsv"),
                micrograph_count=12000,
                denoised_dir=Path("/nfs/bartesaghilab2/zy223/data/prestin_draco_full"),
                stages=("draco",),
                draco_repo=Path("/denoising/external/draco"),
                draco_weights=Path("/denoising/models/draco.ckpt"),
                draco_venv=Path("/venvs/draco311"),
                draco_chunk_size=10,
                max_parallel_gpus=4,
            )

            draco_script = scripts["draco_array"].read_text()

        self.assertEqual(sorted(scripts), ["draco_array"])
        self.assertIn("#SBATCH --array=0-1199%4", draco_script)
        self.assertIn("--output-dir /nfs/bartesaghilab2/zy223/data/prestin_draco_full", draco_script)
        self.assertFalse((root / "02_sam_array.sbatch").exists())
        self.assertFalse((root / "03_filter_merge.sbatch").exists())

    def test_write_slurm_scripts_can_generate_sam_only_from_manifest_without_particles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = write_slurm_scripts(
                output_dir=root,
                manifest=Path("/runs/denoised_manifest.tsv"),
                micrograph_count=16613,
                denoised_dir=Path("/runs/draco_denoised"),
                stages=("sam",),
                segmentation_run_dir=Path("/runs/segmentation"),
                vesicle_picker_dir=Path("/software/vesicle-picker"),
                sam_chunk_size=5,
                max_parallel_gpus=4,
            )

            sam_script = scripts["sam_array"].read_text()

        self.assertEqual(sorted(scripts), ["sam_array"])
        self.assertIn("#SBATCH --array=0-3322%4", sam_script)
        self.assertIn('source "$(conda info --base)/etc/profile.d/conda.sh"', sam_script)
        self.assertIn("conda activate /nfs/bartesaghilab2/mm1229/conda-envs/vesicle", sam_script)
        self.assertIn("--manifest /runs/denoised_manifest.tsv", sam_script)
        self.assertIn("--find-parameters /runs/segmentation/parameters/find_vesicles.ini", sam_script)
        self.assertNotIn("--particles-cs", sam_script)
        self.assertFalse((root / "01_draco_array.sbatch").exists())
        self.assertFalse((root / "03_filter_merge.sbatch").exists())

    def test_write_slurm_scripts_can_filter_from_multiple_particle_export_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = write_slurm_scripts(
                output_dir=root,
                manifest=Path("/runs/micrographs.tsv"),
                micrograph_count=100,
                denoised_dir=Path("/runs/draco"),
                stages=("filter",),
                segmentation_run_dir=Path("/runs/segmentation"),
                filtered_output_dir=Path("/runs/filtered_particles"),
                particle_export_dirs=[
                    Path("/exports/groups/J101_particles"),
                    Path("/exports/groups/J102_particles"),
                ],
                micrographs_dir=Path("/runs/draco"),
            )

            filter_script = scripts["filter_merge"].read_text()

        self.assertIn("--particle-export-dir /exports/groups/J101_particles /exports/groups/J102_particles", filter_script)
        self.assertNotIn("--particles-cs", filter_script)
        self.assertIn("--output-dir /runs/filtered_particles", filter_script)
        self.assertIn("--jobs 16", filter_script)

    def test_write_slurm_scripts_can_override_filter_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = write_slurm_scripts(
                output_dir=root,
                manifest=Path("/runs/micrographs.tsv"),
                micrograph_count=100,
                denoised_dir=Path("/runs/draco"),
                stages=("filter",),
                segmentation_run_dir=Path("/runs/segmentation"),
                filtered_output_dir=Path("/runs/filtered_particles"),
                particles_cs=Path("/exports/particles.cs"),
                particles_csg=Path("/exports/particles.csg"),
                micrographs_dir=Path("/runs/draco"),
                cpu_cpus=32,
                filter_jobs=8,
            )

            filter_script = scripts["filter_merge"].read_text()

        self.assertIn("#SBATCH -c 32", filter_script)
        self.assertIn("--jobs 8", filter_script)


if __name__ == "__main__":
    unittest.main()
