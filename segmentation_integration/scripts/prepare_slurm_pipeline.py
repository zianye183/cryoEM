#!/usr/bin/env python3
"""Write Slurm array scripts for DRACO -> local SAM -> filtering."""

from __future__ import annotations

import argparse
import math
import shlex
from pathlib import Path
from collections.abc import Sequence


DEFAULT_PARTITION_GPU = "bartesaghilab-a6000"
DEFAULT_PARTITION_CPU = "bartesaghilab-cpu"
DEFAULT_DRACO_VENV = Path("/nfs/bartesaghilab2/zy223/venvs/draco311")
DEFAULT_VESICLE_ENV = Path("/nfs/bartesaghilab2/mm1229/conda-envs/vesicle")
DEFAULT_DRACO_REPO = Path("/nfs/bartesaghilab2/zy223/denoising/external/draco")
DEFAULT_DRACO_WEIGHTS = Path("/nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt")
DEFAULT_PROJECT_ROOT = Path("/nfs/bartesaghilab2/zy223")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", "draco", "sam", "filter"],
        default=["all"],
        help="Which scripts to write. Use 'draco' to prepare denoising without particle locations.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--micrograph-count", type=int, required=True)
    parser.add_argument("--denoised-dir", type=Path, required=True)
    parser.add_argument("--segmentation-run-dir", type=Path)
    parser.add_argument("--filtered-output-dir", type=Path)
    parser.add_argument("--particle-export-dir", type=Path, nargs="+")
    parser.add_argument("--particles-cs", type=Path, nargs="+")
    parser.add_argument("--particles-csg", type=Path, nargs="+")
    parser.add_argument("--micrographs-dir", type=Path)
    parser.add_argument("--vesicle-picker-dir", type=Path)
    parser.add_argument("--draco-repo", type=Path, default=DEFAULT_DRACO_REPO)
    parser.add_argument("--draco-weights", type=Path, default=DEFAULT_DRACO_WEIGHTS)
    parser.add_argument("--draco-venv", type=Path, default=DEFAULT_DRACO_VENV)
    parser.add_argument("--vesicle-env", type=Path, default=DEFAULT_VESICLE_ENV)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--draco-chunk-size", type=int, default=10)
    parser.add_argument("--sam-chunk-size", type=int, default=5)
    parser.add_argument("--max-parallel-gpus", type=int, default=4)
    parser.add_argument("--gpu-partition", default=DEFAULT_PARTITION_GPU)
    parser.add_argument("--cpu-partition", default=DEFAULT_PARTITION_CPU)
    parser.add_argument("--gpu-cpus", type=int, default=8)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--cpu-cpus", type=int, default=16)
    parser.add_argument("--filter-jobs", type=int, help="Worker process count for the filter stage. Defaults to --cpu-cpus.")
    parser.add_argument("--cpu-mem", default="64G")
    parser.add_argument("--time", default="12:00:00")
    args = parser.parse_args()
    args.stages = normalize_stages(args.stages)
    validate_stage_args(args, parser)
    return args


def normalize_stages(stages: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(stages)
    if "all" in selected:
        return ("draco", "sam", "filter")
    return selected


def validate_stage_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    missing = []
    if "sam" in args.stages:
        for name in ("segmentation_run_dir", "vesicle_picker_dir"):
            if getattr(args, name) is None:
                missing.append("--" + name.replace("_", "-"))
    if "filter" in args.stages:
        for name in ("segmentation_run_dir", "filtered_output_dir", "micrographs_dir"):
            if getattr(args, name) is None:
                missing.append("--" + name.replace("_", "-"))
        if args.particle_export_dir is None and (args.particles_cs is None or args.particles_csg is None):
            missing.append("--particle-export-dir or both --particles-cs and --particles-csg")
        if args.particle_export_dir is not None and (args.particles_cs is not None or args.particles_csg is not None):
            parser.error("--particle-export-dir cannot be combined with --particles-cs or --particles-csg")
        if args.particles_cs is not None and args.particles_csg is not None and len(args.particles_cs) != len(args.particles_csg):
            parser.error("--particles-cs and --particles-csg must provide the same number of paths")
    if missing:
        parser.error("missing arguments for selected stages: " + ", ".join(sorted(set(missing))))


def quote(value: object) -> str:
    return shlex.quote(str(value))


def array_spec(*, item_count: int, chunk_size: int, max_parallel: int) -> str:
    if item_count <= 0:
        raise ValueError("item_count must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if max_parallel <= 0:
        raise ValueError("max_parallel must be > 0")
    tasks = math.ceil(item_count / chunk_size)
    return f"0-{tasks - 1}%{max_parallel}"


def write_slurm_scripts(
    *,
    output_dir: Path,
    manifest: Path,
    micrograph_count: int,
    denoised_dir: Path,
    segmentation_run_dir: Path | None = None,
    filtered_output_dir: Path | None = None,
    particle_export_dirs: Path | Sequence[Path] | None = None,
    particles_cs: Path | Sequence[Path] | None = None,
    particles_csg: Path | Sequence[Path] | None = None,
    micrographs_dir: Path | None = None,
    vesicle_picker_dir: Path | None = None,
    stages: Sequence[str] = ("draco", "sam", "filter"),
    draco_repo: Path = DEFAULT_DRACO_REPO,
    draco_weights: Path = DEFAULT_DRACO_WEIGHTS,
    draco_venv: Path = DEFAULT_DRACO_VENV,
    vesicle_env: Path = DEFAULT_VESICLE_ENV,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    draco_chunk_size: int = 10,
    sam_chunk_size: int = 5,
    max_parallel_gpus: int = 4,
    gpu_partition: str = DEFAULT_PARTITION_GPU,
    cpu_partition: str = DEFAULT_PARTITION_CPU,
    gpu_cpus: int = 8,
    gpu_mem: str = "64G",
    cpu_cpus: int = 16,
    filter_jobs: int | None = None,
    cpu_mem: str = "64G",
    time: str = "12:00:00",
) -> dict[str, Path]:
    """Write Slurm scripts and return their paths by stage name."""
    selected_stages = normalize_stages(stages)
    particle_export_dirs_seq = normalize_path_sequence(particle_export_dirs)
    particles_cs_seq = normalize_path_sequence(particles_cs)
    particles_csg_seq = normalize_path_sequence(particles_csg)
    resolved_filter_jobs = filter_jobs if filter_jobs is not None else cpu_cpus
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    draco_array = array_spec(
        item_count=micrograph_count,
        chunk_size=draco_chunk_size,
        max_parallel=max_parallel_gpus,
    )
    scripts: dict[str, Path] = {}
    if "draco" in selected_stages:
        scripts["draco_array"] = output_dir / "01_draco_array.sbatch"
        scripts["draco_array"].write_text(
            render_draco_array_script(
                partition=gpu_partition,
                cpus=gpu_cpus,
                mem=gpu_mem,
                time=time,
                array=draco_array,
                logs_dir=logs_dir,
                project_root=project_root,
                draco_venv=draco_venv,
                manifest=manifest,
                denoised_dir=denoised_dir,
                draco_repo=draco_repo,
                draco_weights=draco_weights,
                chunk_size=draco_chunk_size,
            )
        )
    if "sam" in selected_stages:
        require_paths(
            segmentation_run_dir=segmentation_run_dir,
            vesicle_picker_dir=vesicle_picker_dir,
        )
        sam_array = array_spec(
            item_count=micrograph_count,
            chunk_size=sam_chunk_size,
            max_parallel=max_parallel_gpus,
        )
        scripts["sam_array"] = output_dir / "02_sam_array.sbatch"
        scripts["sam_array"].write_text(
            render_sam_array_script(
                partition=gpu_partition,
                cpus=gpu_cpus,
                mem=gpu_mem,
                time=time,
                array=sam_array,
                logs_dir=logs_dir,
                project_root=project_root,
                vesicle_env=vesicle_env,
                vesicle_picker_dir=vesicle_picker_dir,
                manifest=manifest,
                micrographs_dir=micrographs_dir,
                particles_cs=particles_cs,
                find_parameters=segmentation_run_dir / "parameters" / "find_vesicles.ini",
                filter_parameters=segmentation_run_dir / "parameters" / "filter_vesicles.ini",
                chunk_size=sam_chunk_size,
            )
        )
    if "filter" in selected_stages:
        require_paths(
            segmentation_run_dir=segmentation_run_dir,
            filtered_output_dir=filtered_output_dir,
            micrographs_dir=micrographs_dir,
        )
        if particle_export_dirs_seq is None and (particles_cs_seq is None or particles_csg_seq is None):
            raise ValueError("Missing required particle sources: particle_export_dirs or particles_cs/particles_csg")
        if particle_export_dirs_seq is not None and (particles_cs_seq is not None or particles_csg_seq is not None):
            raise ValueError("particle_export_dirs cannot be combined with particles_cs or particles_csg")
        if particles_cs_seq is not None and particles_csg_seq is not None and len(particles_cs_seq) != len(particles_csg_seq):
            raise ValueError("particles_cs and particles_csg must have the same length")
        scripts["filter_merge"] = output_dir / "03_filter_merge.sbatch"
        scripts["filter_merge"].write_text(
            render_filter_merge_script(
                partition=cpu_partition,
                cpus=cpu_cpus,
                mem=cpu_mem,
                time=time,
                logs_dir=logs_dir,
                project_root=project_root,
                particle_export_dirs=particle_export_dirs_seq,
                particles_cs=particles_cs_seq,
                particles_csg=particles_csg_seq,
                vesicle_mask_dir=segmentation_run_dir / "vesicle_masks_filtered",
                micrographs_dir=micrographs_dir,
                filtered_output_dir=filtered_output_dir,
                jobs=resolved_filter_jobs,
            )
        )
    for path in scripts.values():
        path.chmod(0o755)
    return scripts


def require_paths(**paths: Path | None) -> None:
    missing = [name for name, value in paths.items() if value is None]
    if missing:
        raise ValueError("Missing required paths: " + ", ".join(sorted(missing)))


def normalize_path_sequence(paths: Path | Sequence[Path] | None) -> tuple[Path, ...] | None:
    if paths is None:
        return None
    if isinstance(paths, Path):
        return (paths,)
    return tuple(paths)


def render_header(
    *,
    job_name: str,
    partition: str,
    cpus: int,
    mem: str,
    time: str,
    logs_dir: Path,
    array: str | None = None,
    gres_gpu: bool = False,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -c {cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={logs_dir}/{job_name}_%A_%a.out" if array else f"#SBATCH --output={logs_dir}/{job_name}_%j.out",
        f"#SBATCH --error={logs_dir}/{job_name}_%A_%a.err" if array else f"#SBATCH --error={logs_dir}/{job_name}_%j.err",
    ]
    if gres_gpu:
        lines.append("#SBATCH --gres=gpu:1")
    if array:
        lines.append(f"#SBATCH --array={array}")
    return "\n".join(lines) + "\n\n"


def render_env_exports() -> str:
    return """set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"""


def render_conda_activate(prefix: Path) -> str:
    return f"""source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {quote(prefix)}

"""


def render_draco_array_script(
    *,
    partition: str,
    cpus: int,
    mem: str,
    time: str,
    array: str,
    logs_dir: Path,
    project_root: Path,
    draco_venv: Path,
    manifest: Path,
    denoised_dir: Path,
    draco_repo: Path,
    draco_weights: Path,
    chunk_size: int,
) -> str:
    return (
        render_header(
            job_name="draco_array",
            partition=partition,
            cpus=cpus,
            mem=mem,
            time=time,
            logs_dir=logs_dir,
            array=array,
            gres_gpu=True,
        )
        + render_env_exports()
        + f"""source {quote(draco_venv / "bin" / "activate")}

python {quote(project_root / "denoising" / "scripts" / "run_draco_pretrained.py")} \\
  --manifest {quote(manifest)} \\
  --output-dir {quote(denoised_dir)} \\
  --repo {quote(draco_repo)} \\
  --weights {quote(draco_weights)} \\
  --chunk-size {chunk_size} \\
  --shard-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --skip-existing \\
  --extra-arg=--device \\
  --extra-arg=cuda
"""
    )


def render_sam_array_script(
    *,
    partition: str,
    cpus: int,
    mem: str,
    time: str,
    array: str,
    logs_dir: Path,
    project_root: Path,
    vesicle_env: Path,
    vesicle_picker_dir: Path,
    manifest: Path,
    particles_cs: Path | None,
    micrographs_dir: Path | None,
    find_parameters: Path,
    filter_parameters: Path,
    chunk_size: int,
) -> str:
    input_args = (
        f"  --manifest {quote(manifest)} \\\n"
        if particles_cs is None
        else (
            f"  --particles-cs {quote(particles_cs)} \\\n"
            f"  --micrographs-dir {quote(micrographs_dir)} \\\n"
        )
    )
    return (
        render_header(
            job_name="sam_array",
            partition=partition,
            cpus=cpus,
            mem=mem,
            time=time,
            logs_dir=logs_dir,
            array=array,
            gres_gpu=True,
        )
        + render_env_exports()
        + render_conda_activate(vesicle_env)
        + f"""python {quote(project_root / "segmentation_integration" / "scripts" / "run_local_vesicle_picker.py")} \\
  --vesicle-picker-dir {quote(vesicle_picker_dir)} \\
{input_args.rstrip()}
  --find-parameters {quote(find_parameters)} \\
  --filter-parameters {quote(filter_parameters)} \\
  --chunk-size {chunk_size} \\
  --shard-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --skip-existing \\
  --progress-every 1
"""
    )


def render_filter_merge_script(
    *,
    partition: str,
    cpus: int,
    mem: str,
    time: str,
    logs_dir: Path,
    project_root: Path,
    particle_export_dirs: Sequence[Path] | None,
    particles_cs: Sequence[Path] | None,
    particles_csg: Sequence[Path] | None,
    vesicle_mask_dir: Path,
    micrographs_dir: Path,
    filtered_output_dir: Path,
    jobs: int,
) -> str:
    if particle_export_dirs is not None:
        particle_args = f"  --particle-export-dir {quote_join(particle_export_dirs)} \\\n"
    elif particles_cs is not None and particles_csg is not None:
        particle_args = (
            f"  --particles-cs {quote_join(particles_cs)} \\\n"
            f"  --particles-csg {quote_join(particles_csg)} \\\n"
        )
    else:
        raise ValueError("Missing particle sources")
    return (
        render_header(
            job_name="filter_merge",
            partition=partition,
            cpus=cpus,
            mem=mem,
            time=time,
            logs_dir=logs_dir,
        )
        + render_env_exports()
        + f"""python {quote(project_root / "segmentation_integration" / "scripts" / "filter_cryosparc_picks_by_vesicle_masks.py")} \\
{particle_args.rstrip()}
  --vesicle-mask-dir {quote(vesicle_mask_dir)} \\
  --micrographs-dir {quote(micrographs_dir)} \\
  --output-dir {quote(filtered_output_dir)} \\
  --jobs {jobs} \\
  --skip-mask-pngs
"""
    )


def quote_join(values: Sequence[Path]) -> str:
    return " ".join(quote(value) for value in values)


def main() -> None:
    args = parse_args()
    scripts = write_slurm_scripts(
        output_dir=args.output_dir,
        manifest=args.manifest,
        micrograph_count=args.micrograph_count,
        denoised_dir=args.denoised_dir,
        stages=args.stages,
        segmentation_run_dir=args.segmentation_run_dir,
        filtered_output_dir=args.filtered_output_dir,
        particle_export_dirs=args.particle_export_dir,
        particles_cs=args.particles_cs,
        particles_csg=args.particles_csg,
        micrographs_dir=args.micrographs_dir,
        vesicle_picker_dir=args.vesicle_picker_dir,
        draco_repo=args.draco_repo,
        draco_weights=args.draco_weights,
        draco_venv=args.draco_venv,
        vesicle_env=args.vesicle_env,
        project_root=args.project_root,
        draco_chunk_size=args.draco_chunk_size,
        sam_chunk_size=args.sam_chunk_size,
        max_parallel_gpus=args.max_parallel_gpus,
        gpu_partition=args.gpu_partition,
        cpu_partition=args.cpu_partition,
        gpu_cpus=args.gpu_cpus,
        gpu_mem=args.gpu_mem,
        cpu_cpus=args.cpu_cpus,
        filter_jobs=args.filter_jobs,
        cpu_mem=args.cpu_mem,
        time=args.time,
    )
    for name, path in scripts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
