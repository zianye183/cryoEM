"""Standard Prestin dataset paths for vesicle-mask filtering on the HPC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


VESICLE_PICKER_PACKAGE_DIR = Path("/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker/vesicle_picker")
VESICLE_PICKER_ROOT_DIR = VESICLE_PICKER_PACKAGE_DIR.parent


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    particle_export_dir: Path
    micrographs_dir: Path
    vesicle_picker_package_dir: Path = VESICLE_PICKER_PACKAGE_DIR
    vesicle_picker_root_dir: Path = VESICLE_PICKER_ROOT_DIR


@dataclass(frozen=True)
class ResolvedDatasetPaths:
    preset: DatasetPreset
    particles_cs: Path
    particles_csg: Path
    micrographs_dir: Path
    vesicle_picker_dir: Path


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "raw": DatasetPreset(
        name="raw",
        particle_export_dir=Path("/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J71_particles"),
        micrographs_dir=Path("/nfs/bartesaghilab2/zy223/denoising/data/fuller_subset"),
    ),
    "draco": DatasetPreset(
        name="draco",
        particle_export_dir=Path("/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J70_particles"),
        micrographs_dir=Path("/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained_fuller_subset"),
    ),
}


def dataset_choices() -> tuple[str, ...]:
    return tuple(DATASET_PRESETS)


def get_dataset_preset(name: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(dataset_choices())
        raise ValueError(f"Unknown dataset preset {name!r}; expected one of: {choices}") from exc


def resolve_standard_paths(
    dataset: str,
    *,
    particle_export_dir: Path | None = None,
    particles_cs: Path | None = None,
    particles_csg: Path | None = None,
    micrographs_dir: Path | None = None,
    vesicle_picker_dir: Path | None = None,
) -> ResolvedDatasetPaths:
    preset = get_dataset_preset(dataset)
    if particle_export_dir is not None:
        preset = replace(preset, particle_export_dir=Path(particle_export_dir))
    resolved_particles_cs = Path(particles_cs) if particles_cs is not None else find_unique_file(preset.particle_export_dir, "*.cs")
    resolved_particles_csg = Path(particles_csg) if particles_csg is not None else find_unique_file(preset.particle_export_dir, "*.csg")
    return ResolvedDatasetPaths(
        preset=preset,
        particles_cs=resolved_particles_cs,
        particles_csg=resolved_particles_csg,
        micrographs_dir=Path(micrographs_dir) if micrographs_dir is not None else preset.micrographs_dir,
        vesicle_picker_dir=Path(vesicle_picker_dir) if vesicle_picker_dir is not None else preset.vesicle_picker_root_dir,
    )


def resolve_particle_cs_path(
    dataset: str,
    *,
    particle_export_dir: Path | None = None,
    particles_cs: Path | None = None,
) -> Path:
    preset = get_dataset_preset(dataset)
    if particles_cs is not None:
        return Path(particles_cs)
    directory = Path(particle_export_dir) if particle_export_dir is not None else preset.particle_export_dir
    return find_unique_file(directory, "*.cs")


def find_unique_file(directory: Path, pattern: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {pattern} file found in {directory}")
    match_list = "\n".join(str(match) for match in matches)
    raise ValueError(f"Expected exactly one {pattern} file in {directory}, found {len(matches)}:\n{match_list}")
