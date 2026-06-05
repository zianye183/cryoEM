"""Prepare Vesicle Picker parameter files and end-to-end run scripts."""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

from segmentation_integration.dataset_presets import resolve_standard_paths


DEFAULT_MODEL_WEIGHTS_PATH = Path("/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker/sam_vit_h_4b8939.pth")


PRESTIN_SETTINGS = {
    "psize": "1.068",
    "downsample": "4",
    "lowpass_mode": "bilateral",
    "d": "17",
    "sigmaColor": "71",
    "sigmaSpace": "71",
    "model_type": "vit_h",
    "device": "cuda:0",
    "points_per_side": "36",
    "points_per_batch": "16",
    "pred_iou_thresh": "0.9",
    "stability_score_thresh": "0.9",
    "crop_n_layers": "1",
    "crop_n_points_downscale_factor": "2",
    "crop_nms_thresh": "0.1",
    "min_mask_region_area": "100",
    "area_min": "1000",
    "area_max": "18000",
    "roundness_min": "0.82",
    "roundness_max": "1",
    "membrane_band_a": "40",
}


SEGMENTATION_PRESETS = {
    "default": {},
    "recall_heavy": {
        "points_per_side": "64",
        "pred_iou_thresh": "0.8",
        "stability_score_thresh": "0.8",
        "crop_n_layers": "2",
        "crop_nms_thresh": "0.5",
    },
}


def prepare_vesicle_picker_run(
    *,
    output_dir: Path,
    vesicle_picker_dir: Path,
    csparc_login: Path,
    project_id: str,
    workspace_id: str,
    job_id: str,
    job_type: str,
    model_weights_path: Path = DEFAULT_MODEL_WEIGHTS_PATH,
    particles_cs: Path,
    particles_csg: Path,
    micrographs_dir: Path,
    filter_output_dir: Path,
    local_micrographs: bool = False,
    segmentation_preset: str = "default",
) -> dict[str, str]:
    """Write Vesicle Picker configs and a runnable end-to-end shell script."""
    settings = settings_for_segmentation_preset(segmentation_preset)
    output_dir = Path(output_dir)
    parameters_dir = output_dir / "parameters"
    found_masks_dir = output_dir / "vesicle_masks_found"
    filtered_masks_dir = output_dir / "vesicle_masks_filtered"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    found_masks_dir.mkdir(parents=True, exist_ok=True)
    filtered_masks_dir.mkdir(parents=True, exist_ok=True)

    find_path = parameters_dir / "find_vesicles.ini"
    filter_path = parameters_dir / "filter_vesicles.ini"
    script_path = output_dir / "run_end_to_end.sh"

    if local_micrographs:
        find_path.write_text(
            render_local_find_vesicles_ini(
                model_weights_path=model_weights_path,
                output_directory=found_masks_dir,
                settings=settings,
            )
        )
        filter_path.write_text(
            render_local_filter_vesicles_ini(
                input_directory=found_masks_dir,
                output_directory=filtered_masks_dir,
                settings=settings,
            )
        )
    else:
        find_path.write_text(
            render_find_vesicles_ini(
                csparc_login=csparc_login,
                project_id=project_id,
                job_id=job_id,
                job_type=job_type,
                model_weights_path=model_weights_path,
                output_directory=found_masks_dir,
                settings=settings,
            )
        )
        filter_path.write_text(
            render_filter_vesicles_ini(
                csparc_login=csparc_login,
                project_id=project_id,
                job_id=job_id,
                job_type=job_type,
                input_directory=found_masks_dir,
                output_directory=filtered_masks_dir,
                settings=settings,
            )
        )
    script_path.write_text(
        render_end_to_end_script(
            vesicle_picker_dir=vesicle_picker_dir,
            find_parameters=find_path,
            filter_parameters=filter_path,
            particles_cs=particles_cs,
            particles_csg=particles_csg,
            micrographs_dir=micrographs_dir,
            vesicle_mask_dir=filtered_masks_dir,
            filter_output_dir=filter_output_dir,
            workspace_id=workspace_id or "local",
            local_micrographs=local_micrographs,
            settings=settings,
        )
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return {
        "find_parameters": str(find_path),
        "filter_parameters": str(filter_path),
        "run_script": str(script_path),
        "found_masks_dir": str(found_masks_dir),
        "filtered_masks_dir": str(filtered_masks_dir),
        "filter_output_dir": str(filter_output_dir),
        "segmentation_preset": segmentation_preset,
    }


def prepare_vesicle_picker_run_for_dataset(
    *,
    dataset: str,
    output_dir: Path,
    model_weights_path: Path = DEFAULT_MODEL_WEIGHTS_PATH,
    filter_output_dir: Path,
    csparc_login: Path | None = None,
    project_id: str | None = None,
    workspace_id: str | None = None,
    job_id: str | None = None,
    job_type: str = "curate",
    local_micrographs: bool = False,
    particle_export_dir: Path | None = None,
    particles_cs: Path | None = None,
    particles_csg: Path | None = None,
    micrographs_dir: Path | None = None,
    vesicle_picker_dir: Path | None = None,
    segmentation_preset: str = "default",
) -> dict[str, str]:
    """Write Vesicle Picker configs using a named Prestin dataset preset."""
    paths = resolve_standard_paths(
        dataset,
        particle_export_dir=particle_export_dir,
        particles_cs=particles_cs,
        particles_csg=particles_csg,
        micrographs_dir=micrographs_dir,
        vesicle_picker_dir=vesicle_picker_dir,
    )
    summary = prepare_vesicle_picker_run(
        output_dir=output_dir,
        vesicle_picker_dir=paths.vesicle_picker_dir,
        csparc_login=csparc_login or Path("csparc_login.ini"),
        project_id=project_id or "PXX",
        workspace_id=workspace_id or "WXX",
        job_id=job_id or "JXX",
        job_type=job_type,
        model_weights_path=model_weights_path,
        particles_cs=paths.particles_cs,
        particles_csg=paths.particles_csg,
        micrographs_dir=paths.micrographs_dir,
        filter_output_dir=filter_output_dir,
        local_micrographs=local_micrographs,
        segmentation_preset=segmentation_preset,
    )
    summary.update(
        {
            "dataset": dataset,
            "particles_cs": str(paths.particles_cs),
            "particles_csg": str(paths.particles_csg),
            "particle_export_dir": str(paths.preset.particle_export_dir),
            "micrographs_dir": str(paths.micrographs_dir),
            "vesicle_picker_dir": str(paths.vesicle_picker_dir),
            "local_micrographs": str(local_micrographs),
        }
    )
    return summary


def render_local_find_vesicles_ini(
    *,
    model_weights_path: Path,
    output_directory: Path,
    settings: dict[str, str] | None = None,
) -> str:
    settings = PRESTIN_SETTINGS if settings is None else settings
    return f"""[general]
psize = {settings["psize"]}
downsample = {settings["downsample"]}

[preprocessing]
lowpass_mode = {settings["lowpass_mode"]}
d = {settings["d"]}
sigmaColor = {settings["sigmaColor"]}
sigmaSpace = {settings["sigmaSpace"]}

[segmentation]
model_weights_path = {model_weights_path}
model_type = {settings["model_type"]}
device = {settings["device"]}
points_per_side = {settings["points_per_side"]}
points_per_batch = {settings["points_per_batch"]}
pred_iou_thresh = {settings["pred_iou_thresh"]}
stability_score_thresh = {settings["stability_score_thresh"]}
crop_n_layers = {settings["crop_n_layers"]}
crop_n_points_downscale_factor = {settings["crop_n_points_downscale_factor"]}
crop_nms_thresh = {settings["crop_nms_thresh"]}
min_mask_region_area = {settings["min_mask_region_area"]}

[postprocessing]
functions = [postprocess.find_mask_intensity, postprocess.find_contour, postprocess.find_roundness, postprocess.fit_ellipse]

[output]
directory = {ensure_trailing_slash(output_directory)}
"""


def render_local_filter_vesicles_ini(
    *,
    input_directory: Path,
    output_directory: Path,
    settings: dict[str, str] | None = None,
) -> str:
    settings = PRESTIN_SETTINGS if settings is None else settings
    return f"""[input]
directory = {ensure_trailing_slash(input_directory)}

[general]
psize = {settings["psize"]}
downsample = {settings["downsample"]}

[area]
min = {settings["area_min"]}
max = {settings["area_max"]}

[roundness]
min = {settings["roundness_min"]}
max = {settings["roundness_max"]}

[output]
directory = {ensure_trailing_slash(output_directory)}
"""


def render_find_vesicles_ini(
    *,
    csparc_login: Path,
    project_id: str,
    job_id: str,
    job_type: str,
    model_weights_path: Path,
    output_directory: Path,
    settings: dict[str, str] | None = None,
) -> str:
    settings = PRESTIN_SETTINGS if settings is None else settings
    return f"""[csparc_input]
PID = {project_id}
JID = {job_id}
type = {job_type}
login = {csparc_login}

[general]
psize = {settings["psize"]}
downsample = {settings["downsample"]}

[preprocessing]
lowpass_mode = {settings["lowpass_mode"]}
d = {settings["d"]}
sigmaColor = {settings["sigmaColor"]}
sigmaSpace = {settings["sigmaSpace"]}

[segmentation]
model_weights_path = {model_weights_path}
model_type = {settings["model_type"]}
device = {settings["device"]}
points_per_side = {settings["points_per_side"]}
points_per_batch = {settings["points_per_batch"]}
pred_iou_thresh = {settings["pred_iou_thresh"]}
stability_score_thresh = {settings["stability_score_thresh"]}
crop_n_layers = {settings["crop_n_layers"]}
crop_n_points_downscale_factor = {settings["crop_n_points_downscale_factor"]}
crop_nms_thresh = {settings["crop_nms_thresh"]}
min_mask_region_area = {settings["min_mask_region_area"]}

[postprocessing]
functions = [postprocess.find_mask_intensity, postprocess.find_contour, postprocess.find_roundness, postprocess.fit_ellipse]

[output]
directory = {ensure_trailing_slash(output_directory)}
"""


def render_filter_vesicles_ini(
    *,
    csparc_login: Path,
    project_id: str,
    job_id: str,
    job_type: str,
    input_directory: Path,
    output_directory: Path,
    settings: dict[str, str] | None = None,
) -> str:
    settings = PRESTIN_SETTINGS if settings is None else settings
    return f"""[csparc_input]
PID = {project_id}
JID = {job_id}
type = {job_type}
login = {csparc_login}

[input]
directory = {ensure_trailing_slash(input_directory)}

[general]
psize = {settings["psize"]}
downsample = {settings["downsample"]}

[area]
min = {settings["area_min"]}
max = {settings["area_max"]}

[roundness]
min = {settings["roundness_min"]}
max = {settings["roundness_max"]}

[output]
directory = {ensure_trailing_slash(output_directory)}
"""


def render_end_to_end_script(
    *,
    vesicle_picker_dir: Path,
    find_parameters: Path,
    filter_parameters: Path,
    particles_cs: Path,
    particles_csg: Path,
    micrographs_dir: Path,
    vesicle_mask_dir: Path,
    filter_output_dir: Path,
    workspace_id: str,
    local_micrographs: bool = False,
    settings: dict[str, str] | None = None,
) -> str:
    settings = PRESTIN_SETTINGS if settings is None else settings
    visualize_command = [
        "python",
        "segmentation_integration/scripts/visualize_vesicle_masks.py",
        "--particles-cs",
        particles_cs,
        "--vesicle-mask-dir",
        vesicle_mask_dir,
        "--output-dir",
        filter_output_dir / "masks",
        "--psize-a",
        settings["psize"],
        "--downsample",
        settings["downsample"],
        "--membrane-band-a",
        settings["membrane_band_a"],
    ]
    filter_command = [
        "python",
        "segmentation_integration/scripts/filter_cryosparc_picks_by_vesicle_masks.py",
        "--particles-cs",
        particles_cs,
        "--particles-csg",
        particles_csg,
        "--vesicle-mask-dir",
        vesicle_mask_dir,
        "--micrographs-dir",
        micrographs_dir,
        "--output-dir",
        filter_output_dir,
        "--psize-a",
        settings["psize"],
        "--downsample",
        settings["downsample"],
        "--membrane-band-a",
        settings["membrane_band_a"],
        "--skip-mask-pngs",
    ]
    upstream_qc_command = [
        "python",
        "segmentation_integration/scripts/visualize_vesicle_mask_filtering.py",
        "--found-mask-dir",
        find_parameters.parent.parent / "vesicle_masks_found",
        "--filtered-mask-dir",
        vesicle_mask_dir,
        "--output-dir",
        filter_output_dir / "mask_filter_qc",
        "--particles-cs",
        particles_cs,
        "--micrographs-dir",
        micrographs_dir,
        "--downsample",
        settings["downsample"],
    ]
    local_vesicle_command = [
        "python",
        "segmentation_integration/scripts/run_local_vesicle_picker.py",
        "--vesicle-picker-dir",
        vesicle_picker_dir,
        "--particles-cs",
        particles_cs,
        "--micrographs-dir",
        micrographs_dir,
        "--find-parameters",
        find_parameters,
        "--filter-parameters",
        filter_parameters,
        "--progress-every",
        "1",
    ]
    vesicle_picker_commands = (
        quote_command(local_vesicle_command)
        if local_micrographs
        else "\n".join(
            [
                f"python \"$VESICLE_PICKER_DIR/find_vesicles.py\" {shlex.quote(str(find_parameters))}",
                f"python \"$VESICLE_PICKER_DIR/filter_vesicles.py\" {shlex.quote(str(filter_parameters))}",
            ]
        )
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

# WID retained for traceability when comparing this run to CryoSPARC workspaces.
WID={shlex.quote(workspace_id)}
VESICLE_PICKER_DIR={shlex.quote(str(vesicle_picker_dir))}

echo "[segmentation integration] generating vesicle masks"
{vesicle_picker_commands}

echo "[segmentation integration] visualizing found vs filtered masks"
{quote_command(upstream_qc_command)}

echo "[segmentation integration] visualizing masks"
{quote_command(visualize_command)}

echo "[segmentation integration] filtering particle picks"
{quote_command(filter_command)}
"""


def ensure_trailing_slash(path: Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else text + os.sep


def quote_command(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def settings_for_segmentation_preset(segmentation_preset: str) -> dict[str, str]:
    if segmentation_preset not in SEGMENTATION_PRESETS:
        choices = ", ".join(sorted(SEGMENTATION_PRESETS))
        raise ValueError(f"Unknown segmentation preset {segmentation_preset!r}; choices: {choices}")
    settings = dict(PRESTIN_SETTINGS)
    settings.update(SEGMENTATION_PRESETS[segmentation_preset])
    return settings


def segmentation_preset_choices() -> list[str]:
    return sorted(SEGMENTATION_PRESETS)
