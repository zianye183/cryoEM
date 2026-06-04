"""Prepare Vesicle Picker parameter files and end-to-end run scripts."""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

from segmentation_integration.dataset_presets import resolve_standard_paths


PRESTIN_SETTINGS = {
    "psize": "1.068",
    "downsample": "4",
    "lowpass_mode": "bilateral",
    "d": "17",
    "sigmaColor": "71",
    "sigmaSpace": "71",
    "model_type": "vit_l",
    "device": "cuda:0",
    "points_per_side": "36",
    "points_per_batch": "16",
    "pred_iou_thresh": "0.9",
    "stability_score_thresh": "0.9",
    "crop_n_layers": "1",
    "crop_n_points_downscale_factor": "2",
    "crop_nms_thresh": "0.1",
    "min_mask_region_area": "100",
    "area_min": "2.5e3",
    "area_max": "1e5",
    "roundness_min": "0.75",
    "roundness_max": "1",
    "membrane_band_a": "40",
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
    model_weights_path: Path,
    particles_cs: Path,
    particles_csg: Path,
    micrographs_dir: Path,
    filter_output_dir: Path,
) -> dict[str, str]:
    """Write Vesicle Picker configs and a runnable end-to-end shell script."""
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

    find_path.write_text(
        render_find_vesicles_ini(
            csparc_login=csparc_login,
            project_id=project_id,
            job_id=job_id,
            job_type=job_type,
            model_weights_path=model_weights_path,
            output_directory=found_masks_dir,
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
            workspace_id=workspace_id,
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
    }


def prepare_vesicle_picker_run_for_dataset(
    *,
    dataset: str,
    output_dir: Path,
    csparc_login: Path,
    project_id: str,
    workspace_id: str,
    job_id: str,
    job_type: str,
    model_weights_path: Path,
    filter_output_dir: Path,
    particle_export_dir: Path | None = None,
    particles_cs: Path | None = None,
    particles_csg: Path | None = None,
    micrographs_dir: Path | None = None,
    vesicle_picker_dir: Path | None = None,
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
        csparc_login=csparc_login,
        project_id=project_id,
        workspace_id=workspace_id,
        job_id=job_id,
        job_type=job_type,
        model_weights_path=model_weights_path,
        particles_cs=paths.particles_cs,
        particles_csg=paths.particles_csg,
        micrographs_dir=paths.micrographs_dir,
        filter_output_dir=filter_output_dir,
    )
    summary.update(
        {
            "dataset": dataset,
            "particles_cs": str(paths.particles_cs),
            "particles_csg": str(paths.particles_csg),
            "particle_export_dir": str(paths.preset.particle_export_dir),
            "micrographs_dir": str(paths.micrographs_dir),
            "vesicle_picker_dir": str(paths.vesicle_picker_dir),
        }
    )
    return summary


def render_find_vesicles_ini(
    *,
    csparc_login: Path,
    project_id: str,
    job_id: str,
    job_type: str,
    model_weights_path: Path,
    output_directory: Path,
) -> str:
    return f"""[csparc_input]
PID = {project_id}
JID = {job_id}
type = {job_type}
login = {csparc_login}

[general]
psize = {PRESTIN_SETTINGS["psize"]}
downsample = {PRESTIN_SETTINGS["downsample"]}

[preprocessing]
lowpass_mode = {PRESTIN_SETTINGS["lowpass_mode"]}
d = {PRESTIN_SETTINGS["d"]}
sigmaColor = {PRESTIN_SETTINGS["sigmaColor"]}
sigmaSpace = {PRESTIN_SETTINGS["sigmaSpace"]}

[segmentation]
model_weights_path = {model_weights_path}
model_type = {PRESTIN_SETTINGS["model_type"]}
device = {PRESTIN_SETTINGS["device"]}
points_per_side = {PRESTIN_SETTINGS["points_per_side"]}
points_per_batch = {PRESTIN_SETTINGS["points_per_batch"]}
pred_iou_thresh = {PRESTIN_SETTINGS["pred_iou_thresh"]}
stability_score_thresh = {PRESTIN_SETTINGS["stability_score_thresh"]}
crop_n_layers = {PRESTIN_SETTINGS["crop_n_layers"]}
crop_n_points_downscale_factor = {PRESTIN_SETTINGS["crop_n_points_downscale_factor"]}
crop_nms_thresh = {PRESTIN_SETTINGS["crop_nms_thresh"]}
min_mask_region_area = {PRESTIN_SETTINGS["min_mask_region_area"]}

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
) -> str:
    return f"""[csparc_input]
PID = {project_id}
JID = {job_id}
type = {job_type}
login = {csparc_login}

[input]
directory = {ensure_trailing_slash(input_directory)}

[general]
psize = {PRESTIN_SETTINGS["psize"]}
downsample = {PRESTIN_SETTINGS["downsample"]}

[area]
min = {PRESTIN_SETTINGS["area_min"]}
max = {PRESTIN_SETTINGS["area_max"]}

[roundness]
min = {PRESTIN_SETTINGS["roundness_min"]}
max = {PRESTIN_SETTINGS["roundness_max"]}

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
) -> str:
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
        PRESTIN_SETTINGS["psize"],
        "--downsample",
        PRESTIN_SETTINGS["downsample"],
        "--membrane-band-a",
        PRESTIN_SETTINGS["membrane_band_a"],
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
        PRESTIN_SETTINGS["psize"],
        "--downsample",
        PRESTIN_SETTINGS["downsample"],
        "--membrane-band-a",
        PRESTIN_SETTINGS["membrane_band_a"],
        "--skip-mask-pngs",
    ]
    return f"""#!/usr/bin/env bash
set -euo pipefail

# WID retained for traceability when comparing this run to CryoSPARC workspaces.
WID={shlex.quote(workspace_id)}
VESICLE_PICKER_DIR={shlex.quote(str(vesicle_picker_dir))}

python "$VESICLE_PICKER_DIR/find_vesicles.py" {shlex.quote(str(find_parameters))}
python "$VESICLE_PICKER_DIR/filter_vesicles.py" {shlex.quote(str(filter_parameters))}

{quote_command(visualize_command)}
{quote_command(filter_command)}
"""


def ensure_trailing_slash(path: Path) -> str:
    text = str(path)
    return text if text.endswith(os.sep) else text + os.sep


def quote_command(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)
