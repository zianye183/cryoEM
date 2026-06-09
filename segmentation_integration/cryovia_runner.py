"""Wrapper for running CryoVIA pretrained segmentation models."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CRYOVIA_MODEL = "Default_thin"
DEFAULT_CLUSTER_CRYOVIA_HOME = Path("/nfs/bartesaghilab2/zy223/cryovia_runtime_home")
PACKAGED_CRYOVIA_MODELS = {"Default", "Default_thin", "Default_tomogram_slices"}


@dataclass(frozen=True)
class CryoviaSegmentationResult:
    micrographs_seen: int
    micrographs_processed: int
    label_files: list[Path]
    model: str


def resolve_cryovia_model(model: str | Path) -> str | Path:
    """Resolve a packaged CryoVIA model name or explicit model directory."""
    if isinstance(model, Path):
        return model
    text = str(model)
    if text in PACKAGED_CRYOVIA_MODELS:
        return text
    path = Path(text).expanduser()
    if path.exists():
        return path
    choices = ", ".join(sorted(PACKAGED_CRYOVIA_MODELS))
    raise ValueError(f"Unknown CryoVIA model {text!r}; expected one of {choices} or a model directory")


def run_cryovia_segmentation(
    *,
    micrograph_paths: list[Path],
    output_dir: Path,
    cryovia_model: str | Path = DEFAULT_CRYOVIA_MODEL,
    cryovia_home: Path | None = None,
    pixel_size_a: float = 7.0,
    gpus: str = "",
    njobs: int = 2,
    skip_existing: bool = False,
) -> CryoviaSegmentationResult:
    """Run CryoVIA segmentation and return written label paths.

    CryoVIA, GridEdgeDetector, TensorFlow, and their GUI dependencies are imported
    lazily so this repository remains testable without that runtime installed.
    """
    selected_model = resolve_cryovia_model(cryovia_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        Path(path)
        for path in micrograph_paths
        if not skip_existing or not (output_dir / f"{Path(path).stem}_labels.npz").exists()
    ]
    if not pending:
        return CryoviaSegmentationResult(
            micrographs_seen=len(micrograph_paths),
            micrographs_processed=0,
            label_files=sorted(output_dir.glob("*_labels.npz")),
            model=str(selected_model),
        )

    configure_cryovia_home(cryovia_home)
    model = _load_segmentation_model(selected_model)
    kwargs = _cryovia_segmentation_kwargs(model_name=str(selected_model))
    gpu_list = _parse_gpu_list(gpus)
    results = model.predict_multiprocessing(
        pending,
        [pixel_size_a] * len(pending),
        gpu=gpu_list,
        kwargs=kwargs,
        njobs=njobs,
        dataset_name="CryoVIA visual benchmark",
        seg_path=output_dir,
    )
    label_files = [Path(path) for path in results.values()] if isinstance(results, dict) else sorted(output_dir.glob("*_labels.npz"))
    if not label_files:
        label_files = sorted(output_dir.glob("*_labels.npz"))
    return CryoviaSegmentationResult(
        micrographs_seen=len(micrograph_paths),
        micrographs_processed=len(label_files),
        label_files=label_files,
        model=str(selected_model),
    )


def default_cryovia_home() -> Path:
    """Return a writable home directory for CryoVIA runtime files."""
    configured = os.environ.get("CRYOVIA_HOME")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_CLUSTER_CRYOVIA_HOME.parent.exists():
        return DEFAULT_CLUSTER_CRYOVIA_HOME
    return Path.home()


def configure_cryovia_home(cryovia_home: Path | None = None) -> Path:
    """Set the process HOME before CryoVIA initializes its ~/.cryovia path."""
    resolved = Path(cryovia_home).expanduser() if cryovia_home is not None else default_cryovia_home()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(resolved)
    return resolved


def _load_segmentation_model(model: str | Path):
    import cryovia
    from cryovia.gui.path_variables import CRYOVIA_PATH
    from cryovia.gui.segmentation_files.segmentation_model import segmentationModelFactory

    _ensure_default_segmentation_models(cryovia_package_dir=Path(cryovia.__file__).parent, cryovia_home=CRYOVIA_PATH)
    model_dir = _runtime_model_directory(model=model, cryovia_home=CRYOVIA_PATH)
    pickle_path = model_dir / "Segmentator.pickle"
    weights_path = model_dir / "best.weights.h5"
    legacy_weights_path = model_dir / "best_weights.h5"
    if not pickle_path.exists():
        raise FileNotFoundError(f"CryoVIA model pickle not found: {pickle_path}")
    loaded = segmentationModelFactory(filepath=pickle_path)
    loaded.config.name = model_dir.name
    _prepare_cryovia_weights(model=loaded, weights_path=weights_path, legacy_weights_path=legacy_weights_path)
    loaded.print = print
    return loaded


def _prepare_cryovia_weights(*, model, weights_path: Path, legacy_weights_path: Path) -> None:
    if _has_keras_v3_vars_group(weights_path):
        return
    if legacy_weights_path.exists():
        built_model = model.build_model()
        built_model.load_weights(legacy_weights_path)
        built_model.save_weights(weights_path)
        return
    if weights_path.exists():
        return
    raise FileNotFoundError(f"CryoVIA model weights not found: {weights_path}")


def _has_keras_v3_vars_group(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return "vars" in handle
    except OSError:
        return False


def _runtime_model_directory(*, model: str | Path, cryovia_home: Path) -> Path:
    model_root = cryovia_home / "SegmentationModels"
    if not isinstance(model, Path):
        return model_root / model
    model = model.expanduser().resolve()
    runtime_model = model_root / model.name
    if model == runtime_model.resolve():
        return runtime_model
    if not runtime_model.exists():
        shutil.copytree(model, runtime_model)
    return runtime_model


def _ensure_default_segmentation_models(*, cryovia_package_dir: Path, cryovia_home: Path) -> None:
    model_root = cryovia_home / "SegmentationModels"
    model_root.mkdir(parents=True, exist_ok=True)
    for model_name in sorted(PACKAGED_CRYOVIA_MODELS):
        target = model_root / model_name
        if target.exists():
            continue
        source = cryovia_package_dir / "default_models" / model_name
        pickle_path = source / "Segmentator.pickle"
        weights_path = source / "best_weights.h5"
        if not pickle_path.exists():
            raise FileNotFoundError(f"Packaged CryoVIA model pickle not found: {pickle_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Packaged CryoVIA model weights not found: {weights_path}")
        shutil.copytree(source, target)


def _cryovia_segmentation_kwargs(*, model_name: str) -> dict:
    return {
        "run": {"maskGrid": False},
        "general": {
            "use_only_closed": True,
            "rerun": False,
            "step_size": 13,
            "min_size": 50,
            "only_run_for_new_data": False,
            "estimate_middle_plane": True,
            "dark_mode": True,
        },
        "segmentation": {
            "rerun_segmentation": True,
            "identify_instances": True,
            "segmentation_model": model_name,
            "max_batch_size": 0,
            "only_segmentation": True,
            "combine_snippets": False,
            "max_nodes": 30,
        },
        "maskGrid": {},
    }


def _parse_gpu_list(gpus: str) -> list[str]:
    if not gpus:
        return ["/CPU:0"]
    devices = []
    for item in gpus.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("/"):
            devices.append(item)
        elif item.upper().startswith(("GPU:", "CPU:")):
            devices.append(f"/{item.upper()}")
        else:
            devices.append(f"/GPU:{int(item)}")
    return devices
