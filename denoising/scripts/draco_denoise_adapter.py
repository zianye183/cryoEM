#!/usr/bin/env python3
"""Batch DRACO Denoise/B inference for MRC micrographs.

This adapter is based on the official DRACO denoising notebook:
https://github.com/Cellverse/draco/blob/master/notebook/draco_denoise.ipynb

It intentionally lives outside the cloned DRACO repo so project inputs, outputs,
and checkpoint paths are reproducible from this workspace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

DENOISE_CONFIG_RELATIVE = Path("notebook") / "denoise.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input MRC files. Directories should be expanded by run_draco_pretrained.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where denoised MRC files will be written.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the DRACO Denoise/B checkpoint.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Local clone of https://github.com/Cellverse/draco.",
    )
    parser.add_argument(
        "--config",
        help="Optional DRACO denoise YAML. Defaults to <repo>/notebook/denoise.yaml.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--suffix",
        default="_draco_denoised",
        help="Suffix added before .mrc in output filenames.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help=(
            "Run inference in square tiles of this size to avoid GPU OOM. "
            "The official DRACO demo uses a 1024x1024 crop. Use 0 to run the full image at once."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Pixel overlap between adjacent tiles. Must be smaller than --tile-size.",
    )
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help=(
            "Save DRACO output in normalized preprocessing units. By default the output "
            "is approximately returned to the clipped input intensity scale using the "
            "per-micrograph mean/std from DRACO preprocessing."
        ),
    )
    return parser.parse_args()


def import_runtime(repo: Path) -> dict[str, Any]:
    """Import DRACO and scientific dependencies after adding repo to sys.path."""
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "notebook"))

    try:
        import mrcfile  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from draco.configuration import CfgNode  # type: ignore[import-not-found]
        from draco.model import (  # type: ignore[import-not-found]
            build_model,
            load_pretrained,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing DRACO runtime dependency. Activate the DRACO environment first; "
            "see README.md or scripts/setup_draco.py --print-install-commands.\n"
            f"Original import error: {exc}"
        ) from exc

    try:
        from process import preprocess  # type: ignore[import-not-found]
    except ImportError:
        preprocess = fallback_preprocess

    return {
        "mrcfile": mrcfile,
        "torch": torch,
        "CfgNode": CfgNode,
        "build_model": build_model,
        "load_pretrained": load_pretrained,
        "preprocess": preprocess,
    }


def contrast_normalization(
    arr_bin: np.ndarray, tile_size: int = 128, extend: float = 1.5
) -> tuple[float, float]:
    """Fallback copy of DRACO notebook/process.py contrast normalization."""
    ny, nx = arr_bin.shape
    tile_start_x = np.arange(0, nx, tile_size)
    tile_end_x = tile_start_x + tile_size
    tile_start_y = np.arange(0, ny, tile_size)
    tile_end_y = tile_start_y + tile_size
    tile_all_data = np.empty(
        (len(tile_start_y) * len(tile_start_x), 2), dtype=np.float32
    )

    index = 0
    for y in range(len(tile_start_y)):
        for x in range(len(tile_start_x)):
            arr_tile = arr_bin[
                tile_start_y[y] : tile_end_y[y], tile_start_x[x] : tile_end_x[x]
            ]
            tile_all_data[index, 0] = np.percentile(arr_tile, 98)
            tile_all_data[index, 1] = np.percentile(arr_tile, 2)
            index += 1

    high = np.nanmedian(tile_all_data[:, 0])
    low = np.nanmedian(tile_all_data[:, 1])
    vmid = 0.5 * (low + high)
    vrange = abs(low - high)
    return vmid - extend * 0.5 * vrange, vmid + extend * 0.5 * vrange


def fallback_preprocess(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    value_min, value_max = contrast_normalization(image)
    clipped = np.clip(image, value_min, value_max)
    mean = float(clipped.mean())
    std = float(clipped.std())
    if std == 0:
        std = 1.0
    return clipped, mean, std


def select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but torch.cuda.is_available() is false"
        )
    return torch.device(requested)


def load_denoiser(
    runtime: dict[str, Any], config_path: Path, weights: Path, device: Any
) -> tuple[Any, int]:
    torch = runtime["torch"]
    cfg_node = runtime["CfgNode"]
    cfg = cfg_node.load_yaml_with_base(config_path)
    cfg_node.merge_with_dotlist(cfg, [])

    model = runtime["build_model"](cfg).to(device).eval()
    model = runtime["load_pretrained"](model, weights, device)
    patch_size = int(cfg.MODEL.PATCH_SIZE)

    print(f"Loaded DRACO config: {config_path}")
    print(f"Loaded DRACO weights: {weights}")
    print(f"Device: {device}; patch size: {patch_size}")
    torch.set_grad_enabled(False)
    return model, patch_size


def pad_to_multiple(torch: Any, image: Any, multiple: int) -> tuple[Any, int, int]:
    """Pad BCHW tensor height/width up to a multiple required by DRACO patches."""
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h), mode="constant", value=0
        )
    return image, height, width


def unpatchify(patches: Any, height: int, width: int, patch_size: int) -> Any:
    batch = patches.shape[0]
    grid_h = (height + patch_size - 1) // patch_size
    grid_w = (width + patch_size - 1) // patch_size
    images = patches.reshape(batch, grid_h, grid_w, patch_size, patch_size, -1)
    images = images.permute(0, 5, 1, 3, 2, 4)
    images = images.reshape(batch, -1, grid_h * patch_size, grid_w * patch_size)
    return images[..., :height, :width]


def normalize_2d_mrc(data: np.ndarray, input_path: Path) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        return data.astype(np.float32, copy=True)
    if data.ndim == 3 and data.shape[0] == 1:
        return data[0].astype(np.float32, copy=True)
    raise ValueError(
        f"DRACO denoise adapter expects one 2D micrograph per MRC; got shape {data.shape} in {input_path}. "
        "If this is a frame stack, first create summed/even/odd micrographs."
    )


def run_model_on_array(
    *,
    runtime: dict[str, Any],
    model: Any,
    patch_size: int,
    image: np.ndarray,
    device: Any,
) -> np.ndarray:
    """Run DRACO on one 2D normalized array and return a 2D normalized array."""
    torch = runtime["torch"]
    height, width = image.shape
    x = (
        torch.from_numpy(image)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    x, padded_height, padded_width = pad_to_multiple(torch, x, patch_size)

    with torch.inference_mode():
        y = model(x)
        if y.ndim == 3:
            denoised = unpatchify(y, padded_height, padded_width, patch_size)
        elif y.ndim == 4:
            denoised = y[..., :padded_height, :padded_width]
        else:
            raise RuntimeError(f"Unexpected DRACO model output shape: {tuple(y.shape)}")

    return (
        denoised.squeeze(0)
        .squeeze(0)[..., :height, :width]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def denoise_tiled(
    *,
    runtime: dict[str, Any],
    model: Any,
    patch_size: int,
    image: np.ndarray,
    device: Any,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    """Denoise a large 2D normalized image by stitching independently denoised tiles."""
    if tile_size <= 0:
        return run_model_on_array(
            runtime=runtime,
            model=model,
            patch_size=patch_size,
            image=image,
            device=device,
        )
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("--tile-overlap must be >= 0 and smaller than --tile-size")

    height, width = image.shape
    step = tile_size - tile_overlap
    y_starts = tile_starts(height, tile_size, step)
    x_starts = tile_starts(width, tile_size, step)
    output = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    print(
        f"Tiled inference: image={height}x{width}, tile_size={tile_size}, "
        f"overlap={tile_overlap}, tiles={len(y_starts) * len(x_starts)}"
    )
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = image[y0:y1, x0:x1]
            denoised_tile = run_model_on_array(
                runtime=runtime,
                model=model,
                patch_size=patch_size,
                image=tile,
                device=device,
            )
            output[y0:y1, x0:x1] += denoised_tile[: y1 - y0, : x1 - x0]
            weights[y0:y1, x0:x1] += 1.0

    return output / np.maximum(weights, 1.0)


def denoise_one(
    *,
    runtime: dict[str, Any],
    model: Any,
    patch_size: int,
    input_path: Path,
    output_path: Path,
    device: Any,
    save_normalized: bool,
    tile_size: int,
    tile_overlap: int,
) -> None:
    mrcfile = runtime["mrcfile"]
    preprocess = runtime["preprocess"]

    with mrcfile.open(input_path, permissive=True) as src:
        raw = normalize_2d_mrc(src.data, input_path)
        voxel_size = src.voxel_size.copy()

    processed, mean, std = preprocess(raw)
    normalized = (processed.astype(np.float32) - float(mean)) / float(std)
    denoised_np = denoise_tiled(
        runtime=runtime,
        model=model,
        patch_size=patch_size,
        image=normalized,
        device=device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    if not save_normalized:
        denoised_np = denoised_np * float(std) + float(mean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(output_path, overwrite=True) as dst:
        dst.set_data(np.ascontiguousarray(denoised_np.astype(np.float32)))
        dst.voxel_size = voxel_size
        dst.update_header_from_data()
        dst.update_header_stats()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else repo / DENOISE_CONFIG_RELATIVE
    )

    if not repo.is_dir():
        raise SystemExit(f"DRACO repo does not exist: {repo}")
    if not weights.is_file():
        raise SystemExit(f"DRACO Denoise/B checkpoint does not exist: {weights}")
    if not config_path.is_file():
        raise SystemExit(f"DRACO denoise config does not exist: {config_path}")

    runtime = import_runtime(repo)
    device = select_device(runtime["torch"], args.device)
    model, patch_size = load_denoiser(runtime, config_path, weights, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    for raw_input in args.inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"Input MRC does not exist: {input_path}")
        output_path = output_dir / f"{input_path.stem}{args.suffix}.mrc"
        print(f"Denoising {input_path} -> {output_path}")
        denoise_one(
            runtime=runtime,
            model=model,
            patch_size=patch_size,
            input_path=input_path,
            output_path=output_path,
            device=device,
            save_normalized=args.save_normalized,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )


if __name__ == "__main__":
    main()
