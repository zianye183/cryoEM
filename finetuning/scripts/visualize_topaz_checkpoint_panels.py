#!/usr/bin/env python3
"""Create side-by-side PNG panels for Topaz checkpoint comparisons.

For selected train/validation pairs, each panel has two rows (`even`/`odd`) and
one column per model: original, pretrained `unet`, and fine-tuned checkpoints.
Missing denoised outputs are generated into `visualization/denoised_cache` using
`topaz_denoise_model.py` so source micrographs remain read-only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from finetuning_common import read_tsv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Checkpoint evaluation directory containing model_outputs/.",
    )
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Visualization output directory. Default: <eval-dir>/visualization.",
    )
    parser.add_argument("--count", type=int, default=5, help="Pairs per split.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--patch-size", default="4000")
    parser.add_argument("--patch-padding", default="500")
    parser.add_argument(
        "--max-display-pixels",
        type=int,
        default=512,
        help="Downsample each micrograph view so its longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--dpi", type=int, default=160, help="PNG output DPI for rendered panels."
    )
    parser.add_argument(
        "--force-denoise",
        action="store_true",
        help="Regenerate cached denoised files even if they already exist.",
    )
    return parser.parse_args()


def require_imports() -> None:
    missing: list[str] = []
    for module in ["mrcfile", "matplotlib", "numpy"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ImportError(
            "Missing visualization dependency/dependencies: "
            + ", ".join(missing)
            + ". Install in the active Topaz env, e.g. python -m pip install "
            + " ".join(missing)
        )


def model_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "unet":
        return (0, 0, path.name)
    match = re.search(r"epoch(\d+)", path.name)
    if match:
        return (1, int(match.group(1)), path.name)
    return (2, 0, path.name)


def discover_models(eval_dir: Path) -> list[Path]:
    model_root = eval_dir / "model_outputs"
    if not model_root.is_dir():
        raise FileNotFoundError(f"Missing model output directory: {model_root}")
    models = sorted(
        [path for path in model_root.iterdir() if path.is_dir()], key=model_sort_key
    )
    if not models:
        raise ValueError(f"No model output directories found in {model_root}")
    return models


def half_name(column: str) -> str:
    if column == "even_path":
        return "denoised_even"
    if column == "odd_path":
        return "denoised_odd"
    raise ValueError(f"Unsupported half column: {column}")


def half_label(column: str) -> str:
    return "half1" if column == "even_path" else "half2"


def find_output_file(output_dir: Path, input_path: Path) -> Path | None:
    candidates = [
        output_dir / input_path.name,
        output_dir / f"{input_path.stem}.mrc",
        output_dir / f"{input_path.stem}_denoised.mrc",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(output_dir.glob(f"{input_path.stem}*.mrc"))
    return matches[0] if matches else None


def model_arg_for_slug(slug: str, eval_dir: Path) -> str:
    if slug == "unet":
        return "unet"
    checkpoint = (
        eval_dir.parent / "topaz_finetuned_pilot_run001" / "checkpoints" / f"{slug}.sav"
    )
    # More robust fallback for eval dirs not named like the pilot run.
    if not checkpoint.is_file():
        matches = sorted(eval_dir.parent.glob(f"**/checkpoints/{slug}.sav"))
        if matches:
            checkpoint = matches[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Could not infer checkpoint path for model output '{slug}'. Expected {checkpoint}"
        )
    return str(checkpoint)


def denoise_missing(
    *,
    eval_dir: Path,
    model_dir: Path,
    split: str,
    column: str,
    input_paths: list[Path],
    visualization_dir: Path,
    device: str,
    patch_size: str,
    patch_padding: str,
    force: bool,
) -> Path:
    """Return directory containing denoised outputs for requested model/split/half."""
    existing_dir = model_dir / half_name(column)
    missing_from_existing = [
        path for path in input_paths if find_output_file(existing_dir, path) is None
    ]
    if existing_dir.is_dir() and not missing_from_existing and not force:
        return existing_dir

    cache_dir = (
        visualization_dir
        / "denoised_cache"
        / model_dir.name
        / split
        / half_name(column)
    )
    missing_from_cache = [
        path
        for path in input_paths
        if force or find_output_file(cache_dir, path) is None
    ]
    if not missing_from_cache:
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).with_name("topaz_denoise_model.py")
    model_arg = model_arg_for_slug(model_dir.name, eval_dir)
    command = [
        sys.executable,
        str(wrapper),
        "--model",
        model_arg,
        "--device",
        device,
        "--patch-size",
        patch_size,
        "--patch-padding",
        patch_padding,
        "--output",
        str(cache_dir),
    ]
    command.extend(str(path) for path in missing_from_cache)

    log_dir = visualization_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"denoise_{model_dir.name}_{split}_{half_label(column)}.log"
    print(f"# current model/checkpoint: {model_arg}")
    print(
        f"denoising missing visualization outputs: split={split} half={half_label(column)} "
        f"count={len(missing_from_cache)}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return cache_dir


def load_mrc(path: Path) -> Any:
    import mrcfile
    import numpy as np

    with mrcfile.open(path, permissive=True) as mrc:
        arr = np.asarray(mrc.data, dtype=np.float32).copy()
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    return arr


def normalize_for_display(arr: Any, max_pixels: int) -> Any:
    import numpy as np

    image = np.asarray(arr, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    height, width = image.shape[-2:]
    scale = min(1.0, float(max_pixels) / float(max(height, width)))
    if scale < 1.0:
        step = max(1, int(round(1.0 / scale)))
        image = image[::step, ::step]
    return image


def render_panel(
    *,
    row: dict[str, str],
    split: str,
    model_dirs: list[Path],
    output_dirs: dict[tuple[str, str], Path],
    output_path: Path,
    max_display_pixels: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    columns = ["original"] + [model_dir.name for model_dir in model_dirs]
    halves = [
        ("even", "even_path", "denoised_even"),
        ("odd", "odd_path", "denoised_odd"),
    ]
    images: list[list[Any]] = []

    for _, manifest_column, denoised_column in halves:
        input_path = Path(row[manifest_column]).expanduser().resolve()
        row_images = [normalize_for_display(load_mrc(input_path), max_display_pixels)]
        for model_dir in model_dirs:
            denoised_dir = output_dirs[(model_dir.name, denoised_column)]
            denoised_path = find_output_file(denoised_dir, input_path)
            if denoised_path is None:
                raise FileNotFoundError(
                    f"Missing denoised output for {input_path.name} in {denoised_dir}"
                )
            row_images.append(
                normalize_for_display(load_mrc(denoised_path), max_display_pixels)
            )
        images.append(row_images)

    ncols = len(columns)
    fig_width = max(12.0, ncols * 1.25)
    fig, axes = plt.subplots(2, ncols, figsize=(fig_width, 3.4), squeeze=False)
    fig.suptitle(f"{split}: {row.get('micrograph_id', '')}", fontsize=8)

    for row_index, (half_text, _, _) in enumerate(halves):
        stack = np.concatenate([image.ravel() for image in images[row_index]])
        vmin, vmax = np.percentile(stack, [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(np.min(stack)), float(np.max(stack))
        for col_index, title in enumerate(columns):
            ax = axes[row_index][col_index]
            ax.imshow(images[row_index][col_index], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(title.replace("topaz_finetuned_", ""), fontsize=5)
            if col_index == 0:
                ax.set_ylabel(half_text, fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.1, w_pad=0.05, h_pad=0.05)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    require_imports()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    visualization_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else eval_dir / "visualization"
    )
    visualization_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_models(eval_dir)
    selected: list[tuple[str, dict[str, str]]] = []
    for split, manifest in [
        ("val", Path(args.val_manifest).expanduser().resolve()),
        ("train", Path(args.train_manifest).expanduser().resolve()),
    ]:
        rows = read_tsv(manifest)[: args.count]
        selected.extend((split, row) for row in rows)

    rendered: list[str] = []
    for split, row in selected:
        output_dirs: dict[tuple[str, str], Path] = {}
        for model_dir in model_dirs:
            for column in ["even_path", "odd_path"]:
                input_path = Path(row[column]).expanduser().resolve()
                output_dirs[(model_dir.name, half_name(column))] = denoise_missing(
                    eval_dir=eval_dir,
                    model_dir=model_dir,
                    split=split,
                    column=column,
                    input_paths=[input_path],
                    visualization_dir=visualization_dir,
                    device=args.device,
                    patch_size=args.patch_size,
                    patch_padding=args.patch_padding,
                    force=args.force_denoise,
                )

        safe_id = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", row.get("micrograph_id", "micrograph")
        )
        panel_path = visualization_dir / split / f"{safe_id}.png"
        render_panel(
            row=row,
            split=split,
            model_dirs=model_dirs,
            output_dirs=output_dirs,
            output_path=panel_path,
            max_display_pixels=args.max_display_pixels,
            dpi=args.dpi,
        )
        rendered.append(str(panel_path))
        print(f"Rendered: {panel_path}")

    write_json(
        visualization_dir / "visualization_manifest.json",
        {
            "eval_dir": str(eval_dir),
            "model_columns": ["original"]
            + [model_dir.name for model_dir in model_dirs],
            "rendered_panels": rendered,
            "count_per_split": args.count,
        },
    )
    print(f"Visualization directory: {visualization_dir}")


if __name__ == "__main__":
    main()
