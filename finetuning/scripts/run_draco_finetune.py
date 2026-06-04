#!/usr/bin/env python3
"""Fine-tune DRACO denoising with Noise2Noise even/odd micrograph pairs.

This script intentionally does not depend on Topaz at runtime. It borrows the
same practical N2N data contract used by the Topaz launcher in this project:
train/validation manifests contain aligned `even_path` and `odd_path` MRCs, and
training uses random aligned crops from one half to predict the other half.

DRACO's public denoising model returns patchified predictions, so the target
crop is patchified before computing the N2N loss.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from finetuning_common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RESULTS_DIR,
    file_sha256,
    quote_command,
    read_tsv,
    validate_readable_file,
    write_command_log,
    write_json,
)

DEFAULT_DRACO_ROOT = Path("/nfs/bartesaghilab2/zy223/denoising/external/draco")
DEFAULT_DRACO_CONFIG = DEFAULT_DRACO_ROOT / "notebook" / "denoise.yaml"
DEFAULT_DRACO_CHECKPOINT = Path(
    "/nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt"
)

GENERATED_OUTPUT_NAMES = {
    "checkpoints",
    "commands.txt",
    "run_metadata.json",
    "logs",
    "loss_history.tsv",
    "loss_history.json",
}


@dataclass(frozen=True)
class PairRecord:
    micrograph_id: str
    even_path: Path
    odd_path: Path


class Tee:
    """Write stream output to both terminal and a log file."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "train_pairs.tsv"),
        help="Training split TSV with even_path and odd_path columns.",
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DEFAULT_MANIFEST_DIR / "val_pairs.tsv"),
        help="Validation split TSV with even_path and odd_path columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "draco_finetuned_pilot"),
        help="Output directory for checkpoints, logs, and metadata.",
    )
    parser.add_argument("--draco-root", default=str(DEFAULT_DRACO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_DRACO_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_DRACO_CHECKPOINT))
    parser.add_argument(
        "--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Optional max gradient norm. Disabled by default.",
    )
    parser.add_argument(
        "--crop", type=int, default=1024, help="Square random crop size in pixels."
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Checkpoint prefix. Defaults to <output-dir>/checkpoints/draco_finetuned.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume from a checkpoint written by this script, including optimizer state and next epoch.",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly train both even->odd and odd->even directions.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply synchronized random flips and 90-degree rotations.",
    )
    parser.add_argument(
        "--contrast-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply DRACO notebook-style percentile contrast clipping before z-scoring.",
    )
    parser.add_argument(
        "--normalization",
        choices=("independent", "input", "paired"),
        default="independent",
        help=(
            "Z-score policy. independent: each half uses its own mean/std; "
            "input: target uses input half mean/std; paired: both use combined even/odd mean/std."
        ),
    )
    parser.add_argument(
        "--normalization-scope",
        choices=("image", "crop"),
        default="image",
        help=(
            "Scope for contrast clipping and z-score statistics. image matches DRACO notebook-style whole-image "
            "preprocessing before cropping; crop preserves the older crop-local experimental behavior."
        ),
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests; applies before bidirectional sampling.",
    )
    parser.add_argument(
        "--max-val-pairs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--validate-mrc-metadata",
        action="store_true",
        help="Validate even/odd dimensions and crop compatibility before training.",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_generated_outputs(output_dir: Path) -> None:
    for name in GENERATED_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil

            shutil.rmtree(path)


def ensure_output_dir(path: Path, *, overwrite: bool, dry_run: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    contents = [entry for entry in path.iterdir() if entry.name != ".gitkeep"]
    if not contents:
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to replace this script's generated artifacts."
        )
    clean_generated_outputs(path)


def load_pairs(manifest_path: Path, *, limit: int | None = None) -> list[PairRecord]:
    rows = read_tsv(manifest_path)
    if limit is not None:
        rows = rows[:limit]
    pairs: list[PairRecord] = []
    for row_index, row in enumerate(rows):
        micrograph_id = row.get("micrograph_id") or f"row_{row_index:06d}"
        even_path = validate_readable_file(row.get("even_path", ""), label="even_path")
        odd_path = validate_readable_file(row.get("odd_path", ""), label="odd_path")
        pairs.append(
            PairRecord(
                micrograph_id=micrograph_id, even_path=even_path, odd_path=odd_path
            )
        )
    if not pairs:
        raise ValueError(f"No pairs found in {manifest_path}")
    return pairs


def validate_pair_metadata(pairs: Iterable[PairRecord], *, crop: int) -> dict[str, Any]:
    import mrcfile  # type: ignore[import-not-found]

    shape_counts: dict[str, int] = {}
    checked = 0
    for pair in pairs:
        with mrcfile.open(pair.even_path, permissive=True) as even_mrc:
            even_data_shape = tuple(int(dim) for dim in even_mrc.data.shape)
        with mrcfile.open(pair.odd_path, permissive=True) as odd_mrc:
            odd_data_shape = tuple(int(dim) for dim in odd_mrc.data.shape)
        if len(even_data_shape) != 2 or len(odd_data_shape) != 2:
            raise ValueError(
                f"Expected 2D MRCs for {pair.micrograph_id}; got {even_data_shape} and {odd_data_shape}"
            )
        even_shape = even_data_shape
        odd_shape = odd_data_shape
        if even_shape != odd_shape:
            raise ValueError(
                f"Shape mismatch for {pair.micrograph_id}: {even_shape} vs {odd_shape}"
            )
        if min(even_shape) < crop:
            raise ValueError(
                f"Crop {crop} exceeds shape {even_shape} for {pair.micrograph_id}"
            )
        key = f"{even_shape[0]}x{even_shape[1]}"
        shape_counts[key] = shape_counts.get(key, 0) + 1
        checked += 1
    return {"metadata_checked_pairs": checked, "shape_counts": shape_counts}


def contrast_normalization(
    image: np.ndarray, tile_size: int = 128, extend: float = 1.5
) -> tuple[float, float]:
    """Return display-style robust contrast bounds from DRACO's notebook helper."""
    ny, nx = image.shape
    highs: list[float] = []
    lows: list[float] = []
    for y0 in range(0, ny, tile_size):
        for x0 in range(0, nx, tile_size):
            tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
            if tile.size == 0:
                continue
            highs.append(float(np.percentile(tile, 98)))
            lows.append(float(np.percentile(tile, 2)))
    high = float(np.nanmedian(highs))
    low = float(np.nanmedian(lows))
    vmid = 0.5 * (low + high)
    vrange = abs(low - high)
    return vmid - extend * 0.5 * vrange, vmid + extend * 0.5 * vrange


def maybe_clip(image: np.ndarray, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return image.astype(np.float32, copy=False)
    vmin, vmax = contrast_normalization(image)
    return np.clip(image, vmin, vmax).astype(np.float32, copy=False)


def zscore_pair(
    source: np.ndarray, target: np.ndarray, *, policy: str
) -> tuple[np.ndarray, np.ndarray]:
    if policy == "independent":
        source_mean, source_std = float(source.mean()), float(source.std())
        target_mean, target_std = float(target.mean()), float(target.std())
    elif policy == "input":
        source_mean, source_std = float(source.mean()), float(source.std())
        target_mean, target_std = source_mean, source_std
    elif policy == "paired":
        stacked = np.concatenate([source.reshape(-1), target.reshape(-1)])
        source_mean = target_mean = float(stacked.mean())
        source_std = target_std = float(stacked.std())
    else:
        raise ValueError(f"Unknown normalization policy: {policy}")
    if source_std == 0:
        source_std = 1.0
    if target_std == 0:
        target_std = 1.0
    return (source - source_mean) / source_std, (target - target_mean) / target_std


class DracoN2NDataset:
    """Aligned even/odd random-crop dataset for DRACO N2N fine-tuning."""

    def __init__(
        self,
        pairs: list[PairRecord],
        *,
        crop: int,
        bidirectional: bool,
        augment: bool,
        contrast_normalize: bool,
        normalization: str,
        normalization_scope: str,
        random_crop: bool,
        direction: str,
    ) -> None:
        self.pairs = pairs
        self.crop = crop
        self.bidirectional = bidirectional
        self.augment = augment
        self.contrast_normalize = contrast_normalize
        self.normalization = normalization
        self.normalization_scope = normalization_scope
        self.random_crop = random_crop
        self.direction = direction
        if direction not in {"random", "even_to_odd", "odd_to_even", "both"}:
            raise ValueError(f"Unknown direction policy: {direction}")

    def __len__(self) -> int:
        multiplier = 2 if self.direction == "both" else 1
        return len(self.pairs) * multiplier

    @staticmethod
    def _read_mrc(path: Path) -> np.ndarray:
        import mrcfile  # type: ignore[import-not-found]

        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data, dtype=np.float32)
            if data.ndim != 2:
                raise ValueError(
                    f"Expected a 2D micrograph MRC at {path}; got shape {data.shape}. "
                    "Refusing to silently train on the first slice of a stack."
                )
            return np.array(data, dtype=np.float32, copy=True)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch  # type: ignore[import-not-found]

        pair_index = index // 2 if self.direction == "both" else index
        pair = self.pairs[pair_index]
        even = self._read_mrc(pair.even_path)
        odd = self._read_mrc(pair.odd_path)
        if even.shape != odd.shape:
            raise ValueError(
                f"Shape mismatch for {pair.micrograph_id}: {even.shape} vs {odd.shape}"
            )

        height, width = even.shape
        if height < self.crop or width < self.crop:
            raise ValueError(
                f"Crop {self.crop} exceeds shape {even.shape} for {pair.micrograph_id}"
            )
        if self.random_crop:
            y0 = np.random.randint(0, height - self.crop + 1)
            x0 = np.random.randint(0, width - self.crop + 1)
        else:
            y0 = (height - self.crop) // 2
            x0 = (width - self.crop) // 2

        if self.direction == "both":
            use_odd_to_even = index % 2 == 1
        elif self.direction == "odd_to_even":
            use_odd_to_even = True
        elif self.direction == "even_to_odd":
            use_odd_to_even = False
        else:
            use_odd_to_even = self.bidirectional and np.random.rand() > 0.5

        if use_odd_to_even:
            source_full, target_full = odd, even
        else:
            source_full, target_full = even, odd

        if self.normalization_scope == "image":
            source_full = maybe_clip(source_full, enabled=self.contrast_normalize)
            target_full = maybe_clip(target_full, enabled=self.contrast_normalize)
            source_full, target_full = zscore_pair(
                source_full, target_full, policy=self.normalization
            )
            source = source_full[y0 : y0 + self.crop, x0 : x0 + self.crop]
            target = target_full[y0 : y0 + self.crop, x0 : x0 + self.crop]
        elif self.normalization_scope == "crop":
            source = source_full[y0 : y0 + self.crop, x0 : x0 + self.crop]
            target = target_full[y0 : y0 + self.crop, x0 : x0 + self.crop]
            source = maybe_clip(source, enabled=self.contrast_normalize)
            target = maybe_clip(target, enabled=self.contrast_normalize)
            source, target = zscore_pair(source, target, policy=self.normalization)
        else:
            raise ValueError(f"Unknown normalization scope: {self.normalization_scope}")

        if self.augment:
            if np.random.rand() > 0.5:
                source = np.flip(source, axis=0)
                target = np.flip(target, axis=0)
            if np.random.rand() > 0.5:
                source = np.flip(source, axis=1)
                target = np.flip(target, axis=1)
            k = int(np.random.randint(0, 4))
            source = np.rot90(source, k=k)
            target = np.rot90(target, k=k)

        source = np.ascontiguousarray(source, dtype=np.float32)
        target = np.ascontiguousarray(target, dtype=np.float32)
        return torch.from_numpy(source).unsqueeze(0), torch.from_numpy(
            target
        ).unsqueeze(0)


def patchify(image: Any, patch_size: int) -> Any:
    """Patchify `[B, C, H, W]` to `[B, L, P*P*C]`, matching DRACO output."""
    import torch  # type: ignore[import-not-found]

    batch, channels, height, width = image.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    if pad_h or pad_w:
        image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h), mode="constant", value=0
        )
        height = height + pad_h
        width = width + pad_w
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 5, 1)
    return patches.reshape(batch, -1, patch_size * patch_size * channels)


def set_seed(seed: int) -> None:
    import torch  # type: ignore[import-not-found]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> None:
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return
    for key in keys:
        state_dict[key[len(prefix) :]] = state_dict.pop(key)


def load_pretrained_strict(
    model: Any, checkpoint_path: Path, device: Any
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    strip_prefix_if_present(state_dict, "module.")
    strip_prefix_if_present(state_dict, "model.")
    strip_prefix_if_present(state_dict, "_orig_mod.")
    message = model.load_state_dict(state_dict, strict=False)
    missing = list(message.missing_keys)
    unexpected = list(message.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/config mismatch while loading DRACO weights. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    print(f"Loaded pre-trained model from {checkpoint_path} with message: {message}")
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def git_revision(path: Path) -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        return {"commit": rev, "dirty": bool(status), "status_short": status}
    except Exception as exc:
        return {"error": str(exc)}


def runtime_provenance(
    script_path: Path, draco_root: Path, config_path: Path
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "script_sha256": file_sha256(script_path),
        "draco_git": git_revision(draco_root),
    }
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = torch.cuda.is_available()
        provenance["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name_0": torch.cuda.get_device_name(0)
            if cuda_available and torch.cuda.device_count() > 0
            else None,
        }
    except Exception as exc:
        provenance["torch"] = {"error": str(exc)}
    try:
        provenance["draco_config_text"] = config_path.read_text(encoding="utf-8")
    except Exception as exc:
        provenance["draco_config_text_error"] = str(exc)
    return provenance


def count_trainable_parameters(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def build_draco_model(args: argparse.Namespace):
    draco_root = Path(args.draco_root).expanduser().resolve()
    sys.path.insert(0, str(draco_root))

    import torch  # type: ignore[import-not-found]
    from draco.configuration import CfgNode  # type: ignore[import-not-found]
    from draco.model import build_model  # type: ignore[import-not-found]

    cfg = CfgNode.load_yaml_with_base(str(Path(args.config).expanduser().resolve()))
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {requested_device}, but CUDA is not available. "
            "Use --device cpu intentionally or run on a GPU node."
        )
    device = torch.device(requested_device)
    model = build_model(cfg).to(device)
    load_report = load_pretrained_strict(
        model, Path(args.checkpoint).expanduser().resolve(), device
    )
    return model, cfg, device, load_report


def run_epoch(
    *,
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: Any,
    patch_size: int,
    grad_clip: float | None = None,
) -> float:
    import torch  # type: ignore[import-not-found]

    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_examples = 0
    loss_fn = torch.nn.MSELoss(reduction="mean")

    for source, target in dataloader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_patches = patchify(target, patch_size)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(source)
            loss = loss_fn(pred, target_patches)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered: {float(loss.detach().cpu())}"
            )
        if train:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(source.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Empty dataloader")
    return total_loss / total_examples


def write_loss_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss", "seconds"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    import torch  # type: ignore[import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "training_args": vars(args),
        },
        path,
    )


def load_resume_checkpoint(
    path: Path, *, model: Any, optimizer: Any, device: Any
) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError(f"Resume checkpoint is missing model/optimizer state: {path}")
    message = model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = int(checkpoint.get("epoch", 0))
    val_loss = float(checkpoint.get("val_loss", "inf"))
    print(f"Resumed from {path} after epoch {epoch} with message: {message}")
    return {"path": str(path), "completed_epoch": epoch, "val_loss": val_loss}


def command_for_log(script_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        str(script_path),
        "--train-manifest",
        str(Path(args.train_manifest).expanduser().resolve()),
        "--val-manifest",
        str(Path(args.val_manifest).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--draco-root",
        str(Path(args.draco_root).expanduser().resolve()),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--checkpoint",
        str(Path(args.checkpoint).expanduser().resolve()),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--crop",
        str(args.crop),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--save-interval",
        str(args.save_interval),
        "--normalization",
        args.normalization,
        "--normalization-scope",
        args.normalization_scope,
    ]
    if args.grad_clip is not None:
        command.extend(["--grad-clip", str(args.grad_clip)])
    command.append("--bidirectional" if args.bidirectional else "--no-bidirectional")
    command.append("--augment" if args.augment else "--no-augment")
    command.append(
        "--contrast-normalize" if args.contrast_normalize else "--no-contrast-normalize"
    )
    if args.save_prefix:
        command.extend(
            ["--save-prefix", str(Path(args.save_prefix).expanduser().resolve())]
        )
    if args.resume_checkpoint:
        command.extend(
            [
                "--resume-checkpoint",
                str(Path(args.resume_checkpoint).expanduser().resolve()),
            ]
        )
    if args.max_train_pairs is not None:
        command.extend(["--max-train-pairs", str(args.max_train_pairs)])
    if args.max_val_pairs is not None:
        command.extend(["--max-val-pairs", str(args.max_val_pairs)])
    if args.validate_mrc_metadata:
        command.append("--validate-mrc-metadata")
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=args.overwrite, dry_run=args.dry_run)

    train_manifest = Path(args.train_manifest).expanduser().resolve()
    val_manifest = Path(args.val_manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    resume_checkpoint_path = (
        Path(args.resume_checkpoint).expanduser().resolve()
        if args.resume_checkpoint
        else None
    )
    validate_readable_file(str(config_path), label="DRACO config")
    validate_readable_file(str(checkpoint_path), label="DRACO checkpoint")
    if resume_checkpoint_path is not None:
        validate_readable_file(str(resume_checkpoint_path), label="resume checkpoint")

    train_pairs = load_pairs(train_manifest, limit=args.max_train_pairs)
    val_pairs = load_pairs(val_manifest, limit=args.max_val_pairs)

    train_meta: dict[str, Any] = {"pairs": len(train_pairs)}
    val_meta: dict[str, Any] = {"pairs": len(val_pairs)}
    if args.validate_mrc_metadata:
        train_meta.update(validate_pair_metadata(train_pairs, crop=args.crop))
        val_meta.update(validate_pair_metadata(val_pairs, crop=args.crop))

    save_prefix = (
        Path(args.save_prefix).expanduser().resolve()
        if args.save_prefix
        else output_dir / "checkpoints" / "draco_finetuned"
    )
    log_path = (
        Path(args.log_file).expanduser().resolve()
        if args.log_file
        else output_dir / "logs" / "draco_train.log"
    )
    command = command_for_log(script_path, args)
    write_command_log(output_dir / "commands.txt", [command])

    write_json(
        output_dir / "run_metadata.json",
        {
            "execution_mode": "draco_custom_n2n_patch_mse",
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": file_sha256(train_manifest),
            "val_manifest": str(val_manifest),
            "val_manifest_sha256": file_sha256(val_manifest),
            "train_summary": train_meta,
            "val_summary": val_meta,
            "draco_root": str(Path(args.draco_root).expanduser().resolve()),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "resume_checkpoint": str(resume_checkpoint_path)
            if resume_checkpoint_path
            else None,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "crop": args.crop,
            "bidirectional": args.bidirectional,
            "augment": args.augment,
            "contrast_normalize": args.contrast_normalize,
            "normalization": args.normalization,
            "normalization_scope": args.normalization_scope,
            "grad_clip": args.grad_clip,
            "validation_crop_policy": "center",
            "validation_direction_policy": "both"
            if args.bidirectional
            else "even_to_odd",
            "save_prefix": str(save_prefix),
            "command": command,
            "training_log": str(log_path),
            "loss_history_tsv": str(output_dir / "loss_history.tsv"),
            "dry_run": args.dry_run,
            "note": "Custom DRACO fine-tuning using only Noise2Noise MSE; no L_recon term.",
            "provenance": runtime_provenance(
                script_path, Path(args.draco_root).expanduser().resolve(), config_path
            ),
        },
    )

    print(f"Train pairs:       {len(train_pairs)}")
    print(f"Val pairs:         {len(val_pairs)}")
    print(f"Output directory:  {output_dir}")
    print(f"Training log:      {log_path}")
    print(f"DRACO config:      {config_path}")
    print(f"DRACO checkpoint:  {checkpoint_path}")
    print(quote_command(command))

    if args.dry_run:
        print("Dry run only; training was not executed.")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(quote_command(command))
        log_handle.write("\n")
        tee_stdout = Tee(sys.stdout, log_handle)
        tee_stderr = Tee(sys.stderr, log_handle)
        with (
            contextlib.redirect_stdout(tee_stdout),
            contextlib.redirect_stderr(tee_stderr),
        ):
            set_seed(args.seed)
            import torch  # type: ignore[import-not-found]
            from torch.utils.data import DataLoader  # type: ignore[import-not-found]

            model, cfg, device, checkpoint_load_report = build_draco_model(args)
            patch_size = int(cfg.MODEL.PATCH_SIZE)
            metadata_path = output_dir / "run_metadata.json"
            import json

            with metadata_path.open("r", encoding="utf-8") as metadata_handle:
                metadata = json.load(metadata_handle)
            metadata["checkpoint_load_report"] = checkpoint_load_report
            metadata["model_parameters"] = count_trainable_parameters(model)
            write_json(metadata_path, metadata)
            if args.crop % patch_size != 0:
                print(
                    f"# warning: crop {args.crop} is not divisible by patch size {patch_size}; target will be zero-padded.",
                    file=sys.stderr,
                )

            train_dataset = DracoN2NDataset(
                train_pairs,
                crop=args.crop,
                bidirectional=args.bidirectional,
                augment=args.augment,
                contrast_normalize=args.contrast_normalize,
                normalization=args.normalization,
                normalization_scope=args.normalization_scope,
                random_crop=True,
                direction="random",
            )
            val_dataset = DracoN2NDataset(
                val_pairs,
                crop=args.crop,
                bidirectional=False,
                augment=False,
                contrast_normalize=args.contrast_normalize,
                normalization=args.normalization,
                normalization_scope=args.normalization_scope,
                random_crop=False,
                direction="both" if args.bidirectional else "even_to_odd",
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=str(device).startswith("cuda"),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=str(device).startswith("cuda"),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

            resume_report = None
            start_epoch = 1
            best_val_loss = float("inf")
            if resume_checkpoint_path is not None:
                resume_report = load_resume_checkpoint(
                    resume_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                )
                start_epoch = int(resume_report["completed_epoch"]) + 1
                best_val_loss = float(resume_report["val_loss"])
                with metadata_path.open("r", encoding="utf-8") as metadata_handle:
                    metadata = json.load(metadata_handle)
                metadata["resume_report"] = resume_report
                write_json(metadata_path, metadata)

            print("Epoch\tTrain Loss\tVal Loss\tBest Val Loss\tSeconds")
            loss_rows: list[dict[str, Any]] = []
            for epoch in range(start_epoch, args.epochs + 1):
                start = time.time()
                train_loss = run_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    patch_size=patch_size,
                    grad_clip=args.grad_clip,
                )
                with torch.no_grad():
                    val_loss = run_epoch(
                        model=model,
                        dataloader=val_loader,
                        optimizer=None,
                        device=device,
                        patch_size=patch_size,
                    )
                elapsed = time.time() - start
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        save_prefix.parent / f"{save_prefix.name}_best.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        args=args,
                    )
                if args.save_interval and epoch % args.save_interval == 0:
                    save_checkpoint(
                        save_prefix.parent / f"{save_prefix.name}_epoch{epoch:03d}.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        args=args,
                    )
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "seconds": elapsed,
                }
                loss_rows.append(row)
                write_loss_history(output_dir / "loss_history.tsv", loss_rows)
                write_json(output_dir / "loss_history.json", {"rows": loss_rows})
                print(
                    f"{epoch}\t{train_loss:.6g}\t{val_loss:.6g}\t{best_val_loss:.6g}\t{elapsed:.1f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
