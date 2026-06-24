#!/usr/bin/env bash
set -euo pipefail

# Canonical Prestin Topaz-Denoise fine-tuning launcher.
#
# Default behavior is a dry run that prepares symlinked train/validation input
# directories and writes command metadata without starting training. Promote to
# an actual run with `DRY_RUN=0`.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CET_PICK_SHARED_ROOT="${CET_PICK_SHARED_ROOT:-/nfs/bartesaghilab2/zy223}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CET_PICK_SHARED_ROOT}/.cache}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"

topaz_venv="${TOPAZ_VENV:-/nfs/bartesaghilab2/zy223/venvs/topaz}"
manifest_dir="${MANIFEST_DIR:-/nfs/bartesaghilab2/zy223/data/finetuning_data/topaz_prestin}"
output_dir="${OUTPUT_DIR:-/nfs/bartesaghilab2/zy223/finetuning/results/topaz_finetuned_prestin_unet_l2_lr1e-5}"

train_manifest="${TRAIN_MANIFEST:-${manifest_dir}/train_pairs.tsv}"
val_manifest="${VAL_MANIFEST:-${manifest_dir}/val_pairs.tsv}"

device="${DEVICE:-0}"
arch="${ARCH:-unet}"
learning_rate="${LEARNING_RATE:-1e-5}"
epochs="${EPOCHS:-20}"
batch_size="${BATCH_SIZE:-4}"
crop="${CROP:-800}"
criteria="${CRITERIA:-L2}"
optim="${OPTIM:-adagrad}"
num_workers="${NUM_WORKERS:-4}"
save_interval="${SAVE_INTERVAL:-1}"

dry_run="${DRY_RUN:-1}"
overwrite="${OVERWRITE:-0}"
validate_mrc_metadata="${VALIDATE_MRC_METADATA:-0}"

if [[ ! -f "${topaz_venv}/bin/activate" ]]; then
  echo "ERROR: Topaz venv not found: ${topaz_venv}" >&2
  exit 2
fi
if [[ ! -f "${train_manifest}" ]]; then
  echo "ERROR: train manifest not found: ${train_manifest}" >&2
  exit 2
fi
if [[ ! -f "${val_manifest}" ]]; then
  echo "ERROR: val manifest not found: ${val_manifest}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${topaz_venv}/bin/activate"

cmd=(
  python "${script_dir}/run_topaz_finetune.py"
  --train-manifest "${train_manifest}"
  --val-manifest "${val_manifest}"
  --output-dir "${output_dir}"
  --device "${device}"
  --arch "${arch}"
  --learning-rate "${learning_rate}"
  --epochs "${epochs}"
  --batch-size "${batch_size}"
  --crop "${crop}"
  --criteria "${criteria}"
  --optim "${optim}"
  --num-workers "${num_workers}"
  --save-interval "${save_interval}"
)

if [[ "${dry_run}" != "0" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${overwrite}" != "0" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${validate_mrc_metadata}" != "0" ]]; then
  cmd+=(--validate-mrc-metadata)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
