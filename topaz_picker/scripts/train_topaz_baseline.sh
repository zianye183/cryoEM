#!/usr/bin/env sh
set -eu

# Train the J42 baseline Topaz model from converter-generated metadata.
# Run from the repository root:
#   sh scripts/train_topaz_baseline.sh

TOPAZ_BIN="${TOPAZ_BIN:-/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz}"
RUN_DIR="${RUN_DIR:-runs/j42_baseline}"
METADATA_DIR="${METADATA_DIR:-${RUN_DIR}/metadata}"
MODELS_DIR="${MODELS_DIR:-${RUN_DIR}/models}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/eval}"
SAVE_PREFIX="${SAVE_PREFIX:-${MODELS_DIR}/prestin_topaz}"
TRAIN_LOG="${TRAIN_LOG:-${EVAL_DIR}/topaz_train.log}"
TRAIN_CURVE="${TRAIN_CURVE:-${EVAL_DIR}/train_curve.tsv}"

# Baseline hyperparameters. Override via environment variables if needed, e.g.:
#   NUM_EPOCHS=20 RADIUS=8 sh scripts/train_topaz_baseline.sh
METHOD="${METHOD:-GE-binomial}"
MODEL="${MODEL:-resnet8}"
NUM_PARTICLES="${NUM_PARTICLES:-30}"
RADIUS="${RADIUS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

# Runtime controls. Useful for smoke tests and HPC resource tuning.
DEVICE="${DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-0}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
EPOCH_SIZE="${EPOCH_SIZE:-1000}"
PRELOAD="${PRELOAD:-auto}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
PATCH_SIZE="${PATCH_SIZE:-96}"
PATCH_PADDING="${PATCH_PADDING:-48}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:-}"

TRAIN_IMAGES_TSV="${METADATA_DIR}/train_images.tsv"
TRAIN_TARGETS_TSV="${METADATA_DIR}/train_particles.tsv"
TEST_IMAGES_TSV="${METADATA_DIR}/test_images.tsv"
TEST_TARGETS_TSV="${METADATA_DIR}/test_particles.tsv"

# Stage Topaz-specific files without changing converter outputs.
# This Topaz install detects coordinate format from file extension in some code paths,
# even when --format coord is supplied. It maps .txt/.tab to coord format.
# It also strips image extensions from coordinate image_name values, so staged image
# lists use extensionless image_name values too.
TRAIN_IMAGES="${METADATA_DIR}/train_images_topaz.tab"
TEST_IMAGES="${METADATA_DIR}/test_images_topaz.tab"
TRAIN_TARGETS="${METADATA_DIR}/train_particles.tab"
TEST_TARGETS="${METADATA_DIR}/test_particles.tab"

if [ ! -x "${TOPAZ_BIN}" ]; then
  echo "ERROR: Topaz executable not found or not executable: ${TOPAZ_BIN}" >&2
  echo "Set TOPAZ_BIN=/path/to/topaz and rerun." >&2
  exit 1
fi

for required_file in "${TRAIN_IMAGES_TSV}" "${TRAIN_TARGETS_TSV}" "${TEST_IMAGES_TSV}" "${TEST_TARGETS_TSV}"; do
  if [ ! -f "${required_file}" ]; then
    echo "ERROR: required metadata file missing: ${required_file}" >&2
    echo "Run scripts/cryosparc_cs_to_topaz.py first." >&2
    exit 1
  fi
done

mkdir -p "${MODELS_DIR}" "${EVAL_DIR}"
python3 - "${TRAIN_IMAGES_TSV}" "${TRAIN_IMAGES}" "${TRAIN_TARGETS_TSV}" "${TRAIN_TARGETS}" "${TEST_IMAGES_TSV}" "${TEST_IMAGES}" "${TEST_TARGETS_TSV}" "${TEST_TARGETS}" <<'PY'
import csv
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {'.mrc', '.mrcs', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}

def topaz_name(name):
    path = Path(name)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return path.stem
    return name

def stage_tsv(src, dst):
    with open(src, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit(f'empty TSV: {src}')
        rows = []
        for row in reader:
            if 'image_name' in row:
                row['image_name'] = topaz_name(row['image_name'])
            rows.append(row)
    with open(dst, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

args = sys.argv[1:]
for src, dst in zip(args[0::2], args[1::2]):
    stage_tsv(src, dst)
PY

: > "${TRAIN_LOG}"
{
  echo "# Topaz training command"
  echo "# Started: $(date)"
  echo "# TOPAZ_BIN=${TOPAZ_BIN}"
  echo "# RUN_DIR=${RUN_DIR}"
  echo "# METHOD=${METHOD}"
  echo "# MODEL=${MODEL}"
  echo "# NUM_PARTICLES=${NUM_PARTICLES}"
  echo "# RADIUS=${RADIUS}"
  echo "# NUM_EPOCHS=${NUM_EPOCHS}"
  echo "# DEVICE=${DEVICE}"
  echo "# NUM_WORKERS=${NUM_WORKERS}"
  echo "# NUM_THREADS=${NUM_THREADS}"
  echo "# MINIBATCH_SIZE=${MINIBATCH_SIZE}"
  echo "# EPOCH_SIZE=${EPOCH_SIZE}"
  echo "# PRELOAD=${PRELOAD}"
  echo "# TEST_BATCH_SIZE=${TEST_BATCH_SIZE}"
  echo "# PATCH_SIZE=${PATCH_SIZE}"
  echo "# PATCH_PADDING=${PATCH_PADDING}"
  echo "# PRETRAINED_FLAG=${PRETRAINED_FLAG}"
  echo "# TRAIN_IMAGES=${TRAIN_IMAGES}"
  echo "# TRAIN_TARGETS=${TRAIN_TARGETS}"
  echo "# TEST_IMAGES=${TEST_IMAGES}"
  echo "# TEST_TARGETS=${TEST_TARGETS}"
  echo
  set -x
} >> "${TRAIN_LOG}" 2>&1

set +e
"${TOPAZ_BIN}" train \
  --train-images "${TRAIN_IMAGES}" \
  --train-targets "${TRAIN_TARGETS}" \
  --test-images "${TEST_IMAGES}" \
  --test-targets "${TEST_TARGETS}" \
  --format coord \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-threads "${NUM_THREADS}" \
  --method "${METHOD}" \
  --model "${MODEL}" \
  -n "${NUM_PARTICLES}" \
  -r "${RADIUS}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --epoch-size "${EPOCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --preload "${PRELOAD}" \
  --test-batch-size "${TEST_BATCH_SIZE}" \
  --patch-size "${PATCH_SIZE}" \
  --patch-padding "${PATCH_PADDING}" \
  ${PRETRAINED_FLAG} \
  --save-prefix "${SAVE_PREFIX}" \
  -o "${TRAIN_CURVE}" >> "${TRAIN_LOG}" 2>&1
status=$?
set -e
set +x

{
  echo
  echo "# Finished: $(date)"
  echo "# Exit status: ${status}"
} >> "${TRAIN_LOG}" 2>&1

cat "${TRAIN_LOG}"

if [ "${status}" -ne 0 ]; then
  echo "ERROR: topaz train failed; see ${TRAIN_LOG}" >&2
  exit "${status}"
fi

echo "Training log: ${TRAIN_LOG}"
echo "Training curve: ${TRAIN_CURVE}"
echo "Model prefix: ${SAVE_PREFIX}"
