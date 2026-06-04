#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

topaz_venv="${TOPAZ_VENV:-/nfs/bartesaghilab2/zy223/venvs/topaz}"
run_dir="${RUN_DIR:-${project_root}/runs/j42_baseline}"
metadata_dir="${METADATA_DIR:-${run_dir}/metadata}"
models_dir="${MODELS_DIR:-${run_dir}/models}"
predictions_dir="${PREDICTIONS_DIR:-${run_dir}/predictions}"
eval_dir="${EVAL_DIR:-${run_dir}/eval}"
test_targets="${TEST_TARGETS:-${metadata_dir}/test_particles.txt}"

radius="${RADIUS:-10}"
assignment_radius="${ASSIGNMENT_RADIUS:-${radius}}"
threshold="${THRESHOLD:--6}"
num_epochs="${NUM_EPOCHS:-10}"
model_path="${MODEL_PATH:-${models_dir}/prestin_topaz_epoch${num_epochs}.sav}"
predictions_path="${PREDICTIONS_PATH:-${predictions_dir}/test_predictions.tsv}"
pr_curve_path="${PR_CURVE_PATH:-${eval_dir}/pr_curve.tsv}"
test_image_paths="${TEST_IMAGE_PATHS:-${metadata_dir}/test_image_paths.txt}"
eval_log="${EVAL_LOG:-${eval_dir}/evaluate.log}"

mkdir -p "${predictions_dir}" "${eval_dir}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${eval_log}"
}

count_data_rows() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  awk 'END { if (NR > 0) print NR - 1; else print 0 }' "${path}"
}

if [[ ! -f "${model_path}" ]]; then
  echo "Missing model checkpoint: ${model_path}" >&2
  echo "Set MODEL_PATH=/path/to/model.sav if Topaz saved a different checkpoint name." >&2
  exit 1
fi

if [[ ! -f "${test_targets}" ]]; then
  fallback_targets="${test_targets%.*}.tsv"
  if [[ -f "${fallback_targets}" ]]; then
    cp "${fallback_targets}" "${test_targets}"
    echo "Created Topaz coordinate target ${test_targets} from ${fallback_targets}"
  else
    echo "Missing Topaz coordinate target: ${test_targets}" >&2
    echo "Run prepare_j42_baseline.sh with the updated converter, or set TEST_TARGETS." >&2
    exit 1
  fi
fi

python - "${test_targets}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    expected = ["image_name", "x_coord", "y_coord"]
    if reader.fieldnames != expected:
        raise SystemExit(f"{path} columns must be {expected}, got {reader.fieldnames}")
    first = next(reader, None)
    if first is None:
        raise SystemExit(f"{path} has no particle rows")
PY

awk 'NR > 1 { gsub(/\r$/, "", $2); print $2 }' "${metadata_dir}/test_images.tsv" > "${test_image_paths}"
test_images=()
while IFS= read -r image_path; do
  [[ -n "${image_path}" ]] && test_images+=("${image_path}")
done < "${test_image_paths}"

if [[ "${#test_images[@]}" -eq 0 ]]; then
  echo "No held-out image paths found in ${metadata_dir}/test_images.tsv" >&2
  exit 1
fi

source "${topaz_venv}/bin/activate"
: > "${eval_log}"

log "Topaz evaluation setup"
log "  run_dir=${run_dir}"
log "  model_path=${model_path}"
log "  test_images=${#test_images[@]}"
log "  test_targets=${test_targets} ($(count_data_rows "${test_targets}") particles)"
log "  predictions_path=${predictions_path}"
log "  pr_curve_path=${pr_curve_path}"
log "  radius=${radius}, assignment_radius=${assignment_radius}, threshold=${threshold}"

extract_cmd=(
  topaz extract
  -m "${model_path}" \
  -r "${radius}" \
  -t "${threshold}" \
  -o "${predictions_path}" \
  "${test_images[@]}"
)

log "Extracting predictions without inline target scoring"
log "Command: ${extract_cmd[*]}"
set +e
"${extract_cmd[@]}" 2>&1 | tee -a "${eval_log}"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -ne 0 ]]; then
  log "Topaz extraction failed with exit status ${status}"
  exit "${status}"
fi

prediction_count="$(count_data_rows "${predictions_path}")"
log "Predictions written: ${prediction_count}"
if [[ "${prediction_count}" -eq 0 ]]; then
  log "No predictions were extracted at threshold=${threshold}; skipping precision-recall curve."
  log "Try a lower threshold, for example THRESHOLD=-10 or THRESHOLD=-20."
  echo "Predictions: ${predictions_path}"
  exit 0
fi

pr_cmd=(
  topaz precision_recall_curve
  --predicted "${predictions_path}" \
  --targets "${test_targets}" \
  -r "${assignment_radius}"
)

log "Computing precision-recall curve"
log "Command: ${pr_cmd[*]} > ${pr_curve_path}"
set +e
"${pr_cmd[@]}" > "${pr_curve_path}" 2>> "${eval_log}"
status="$?"
set -e
if [[ "${status}" -ne 0 ]]; then
  log "Precision-recall curve failed with exit status ${status}"
  log "Predictions are still available for visual overlay at ${predictions_path}"
  exit "${status}"
fi

echo "Predictions: ${predictions_path}"
echo "Precision-recall curve: ${pr_curve_path}"
echo "Evaluation log: ${eval_log}"
