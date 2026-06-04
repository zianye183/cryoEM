#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

topaz_venv="${TOPAZ_VENV:-/nfs/bartesaghilab2/zy223/venvs/topaz}"
run_dir="${RUN_DIR:-${project_root}/runs/j42_baseline}"
metadata_dir="${METADATA_DIR:-${run_dir}/metadata}"
models_dir="${MODELS_DIR:-${run_dir}/models}"
eval_dir="${EVAL_DIR:-${run_dir}/eval}"
train_targets="${TRAIN_TARGETS:-${metadata_dir}/train_particles.txt}"
test_targets="${TEST_TARGETS:-${metadata_dir}/test_particles.txt}"

method="${TOPAZ_METHOD:-GE-binomial}"
model="${TOPAZ_MODEL:-resnet8}"
num_particles="${NUM_PARTICLES:-}"
radius="${RADIUS:-10}"
num_epochs="${NUM_EPOCHS:-10}"
preload="${TOPAZ_PRELOAD:-auto}"
save_prefix="${SAVE_PREFIX:-${models_dir}/prestin_topaz}"
train_curve="${TRAIN_CURVE:-${eval_dir}/train_curve.tsv}"
train_log="${TRAIN_LOG:-${eval_dir}/train.log}"

mkdir -p "${models_dir}" "${eval_dir}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${train_log}"
}

count_data_rows() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  awk 'END { if (NR > 0) print NR - 1; else print 0 }' "${path}"
}

ensure_coord_target() {
  local target_path="$1"
  local fallback_path="${target_path%.*}.tsv"

  if [[ ! -f "${target_path}" ]]; then
    if [[ -f "${fallback_path}" ]]; then
      cp "${fallback_path}" "${target_path}"
      echo "Created Topaz coordinate target ${target_path} from ${fallback_path}"
    else
      echo "Missing Topaz coordinate target: ${target_path}" >&2
      echo "Run prepare_j42_baseline.sh with the updated converter, or set TRAIN_TARGETS/TEST_TARGETS." >&2
      exit 1
    fi
  fi

  python - "${target_path}" <<'PY'
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
}

for target_path in "${train_targets}" "${test_targets}"; do
  ensure_coord_target "${target_path}"
done

train_images="${metadata_dir}/train_images.tsv"
test_images="${metadata_dir}/test_images.tsv"
if [[ ! -f "${train_images}" ]]; then
  echo "Missing train image list: ${train_images}" >&2
  exit 1
fi
if [[ ! -f "${test_images}" ]]; then
  echo "Missing test image list: ${test_images}" >&2
  exit 1
fi

if [[ -z "${num_particles}" ]]; then
  if [[ ! -f "${metadata_dir}/summary.json" ]]; then
    echo "Missing metadata summary: ${metadata_dir}/summary.json" >&2
    echo "Run prepare_j42_baseline.sh first, or set NUM_PARTICLES explicitly." >&2
    exit 1
  fi
  num_particles="$(
    python -c 'import json, sys
summary = json.load(open(sys.argv[1]))
train_micrographs = summary["num_train_micrographs"]
if train_micrographs <= 0:
    raise SystemExit("summary.json has no training micrographs")
mean_particles = summary["num_train_particles"] / train_micrographs
print(max(1, int(mean_particles + 0.5)))' "${metadata_dir}/summary.json"
  )"
  echo "Derived NUM_PARTICLES=${num_particles} from ${metadata_dir}/summary.json"
fi

source "${topaz_venv}/bin/activate"
: > "${train_log}"

train_micrographs="$(count_data_rows "${train_images}")"
test_micrographs="$(count_data_rows "${test_images}")"
train_particles_count="$(count_data_rows "${train_targets}")"
test_particles_count="$(count_data_rows "${test_targets}")"

log "Topaz training setup"
log "  run_dir=${run_dir}"
log "  metadata_dir=${metadata_dir}"
log "  train_micrographs=${train_micrographs}, train_particles=${train_particles_count}"
log "  test_micrographs=${test_micrographs}, test_particles=${test_particles_count}"
log "  method=${method}, model=${model}, radius=${radius}, num_particles=${num_particles}, num_epochs=${num_epochs}, preload=${preload}"
log "  train_targets=${train_targets}"
log "  test_targets=${test_targets}"
log "  save_prefix=${save_prefix}"
log "  train_curve=${train_curve}"
log "  train_log=${train_log}"
log "Starting Topaz. If preprocessing is the bottleneck, the largest wait is usually before the first epoch output."

topaz_cmd=(
  topaz train
  --train-images "${train_images}"
  --train-targets "${train_targets}"
  --test-images "${test_images}"
  --test-targets "${test_targets}"
  --format coord
  --method "${method}"
  --model "${model}"
  -n "${num_particles}"
  -r "${radius}"
  --num-epochs "${num_epochs}"
  --preload "${preload}"
  --save-prefix "${save_prefix}"
  -o "${train_curve}"
)

log "Command: ${topaz_cmd[*]}"
set +e
"${topaz_cmd[@]}" 2>&1 | tee -a "${train_log}"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" -ne 0 ]]; then
  log "Topaz training failed with exit status ${status}"
  exit "${status}"
fi

log "Topaz training finished"
