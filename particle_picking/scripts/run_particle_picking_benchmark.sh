#!/usr/bin/env bash
# Run a small denoising-assisted particle-picking benchmark.
# Stages symlinks only; raw/preprocessed input data are never copied or modified.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_CONFIG="${BENCH_DIR}/configs/particle_picking_benchmark_pilot.env"
CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

RAW_DIR=${RAW_DIR:-}
MRC_GLOB=${MRC_GLOB:-*.mrc}
LIMIT=${LIMIT:-5}
TOPAZ_DIR=${TOPAZ_DIR:-}
DRACO_DIR=${DRACO_DIR:-}
TOPAZ_SUFFIX=${TOPAZ_SUFFIX:-}
DRACO_SUFFIX=${DRACO_SUFFIX:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BENCH_DIR}/results/particle_picking_benchmark}
RUN_ID=${RUN_ID:-}
DOG_SIGMA_SMALL=${DOG_SIGMA_SMALL:-3.0}
DOG_SIGMA_LARGE=${DOG_SIGMA_LARGE:-9.0}
DOG_POLARITY=${DOG_POLARITY:-absolute}
DOG_THRESHOLD_SIGMA=${DOG_THRESHOLD_SIGMA:-4.0}
DOG_MIN_DISTANCE=${DOG_MIN_DISTANCE:-32}
DOG_MAX_PICKS=${DOG_MAX_PICKS:-2000}
DOG_MATCH_RADIUS=${DOG_MATCH_RADIUS:-48}
DOG_BOX_SIZE=${DOG_BOX_SIZE:-160}
DOG_OVERLAY_MAX_PICKS=${DOG_OVERLAY_MAX_PICKS:-300}
DOG_OVERLAY_DOWNSAMPLE=${DOG_OVERLAY_DOWNSAMPLE:-4}
DOG_OVERLAY_CONTRAST_PERCENTILES=${DOG_OVERLAY_CONTRAST_PERCENTILES:-1,99}
RUN_NEXTPYP=${RUN_NEXTPYP:-0}
PYP_SIF=${PYP_SIF:-/nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-apptainer}
PYP_BINDS=${PYP_BINDS:-}
SCOPE_VOLTAGE=${SCOPE_VOLTAGE:-300}
SCOPE_PIXEL=${SCOPE_PIXEL:-}
SCOPE_CS=${SCOPE_CS:-}
SCOPE_WGH=${SCOPE_WGH:-}
CTF_MIN_RES=${CTF_MIN_RES:-}
CTF_MAX_RES=${CTF_MAX_RES:-}
CTF_MIN_DEF=${CTF_MIN_DEF:-}
CTF_MAX_DEF=${CTF_MAX_DEF:-}
CTF_TILE=${CTF_TILE:-}
CTF_FSTEP=${CTF_FSTEP:-}
CTF_USE_AST=${CTF_USE_AST:-}
DETECT_RAD=${DETECT_RAD:-}
DETECT_METHOD=${DETECT_METHOD:-auto}
DETECT_DIST=${DETECT_DIST:-}
DETECT_THRE=${DETECT_THRE:-}
SLURM_TASKS=${SLURM_TASKS:-1}
SLURM_MEMORY=${SLURM_MEMORY:-32}

if [[ -z "${RAW_DIR}" ]]; then
  echo "RAW_DIR is required in ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${RAW_DIR}" ]]; then
  echo "RAW_DIR does not exist or is not a directory: ${RAW_DIR}" >&2
  exit 2
fi
if [[ "${LIMIT}" -le 0 ]]; then
  echo "LIMIT must be positive; start with LIMIT=5 or LIMIT=10 for the pilot" >&2
  exit 2
fi
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID=$(date +%Y%m%d_%H%M%S)
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/inputs/raw" \
  "${RUN_DIR}/dog/boxes" \
  "${RUN_DIR}/dog/overlays" \
  "${RUN_DIR}/nextpyp"

log() {
  printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"
}

stage_raw_inputs() {
  local count=0
  local raw_path
  while IFS= read -r raw_path; do
    ln -s "${raw_path}" "${RUN_DIR}/inputs/raw/$(basename "${raw_path}")"
    count=$((count + 1))
    if [[ "${count}" -ge "${LIMIT}" ]]; then
      break
    fi
  done < <(find -L "${RAW_DIR}" -maxdepth 1 -name "${MRC_GLOB}" | sort)

  if [[ "${count}" -eq 0 ]]; then
    echo "No raw micrographs found in ${RAW_DIR} with glob ${MRC_GLOB}" >&2
    exit 2
  fi
  log "Staged ${count} raw symlinks"
}

find_condition_file() {
  local source_dir=$1
  local suffix=$2
  local raw_name=$3
  local raw_stem=${raw_name%.*}
  local raw_ext=${raw_name##*.}

  if [[ -f "${source_dir}/${raw_name}" ]]; then
    printf '%s\n' "${source_dir}/${raw_name}"
  elif [[ -n "${suffix}" && -f "${source_dir}/${raw_stem}${suffix}.${raw_ext}" ]]; then
    printf '%s\n' "${source_dir}/${raw_stem}${suffix}.${raw_ext}"
  fi
}

stage_condition_inputs() {
  local method=$1
  local source_dir=$2
  local suffix=$3
  local count=0
  local missing=0
  local raw_link raw_name condition_path

  if [[ -z "${source_dir}" ]]; then
    log "Skipping ${method}: no source directory configured"
    return 0
  fi
  if [[ ! -d "${source_dir}" ]]; then
    echo "Configured ${method} directory does not exist: ${source_dir}" >&2
    exit 2
  fi

  mkdir -p "${RUN_DIR}/inputs/${method}"
  for raw_link in "${RUN_DIR}"/inputs/raw/*.mrc; do
    raw_name=$(basename "${raw_link}")
    condition_path=$(find_condition_file "${source_dir}" "${suffix}" "${raw_name}" || true)
    if [[ -n "${condition_path}" ]]; then
      ln -s "${condition_path}" "${RUN_DIR}/inputs/${method}/${raw_name}"
      count=$((count + 1))
    else
      missing=$((missing + 1))
    fi
  done

  if [[ "${count}" -eq 0 ]]; then
    log "Skipping ${method}: no matching files found for staged raw subset"
    rmdir "${RUN_DIR}/inputs/${method}" 2>/dev/null || true
  else
    log "Staged ${count} ${method} symlinks (${missing} missing)"
  fi
}

write_resolved_config() {
  cat > "${RUN_DIR}/config.env" <<EOF
CONFIG_PATH=${CONFIG_PATH}
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
RAW_DIR=${RAW_DIR}
MRC_GLOB=${MRC_GLOB}
LIMIT=${LIMIT}
TOPAZ_DIR=${TOPAZ_DIR}
DRACO_DIR=${DRACO_DIR}
TOPAZ_SUFFIX=${TOPAZ_SUFFIX}
DRACO_SUFFIX=${DRACO_SUFFIX}
DOG_SIGMA_SMALL=${DOG_SIGMA_SMALL}
DOG_SIGMA_LARGE=${DOG_SIGMA_LARGE}
DOG_POLARITY=${DOG_POLARITY}
DOG_THRESHOLD_SIGMA=${DOG_THRESHOLD_SIGMA}
DOG_MIN_DISTANCE=${DOG_MIN_DISTANCE}
DOG_MAX_PICKS=${DOG_MAX_PICKS}
DOG_MATCH_RADIUS=${DOG_MATCH_RADIUS}
DOG_BOX_SIZE=${DOG_BOX_SIZE}
DOG_OVERLAY_MAX_PICKS=${DOG_OVERLAY_MAX_PICKS}
DOG_OVERLAY_DOWNSAMPLE=${DOG_OVERLAY_DOWNSAMPLE}
DOG_OVERLAY_CONTRAST_PERCENTILES=${DOG_OVERLAY_CONTRAST_PERCENTILES}
RUN_NEXTPYP=${RUN_NEXTPYP}
PYP_SIF=${PYP_SIF}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME}
PYP_BINDS=${PYP_BINDS}
SCOPE_VOLTAGE=${SCOPE_VOLTAGE}
SCOPE_PIXEL=${SCOPE_PIXEL}
SCOPE_CS=${SCOPE_CS}
SCOPE_WGH=${SCOPE_WGH}
CTF_MIN_RES=${CTF_MIN_RES}
CTF_MAX_RES=${CTF_MAX_RES}
CTF_MIN_DEF=${CTF_MIN_DEF}
CTF_MAX_DEF=${CTF_MAX_DEF}
CTF_TILE=${CTF_TILE}
CTF_FSTEP=${CTF_FSTEP}
CTF_USE_AST=${CTF_USE_AST}
DETECT_RAD=${DETECT_RAD}
DETECT_METHOD=${DETECT_METHOD}
DETECT_DIST=${DETECT_DIST}
DETECT_THRE=${DETECT_THRE}
SLURM_TASKS=${SLURM_TASKS}
SLURM_MEMORY=${SLURM_MEMORY}
EOF
}

build_dog_method_args() {
  DOG_METHOD_ARGS=()
  if [[ -d "${RUN_DIR}/inputs/topaz" ]]; then
    DOG_METHOD_ARGS+=(--method "topaz:${RUN_DIR}/inputs/topaz")
  fi
  if [[ -d "${RUN_DIR}/inputs/draco" ]]; then
    DOG_METHOD_ARGS+=(--method "draco:${RUN_DIR}/inputs/draco")
  fi
}

run_dog_benchmark() {
  build_dog_method_args
  log "Running DoG/NMS benchmark"
  python "${SCRIPT_DIR}/evaluate_dog_picking.py" \
    --raw-dir "${RUN_DIR}/inputs/raw" \
    --glob "*.mrc" \
    --output-dir "${RUN_DIR}/dog" \
    --overwrite \
    --sigma-small "${DOG_SIGMA_SMALL}" \
    --sigma-large "${DOG_SIGMA_LARGE}" \
    --polarity "${DOG_POLARITY}" \
    --threshold-sigma "${DOG_THRESHOLD_SIGMA}" \
    --min-distance "${DOG_MIN_DISTANCE}" \
    --max-picks "${DOG_MAX_PICKS}" \
    --match-radius "${DOG_MATCH_RADIUS}" \
    "${DOG_METHOD_ARGS[@]}" \
    > "${RUN_DIR}/logs/evaluate_dog_picking.log" 2>&1

  python "${SCRIPT_DIR}/export_dog_picks_to_box.py" \
    --picks "${RUN_DIR}/dog/dog_picks.tsv" \
    --output-dir "${RUN_DIR}/dog/boxes" \
    --box-size "${DOG_BOX_SIZE}" \
    --include-score \
    > "${RUN_DIR}/logs/export_dog_picks_to_box.log" 2>&1

  python "${SCRIPT_DIR}/plot_picking_overlays.py" \
    --picks "${RUN_DIR}/dog/dog_picks.tsv" \
    --raw-dir "${RUN_DIR}/inputs/raw" \
    --glob "*.mrc" \
    --output-dir "${RUN_DIR}/dog/overlays" \
    --max-picks "${DOG_OVERLAY_MAX_PICKS}" \
    --downsample "${DOG_OVERLAY_DOWNSAMPLE}" \
    --contrast-percentiles "${DOG_OVERLAY_CONTRAST_PERCENTILES}" \
    > "${RUN_DIR}/logs/plot_picking_overlays.log" 2>&1
}

write_nextpyp_script() {
  local method=$1
  local input_dir="${RUN_DIR}/inputs/${method}"
  local project_dir="${RUN_DIR}/nextpyp/${method}"
  local script_path="${project_dir}/run_nextpyp.sh"

  if [[ ! -d "${input_dir}" ]]; then
    return 0
  fi
  mkdir -p "${project_dir}"

  cat > "${script_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${project_dir}"
# Generated by run_particle_picking_benchmark.sh.
# If nextPYP rejects the command, set SCOPE_PIXEL, DETECT_RAD, and DETECT_DIST
# in the benchmark config and regenerate this file.
${CONTAINER_RUNTIME} exec ${PYP_BINDS} "${PYP_SIF}" pyp \\
  -data_mode spr \\
  -scope_voltage "${SCOPE_VOLTAGE}" \\
EOF

  if [[ -n "${SCOPE_PIXEL}" ]]; then
    printf '  -scope_pixel "%s" \\\n' "${SCOPE_PIXEL}" >> "${script_path}"
  fi
  if [[ -n "${SCOPE_CS}" ]]; then
    printf '  -scope_cs "%s" \\\n' "${SCOPE_CS}" >> "${script_path}"
  fi
  if [[ -n "${SCOPE_WGH}" ]]; then
    printf '  -scope_wgh "%s" \\\n' "${SCOPE_WGH}" >> "${script_path}"
  fi

  cat >> "${script_path}" <<EOF
  -data_path="${input_dir}/*.mrc" \\
EOF

  if [[ -n "${CTF_MIN_RES}" ]]; then
    printf '  -ctf_min_res "%s" \\\n' "${CTF_MIN_RES}" >> "${script_path}"
  fi
  if [[ -n "${CTF_MAX_RES}" ]]; then
    printf '  -ctf_max_res "%s" \\\n' "${CTF_MAX_RES}" >> "${script_path}"
  fi
  if [[ -n "${CTF_MIN_DEF}" ]]; then
    printf '  -ctf_min_def "%s" \\\n' "${CTF_MIN_DEF}" >> "${script_path}"
  fi
  if [[ -n "${CTF_MAX_DEF}" ]]; then
    printf '  -ctf_max_def "%s" \\\n' "${CTF_MAX_DEF}" >> "${script_path}"
  fi
  if [[ -n "${CTF_TILE}" ]]; then
    printf '  -ctf_tile "%s" \\\n' "${CTF_TILE}" >> "${script_path}"
  fi
  if [[ -n "${CTF_FSTEP}" ]]; then
    printf '  -ctf_fstep "%s" \\\n' "${CTF_FSTEP}" >> "${script_path}"
  fi
  if [[ "${CTF_USE_AST}" == "true" || "${CTF_USE_AST}" == "1" ]]; then
    printf '  -ctf_use_ast \\\n' >> "${script_path}"
  elif [[ "${CTF_USE_AST}" == "false" || "${CTF_USE_AST}" == "0" ]]; then
    printf '  -no-ctf_use_ast \\\n' >> "${script_path}"
  fi

  cat >> "${script_path}" <<EOF
  -detect_method "${DETECT_METHOD}" \\
EOF

  if [[ -n "${DETECT_RAD}" ]]; then
    printf '  -detect_rad "%s" \\\n' "${DETECT_RAD}" >> "${script_path}"
  fi
  if [[ -n "${DETECT_DIST}" ]]; then
    printf '  -detect_dist "%s" \\\n' "${DETECT_DIST}" >> "${script_path}"
  fi
  if [[ -n "${DETECT_THRE}" ]]; then
    printf '  -detect_thre "%s" \\\n' "${DETECT_THRE}" >> "${script_path}"
  fi

  cat >> "${script_path}" <<EOF
  -slurm_tasks "${SLURM_TASKS}" \\
  -slurm_memory "${SLURM_MEMORY}"
EOF
  chmod +x "${script_path}"
  log "Wrote nextPYP command: ${script_path}"
}

maybe_run_nextpyp() {
  local method script_path
  for method in raw topaz draco; do
    write_nextpyp_script "${method}"
  done

  if [[ "${RUN_NEXTPYP}" != "1" ]]; then
    log "RUN_NEXTPYP=${RUN_NEXTPYP}; wrote command files only"
    return 0
  fi
  if [[ -z "${SCOPE_PIXEL}" || -z "${DETECT_RAD}" || -z "${DETECT_DIST}" ]]; then
    echo "RUN_NEXTPYP=1 requires SCOPE_PIXEL, DETECT_RAD, and DETECT_DIST" >&2
    exit 2
  fi
  if ! command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
    echo "Container runtime not found on PATH: ${CONTAINER_RUNTIME}" >&2
    exit 2
  fi
  if [[ ! -f "${PYP_SIF}" ]]; then
    echo "PYP container not found: ${PYP_SIF}" >&2
    exit 2
  fi

  for method in raw topaz draco; do
    script_path="${RUN_DIR}/nextpyp/${method}/run_nextpyp.sh"
    if [[ -x "${script_path}" ]]; then
      log "Executing nextPYP for ${method}"
      "${script_path}" > "${RUN_DIR}/logs/nextpyp_${method}.log" 2>&1
    fi
  done
}

log "Creating run directory: ${RUN_DIR}"
stage_raw_inputs
stage_condition_inputs topaz "${TOPAZ_DIR}" "${TOPAZ_SUFFIX}"
stage_condition_inputs draco "${DRACO_DIR}" "${DRACO_SUFFIX}"
write_resolved_config
run_dog_benchmark
maybe_run_nextpyp

log "Benchmark complete: ${RUN_DIR}"
log "Review ${RUN_DIR}/dog/dog_summary.tsv and ${RUN_DIR}/dog/overlays/*.png"
