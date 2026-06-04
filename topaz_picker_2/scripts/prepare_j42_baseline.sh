#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

run_dir="${RUN_DIR:-${project_root}/runs/j42_baseline}"
metadata_dir="${METADATA_DIR:-${run_dir}/metadata}"
micrographs_dir="${MICROGRAPHS_DIR:-/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs}"
particles_dir="${PARTICLES_DIR:-/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles}"
particles_cs="${PARTICLES_CS:-}"
downsample_factor="${DOWNSAMPLE_FACTOR:-1}"
test_fraction="${TEST_FRACTION:-0.2}"
seed="${SEED:-0}"

mkdir -p "${metadata_dir}" "${run_dir}/models" "${run_dir}/predictions" "${run_dir}/eval"

cmd=(
  python "${script_dir}/cryosparc_cs_to_topaz.py"
  --micrographs-dir "${micrographs_dir}"
  --particles-dir "${particles_dir}"
  --output-dir "${metadata_dir}"
  --downsample-factor "${downsample_factor}"
  --test-fraction "${test_fraction}"
  --seed "${seed}"
)

if [[ -n "${particles_cs}" ]]; then
  cmd+=(--particles-cs "${particles_cs}")
fi

"${cmd[@]}"
