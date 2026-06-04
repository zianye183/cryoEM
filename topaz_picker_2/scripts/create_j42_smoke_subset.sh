#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

base_run_dir="${BASE_RUN_DIR:-${project_root}/runs/j42_baseline}"
smoke_run_dir="${SMOKE_RUN_DIR:-${project_root}/runs/j42_smoke}"
base_metadata_dir="${BASE_METADATA_DIR:-${base_run_dir}/metadata}"
smoke_metadata_dir="${SMOKE_METADATA_DIR:-${smoke_run_dir}/metadata}"
train_image_lines="${TRAIN_IMAGE_LINES:-9}"
test_image_lines="${TEST_IMAGE_LINES:-3}"

mkdir -p "${smoke_metadata_dir}" "${smoke_run_dir}/models" "${smoke_run_dir}/predictions" "${smoke_run_dir}/eval"

required_inputs=(
  "${base_metadata_dir}/train_images.tsv"
  "${base_metadata_dir}/test_images.tsv"
  "${base_metadata_dir}/train_particles.tsv"
  "${base_metadata_dir}/test_particles.tsv"
)

for input_path in "${required_inputs[@]}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "Missing baseline metadata input: ${input_path}" >&2
    echo "Run prepare_j42_baseline.sh first, or set BASE_RUN_DIR/BASE_METADATA_DIR." >&2
    exit 1
  fi
done

head -n "${train_image_lines}" "${base_metadata_dir}/train_images.tsv" \
  > "${smoke_metadata_dir}/train_images.tsv"
head -n "${test_image_lines}" "${base_metadata_dir}/test_images.tsv" \
  > "${smoke_metadata_dir}/test_images.tsv"

awk 'NR==FNR { if (FNR > 1) keep[$1]=1; next } FNR==1 || keep[$1]' \
  "${smoke_metadata_dir}/train_images.tsv" \
  "${base_metadata_dir}/train_particles.tsv" \
  > "${smoke_metadata_dir}/train_particles.tsv"

awk 'NR==FNR { if (FNR > 1) keep[$1]=1; next } FNR==1 || keep[$1]' \
  "${smoke_metadata_dir}/test_images.tsv" \
  "${base_metadata_dir}/test_particles.tsv" \
  > "${smoke_metadata_dir}/test_particles.tsv"

cp "${smoke_metadata_dir}/train_particles.tsv" "${smoke_metadata_dir}/train_particles.txt"
cp "${smoke_metadata_dir}/test_particles.tsv" "${smoke_metadata_dir}/test_particles.txt"

python - "${smoke_metadata_dir}" "${base_metadata_dir}" <<'PY'
import csv
import json
import sys
from pathlib import Path

metadata_dir = Path(sys.argv[1])
base_metadata_dir = Path(sys.argv[2])

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

train_images = read_rows(metadata_dir / "train_images.tsv")
test_images = read_rows(metadata_dir / "test_images.tsv")
train_particles = read_rows(metadata_dir / "train_particles.txt")
test_particles = read_rows(metadata_dir / "test_particles.txt")

expected_particle_fields = ["image_name", "x_coord", "y_coord"]
for name, rows in {
    "train_particles.txt": train_particles,
    "test_particles.txt": test_particles,
}.items():
    if not rows:
        raise SystemExit(f"{name} has no particle rows")
    if list(rows[0].keys()) != expected_particle_fields:
        raise SystemExit(f"{name} columns must be {expected_particle_fields}, got {list(rows[0].keys())}")

summary = {
    "num_train_micrographs": len(train_images),
    "num_test_micrographs": len(test_images),
    "num_train_particles": len(train_particles),
    "num_test_particles": len(test_particles),
    "source_metadata_dir": str(base_metadata_dir),
}
(metadata_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
