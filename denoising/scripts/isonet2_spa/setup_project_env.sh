#!/usr/bin/env bash
# Create/install the IsoNet2 smoke-test environment entirely under project storage.
#
# This intentionally avoids ~/.conda, ~/.cache/pip, and ~/.local because HPC home
# quotas are small. Run this from any shell with conda available, for example after
# loading the cluster miniforge/anaconda module.

set -euo pipefail

PROJECT_ROOT="/nfs/bartesaghilab2/zy223"
DENOISING_ROOT="${PROJECT_ROOT}/denoising"
ENV_PREFIX="${PROJECT_ROOT}/conda_envs/isonet2_environment"
CONDA_PKGS="${PROJECT_ROOT}/conda_pkgs"
PIP_CACHE="${PROJECT_ROOT}/.cache/pip"
PYTHONUSERBASE_DIR="${PROJECT_ROOT}/.local"
ISONET_REPO="${DENOISING_ROOT}/external/IsoNet2"

mkdir -p "${CONDA_PKGS}" "${PIP_CACHE}" "${PYTHONUSERBASE_DIR}" "$(dirname "${ENV_PREFIX}")"

export CONDA_PKGS_DIRS="${CONDA_PKGS}"
export PIP_CACHE_DIR="${PIP_CACHE}"
export PYTHONUSERBASE="${PYTHONUSERBASE_DIR}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Load a recent cluster conda/miniforge/mambaforge module first." >&2
  exit 1
fi

CONDA_VERSION="$(conda --version | awk '{print $2}')"
CONDA_MAJOR="$(printf '%s' "${CONDA_VERSION}" | cut -d. -f1)"
if [ "${CONDA_MAJOR}" -lt 22 ] 2>/dev/null; then
  cat >&2 <<EOF
ERROR: conda ${CONDA_VERSION} is too old for modern conda-forge/PyTorch metadata.

You appear to be using an old Anaconda module, e.g. Anaconda3/3.5.2. That can
fail during solving with errors like:
  CondaValueError: Malformed version string '~'

Use a newer Miniforge/Mambaforge/Anaconda module if available, for example:
  module purge
  module avail Anaconda Miniforge Mambaforge miniconda
  module load <newer-conda-or-miniforge-module>

Then rerun:
  bash ${DENOISING_ROOT}/scripts/isonet2_spa/setup_project_env.sh
EOF
  exit 1
fi

if [ ! -d "${ISONET_REPO}" ]; then
  echo "ERROR: IsoNet2 repo not found: ${ISONET_REPO}" >&2
  exit 1
fi

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "Creating project-local conda environment: ${ENV_PREFIX}"
  conda create --yes --prefix "${ENV_PREFIX}" \
    python=3.10 \
    pytorch=2.1.1 \
    pytorch-cuda=12.1 \
    numpy \
    scipy \
    mrcfile \
    starfile \
    matplotlib \
    -c pytorch -c nvidia -c conda-forge
else
  echo "Environment already exists: ${ENV_PREFIX}"
fi

echo "Installing IsoNet2 editable into project-local environment"
conda run --prefix "${ENV_PREFIX}" python -m pip install --no-user --no-build-isolation -e "${ISONET_REPO}"

echo "Verifying environment"
conda run --prefix "${ENV_PREFIX}" python - <<'PY'
import importlib.util
import sys

mods = ["torch", "numpy", "mrcfile", "starfile", "matplotlib"]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing modules after install: {missing}")

import torch
print("python", sys.executable)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

cat <<EOF

Done.

Activate with:
  conda activate ${ENV_PREFIX}

Before installing more packages, keep caches project-local:
  export CONDA_PKGS_DIRS=${CONDA_PKGS}
  export PIP_CACHE_DIR=${PIP_CACHE}
  export PYTHONUSERBASE=${PYTHONUSERBASE_DIR}
  export PYTHONNOUSERSITE=1

Run a smoke-test dry run:
  python ${DENOISING_ROOT}/scripts/isonet2_spa/run_smoke_pipeline.py \
    ${DENOISING_ROOT}/data/fuller_subset \
    --limit 3 \
    --dry-run-predict
EOF
