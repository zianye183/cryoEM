# IsoNet2 SPA smoke-test scripts

Prediction-only wrapper for trying a pretrained 3D IsoNet2 model on 2D SPA micrographs.

This is a qualitative software/readout test only. Each 2D micrograph is repeated along fake Z to create a pseudo-volume, so the result is not a scientifically validated SPA denoising method.

## Default paths

- Model: `/nfs/bartesaghilab/nextpyp/workshop_dhvi/10453/isonet_network_isonet2-n2n_unet-medium_96_full.pt`
- Inputs: `/nfs/bartesaghilab2/zy223/denoising/data/fuller_subset/*.mrc`
- Work dir: `/nfs/bartesaghilab2/zy223/denoising/experiments/isonet2_spa_smoke`

## Environment

Home-directory conda caches on the HPC are small, so create/install IsoNet2 under project storage instead of `/hpc/home/zy223/.conda` or `/hpc/home/zy223/.local`.

Load a conda/miniforge module if needed, then run:

```bash
bash /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/setup_project_env.sh
```

The setup script uses:

```text
Environment prefix: /nfs/bartesaghilab2/zy223/conda_envs/isonet2_environment
Conda package cache: /nfs/bartesaghilab2/zy223/conda_pkgs
Pip cache: /nfs/bartesaghilab2/zy223/.cache/pip
Python userbase: /nfs/bartesaghilab2/zy223/.local
```

Activate with the full prefix, not a named home environment:

```bash
conda activate /nfs/bartesaghilab2/zy223/conda_envs/isonet2_environment
```

Before installing any additional packages, keep caches project-local:

```bash
export CONDA_PKGS_DIRS=/nfs/bartesaghilab2/zy223/conda_pkgs
export PIP_CACHE_DIR=/nfs/bartesaghilab2/zy223/.cache/pip
export PYTHONUSERBASE=/nfs/bartesaghilab2/zy223/.local
export PYTHONNOUSERSITE=1
```

Then verify:

```bash
isonet.py --help
python -c "import torch, mrcfile, numpy; print(torch.cuda.is_available())"
```

## Minimal run

Prepare pseudo-volumes and STAR, and print the IsoNet2 prediction command without launching prediction:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/run_smoke_pipeline.py \
  /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset \
  --limit 3 \
  --crop-size 1024 \
  --dry-run-predict
```

Run prediction:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/run_smoke_pipeline.py \
  /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset \
  --limit 3 \
  --crop-size 1024 \
  --run-predict \
  --gpu-id 0
```

Defaults intentionally pass `--apply_mw_x1 False` and `--isCTFflipped True` to avoid tomographic missing-wedge and CTF sign filtering in this first SPA smoke test. The driver also defaults to `--crop-size 1024`; use a small crop for the first run because full-size repeated-Z micrographs can be killed by RAM/VRAM limits.

## Step-by-step scripts

1. `inspect_checkpoint.py` writes checkpoint metadata.
2. `make_pseudo_volumes.py` creates repeated-Z MRC volumes.
3. `make_spa_star.py` writes the IsoNet2 STAR file with duplicated half1/half2 paths.
4. `run_predict.py` launches `isonet.py predict`.
5. `extract_slices.py` extracts center, mean-Z, and z-std MRCs.
6. `qc_outputs.py` writes `image_stats.csv` and optional PNG panels.
