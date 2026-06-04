# Prestin denoiser fine-tuning

This directory contains the planning and future implementation scaffold for Prestin-specific denoiser fine-tuning, starting with Topaz and leaving explicit hooks for DRACO masking requirements.

The goal is to keep model fine-tuning work separate from the current pretrained denoising tests in `zy223/denoising`.

## Scope

Initial scope:

1. prepare paired even/odd SPA micrograph datasets,
2. define reproducible train/validation/test splits,
3. establish evaluation metrics before tuning model hyperparameters,
4. run clean pretrained-vs-fine-tuned Topaz comparisons,
5. preserve enough metadata to later add DRACO fine-tuning and mask-aware evaluation.

Out of scope for the first implementation pass:

- launching large HPC jobs automatically,
- copying raw/preprocessed datasets into this directory,
- overwriting source micrographs or preprocessing outputs,
- treating visual denoising quality as sufficient evidence of biological improvement.

## Proposed layout

```text
zy223/finetuning/
  README.md
  configs/
    topaz_finetune_pilot.yaml      # pilot config template; no hardcoded data paths
  docs/
    topaz_finetuning_plan.md       # implementation plan and evaluation design
  data/
    manifests/                     # future TSV/CSV manifests, split files, metadata
  scripts/
    prepare_pairs.py               # future: build/validate even/odd pair manifests
    split_pairs.py                 # future: train/val/test split by micrograph
    run_topaz_baseline.py          # future: pretrained inference launcher
    run_topaz_finetune.py          # future: Topaz training launcher
    evaluate_denoising.py          # future: metric computation and plots
  results/
    topaz_pretrained_pilot/         # future generated outputs; keep out of git
    topaz_finetuned_pilot/          # future generated outputs; keep out of git
```

Generated denoised micrographs, checkpoints, and large result arrays should remain untracked and should not be committed.

## Current sample dataset

Use this external read-only sample dataset path for the initial pilot; do not import or copy it here:

```text
/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-test-WNXj0kBZzgoPdMbE/sp-preprocessing-ghcTLDemYJJ8gcXE/mrc
```

It should contain complete micrographs plus even/odd split files named with `_half1` and `_half2`.

## Start here

Read the implementation plan:

```text
zy223/finetuning/docs/topaz_finetuning_plan.md
```

Use the pilot config template as the first concrete configuration:

```text
zy223/finetuning/configs/topaz_finetune_pilot.yaml
```

## Script workflow

From the workspace root or `zy223`, generate a pilot manifest and split files without copying data:

```bash
python /nfs/bartesaghilab2/zy223/finetuning/scripts/prepare_pairs.py \
  --limit 300 \
  --skip-metadata

python /nfs/bartesaghilab2/zy223/finetuning/scripts/split_pairs.py \
  --limit 300 \
  --train-count 240 \
  --val-count 30 \
  --test-count 30
```

Run the pretrained Topaz baseline on held-out halves first:

```bash
python /nfs/bartesaghilab2/zy223/finetuning/scripts/run_topaz_baseline.py \
  --manifest /nfs/bartesaghilab2/zy223/finetuning/data/manifests/topaz_pilot/test_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/finetuning/results/topaz_pretrained_pilot \
  --dry-run
```

Prepare and dry-run the fine-tuning launch command:

```bash
python /nfs/bartesaghilab2/zy223/finetuning/scripts/run_topaz_finetune.py \
  --train-manifest /nfs/bartesaghilab2/zy223/finetuning/data/manifests/topaz_pilot/train_pairs.tsv \
  --val-manifest /nfs/bartesaghilab2/zy223/finetuning/data/manifests/topaz_pilot/val_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/finetuning/results/topaz_finetuned_pilot \
  --dry-run
```

This Topaz install exposes Noise2Noise training through `topaz denoise --dir-a ... --dir-b ...`; it does not list a separate `denoise-train` subcommand. Before removing `--dry-run`, confirm `topaz denoise --help` still shows the expected `--dir-a`, `--dir-b`, `--lr`, and `--num-epochs` flags. Use repeated `--extra-arg` for additional Topaz options.

## DRACO Noise2Noise fine-tuning

The public DRACO repo provides model/config/checkpoint loading and inference notebooks, but no denoising fine-tuning API. The project script therefore uses the DRACO architecture/checkpoint directly and implements our own Topaz-style N2N loop over the same even/odd manifests:

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  /nfs/bartesaghilab2/zy223/finetuning/scripts/run_draco_finetune.py \
  --train-manifest /nfs/bartesaghilab2/zy223/finetuning/data/manifests/topaz_pilot/train_pairs.tsv \
  --val-manifest /nfs/bartesaghilab2/zy223/finetuning/data/manifests/topaz_pilot/val_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/finetuning/results/draco_finetuned_pilot \
  --dry-run
```

Default DRACO inputs:

```text
config:     /nfs/bartesaghilab2/zy223/denoising/external/draco/notebook/denoise.yaml
checkpoint: /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt
```

The script trains only the N2N term for now: random aligned crop from one half predicts the paired crop from the other half with patch-space MSE. It does not yet implement DRACO's paper `L_recon` term from complete micrographs.

Note: the Python 3.9 DRACO venv currently fails on DRACO's Python 3.10+ type syntax. Use `/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python` for this trainer unless the Python 3.9 environment is repaired.

## Environment

Use the project-local Topaz environment:

```bash
source /nfs/bartesaghilab2/zy223/venvs/topaz/bin/activate
```

The installed Topaz CLI has been checked: denoising training is handled by `topaz denoise` with `--dir-a` and `--dir-b`. Re-check before long jobs:

```bash
topaz denoise --help
```

Exact command flags should still be derived from the installed Topaz version rather than assumed for future environments.
