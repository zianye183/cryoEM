# Topaz Picker

This workspace is for training and evaluating a Topaz particle picker model using CryoSPARC-exported micrographs and particle picks.

## Current CryoSPARC Export Inputs

Use these exported groups as the current input subset for Topaz picker training/evaluation:

- Micrographs:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs`
- Corresponding particle picks:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles`

These paths should be treated as read-only inputs. Do not move, rename, delete, or overwrite files in these export directories.

## Current Baseline Decision

Start with **Option 1**: train/evaluate Topaz on the exported CryoSPARC micrographs exactly as provided in `J42_micrographs`, with `downsample_factor = 1` for coordinate conversion.

This first baseline is intended to validate the CryoSPARC `.cs` to Topaz TSV conversion and coordinate mapping before introducing additional preprocessing.

Planned follow-up: **Option 2** will train/evaluate on Topaz-preprocessed/downsampled versions of the same micrographs. When moving to Option 2, particle coordinates must be divided by the Topaz preprocessing downsample factor.

## Intended Workflow

1. Inspect the CryoSPARC export contents and identify the `.cs` files/metadata fields for micrographs and particles.
2. Convert CryoSPARC particle coordinates into Topaz-compatible TSV files:
   - `images.tsv`-style files with `image_name` and `path`
   - `particles.tsv`-style files with `image_name`, `x_coord`, and `y_coord`
3. Split by micrograph into train/test sets.
4. Train a Topaz picker model using the project-local Topaz environment:
   `/nfs/bartesaghilab2/zy223/venvs/topaz`
5. Evaluate held-out predictions using Topaz precision-recall tooling and visual inspection.

## CryoSPARC to Topaz coordinate conversion

CryoSPARC particle `.cs` files commonly store coordinates as fractional image positions:

- `location/center_x_frac`
- `location/center_y_frac`
- `location/micrograph_shape`

CryoSPARC shape is expected to be `[ny, nx]`.

Convert to pixel coordinates with:

```python
x = center_x_frac * nx
y = center_y_frac * ny
```

If Topaz training uses downsampled/preprocessed images, divide coordinates by the same downsample factor:

```python
x_topaz = x / downsample_factor
y_topaz = y / downsample_factor
```

For the current `j42_baseline`, use:

```text
downsample_factor = 1
```

## Generate Topaz metadata

Run from the project root on the HPC:

```bash
cd /nfs/bartesaghilab2/zy223/topaz_picker

python scripts/cryosparc_cs_to_topaz.py \
  --micrographs-dir /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs \
  --particles-dir /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles \
  --output-dir runs/j42_baseline/metadata \
  --downsample-factor 1 \
  --test-fraction 0.2 \
  --seed 0
```

The converter automatically creates the output directory if needed.

If multiple `.cs` files are found in the particle export directory, the converter fails and asks for an explicit file. Re-run with:

```bash
python scripts/cryosparc_cs_to_topaz.py \
  --micrographs-dir /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs \
  --particles-dir /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles \
  --particle-cs-name PARTICLES_FILE.cs \
  --output-dir runs/j42_baseline/metadata \
  --downsample-factor 1 \
  --test-fraction 0.2 \
  --seed 0
```

Expected outputs:

```text
runs/j42_baseline/metadata/images_all.tsv
runs/j42_baseline/metadata/particles_all.tsv
runs/j42_baseline/metadata/train_images.tsv
runs/j42_baseline/metadata/train_particles.tsv
runs/j42_baseline/metadata/test_images.tsv
runs/j42_baseline/metadata/test_particles.tsv
runs/j42_baseline/metadata/summary.json
```

The split is by micrograph, not by particle, to avoid train/test leakage.

## Train Topaz baseline

Create run output folders:

```bash
mkdir -p runs/j42_baseline/models runs/j42_baseline/eval runs/j42_baseline/predictions
```

Then train:

```bash
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz train \
  --train-images runs/j42_baseline/metadata/train_images.tsv \
  --train-targets runs/j42_baseline/metadata/train_particles.tsv \
  --test-images runs/j42_baseline/metadata/test_images.tsv \
  --test-targets runs/j42_baseline/metadata/test_particles.tsv \
  --format coord \
  --method GE-binomial \
  --model resnet8 \
  -n 30 \
  -r 10 \
  --num-epochs 10 \
  --save-prefix runs/j42_baseline/models/prestin_topaz \
  -o runs/j42_baseline/eval/train_curve.tsv
```

Initial parameter choices:

| Parameter | Initial value | Reason |
|---|---:|---|
| `--method` | `GE-binomial` | Positive-unlabeled learning default for Topaz picker training |
| `--model` | `resnet8` | Lightweight standard Topaz model |
| `-n` | `30` | Approximate expected particles per micrograph for the first baseline |
| `-r` | `10` | Starting radius for full-size, non-downsampled images |
| `--num-epochs` | `10` | Small baseline run before larger sweeps |

Before relying on this command, verify installed Topaz flags on the HPC:

```bash
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz train --help
```

## Inference on held-out micrographs

After training, use `topaz extract` with the saved model. The exact model filename depends on the Topaz version and saved epoch naming. Check `runs/j42_baseline/models/` after training.

Command shape:

```bash
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz extract \
  -m runs/j42_baseline/models/prestin_topaz_epoch10.sav \
  -r 10 \
  -t -6 \
  -o runs/j42_baseline/predictions/test_predictions.tab \
  TEST_MICROGRAPH_1.mrc TEST_MICROGRAPH_2.mrc TEST_MICROGRAPH_3.mrc
```

Use paths from:

```text
runs/j42_baseline/metadata/test_images.tsv
```

A future helper script can read `test_images.tsv` and call `topaz extract` without manually expanding image paths.

## Held-out precision-recall evaluation

Compare held-out Topaz predictions to held-out CryoSPARC pseudo-labels:

```bash
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/topaz precision_recall_curve \
  --predicted runs/j42_baseline/predictions/test_predictions.tab \
  --targets runs/j42_baseline/metadata/test_particles.tab \
  -r 10 \
  > runs/j42_baseline/eval/pr_curve.tsv
```

These targets are CryoSPARC-derived pseudo-labels, not ground truth. Use precision-recall as a useful first metric, but also visually inspect picks and evaluate downstream 2D classes/reconstruction quality.
