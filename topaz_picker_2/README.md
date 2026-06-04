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

This first baseline is intended to validate the CryoSPARC `.cs` to Topaz metadata conversion and coordinate mapping before introducing additional preprocessing.

Planned follow-up: **Option 2** will train/evaluate on Topaz-preprocessed/downsampled versions of the same micrographs. When moving to Option 2, particle coordinates must be divided by the Topaz preprocessing downsample factor.

## Intended Workflow

1. Inspect the CryoSPARC export contents and identify the `.cs` files/metadata fields for micrographs and particles.
2. Convert CryoSPARC particle coordinates into Topaz-compatible metadata files:
   - `images.tsv` with `image_name` and `path`
   - `particles.txt` with `image_name`, `x_coord`, and `y_coord`
3. Split by micrograph into train/test sets.
4. Train a Topaz picker model using the project-local Topaz environment:
   `/nfs/bartesaghilab2/zy223/venvs/topaz`
5. Evaluate held-out predictions using Topaz precision-recall tooling and visual inspection.

## Convert CryoSPARC `.cs` Exports to Topaz Metadata

Run this after uploading the code to the HPC:

```bash
topaz_picker/scripts/prepare_j42_baseline.sh
```

If the particle export directory contains multiple compatible `.cs` files, pass the exact one:

```bash
PARTICLES_CS=/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles/particles_accepted.cs \
  topaz_picker/scripts/prepare_j42_baseline.sh
```

The converter writes:

- `images_all.tsv`
- `particles_all.tsv`
- `particles_all.txt`
- `train_images.tsv`
- `train_particles.tsv`
- `train_particles.txt`
- `test_images.tsv`
- `test_particles.tsv`
- `test_particles.txt`
- `summary.json`

The `.tsv` particle files are kept for inspection. The `.txt` particle files are the Topaz training/evaluation targets because this Topaz install detects coordinate format from file extension and recognizes `.txt` as `coord`, not `.tsv` or `.coord`.

The train/test split is by micrograph, not by particle.

The script validates that:

- the particle `.cs` file has the required CryoSPARC location fields,
- exported micrograph paths can be resolved under `--micrographs-dir`,
- fractional coordinates are finite,
- converted coordinates stay inside image bounds,
- Topaz coordinate targets have exact columns `image_name`, `x_coord`, and `y_coord`,
- train and test target files are non-empty.

## End-to-End Baseline Run

Run the baseline in this order:

```bash
topaz_picker/scripts/prepare_j42_baseline.sh
topaz_picker/scripts/train_j42_baseline.sh
topaz_picker/scripts/evaluate_j42_baseline.sh
```

The scripts use these defaults:

| Variable | Default |
|---|---|
| `TOPAZ_VENV` | `/nfs/bartesaghilab2/zy223/venvs/topaz` |
| `RUN_DIR` | `topaz_picker/runs/j42_baseline` under this project |
| `MICROGRAPHS_DIR` | `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs` |
| `PARTICLES_DIR` | `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles` |
| `DOWNSAMPLE_FACTOR` | `1` |
| `TEST_FRACTION` | `0.2` |
| `SEED` | `0` |
| `TOPAZ_METHOD` | `GE-binomial` |
| `TOPAZ_MODEL` | `resnet8` |
| `NUM_PARTICLES` | Derived from train particles per micrograph in `summary.json` |
| `RADIUS` | `10` |
| `NUM_EPOCHS` | `10` |
| `TOPAZ_PRELOAD` | `auto` |
| `THRESHOLD` | `-6` |

Example parameter override:

```bash
NUM_PARTICLES=50 RADIUS=15 NUM_EPOCHS=20 topaz_picker/scripts/train_j42_baseline.sh
NUM_EPOCHS=20 RADIUS=15 topaz_picker/scripts/evaluate_j42_baseline.sh
```

If Topaz saves a different checkpoint name than the script expects, pass it explicitly:

```bash
MODEL_PATH=topaz_picker/runs/j42_baseline/models/prestin_topaz_epoch9.sav \
  topaz_picker/scripts/evaluate_j42_baseline.sh
```

The evaluation script writes:

- `predictions/test_predictions.tsv`
- `eval/pr_curve.tsv`
- `metadata/test_image_paths.txt`
- `eval/evaluate.log`

Evaluation extracts predictions first, then computes the precision-recall curve. This avoids Topaz's brittle inline scoring path in `topaz extract --targets`, which can fail on weak smoke models with `ValueError: need at least one array to concatenate`. If a one-epoch smoke model produces no predictions at the default `THRESHOLD=-6`, rerun with a lower threshold:

```bash
RUN_DIR=/nfs/bartesaghilab2/zy223/topaz_picker_2/runs/j42_smoke \
NUM_EPOCHS=1 \
THRESHOLD=-10 \
scripts/evaluate_j42_baseline.sh
```

## Visual Coordinate Checks

Before evaluation, verify that converted CryoSPARC target coordinates land on the expected micrograph locations:

```bash
python scripts/plot_topaz_overlay.py \
  --images runs/j42_smoke/metadata/test_images.tsv \
  --targets runs/j42_smoke/metadata/test_particles.txt \
  --output-dir runs/j42_smoke/eval/target_overlays \
  --max-images 2
```

After evaluation, overlay Topaz predictions against the held-out target labels:

```bash
python scripts/plot_topaz_overlay.py \
  --images runs/j42_smoke/metadata/test_images.tsv \
  --targets runs/j42_smoke/metadata/test_particles.txt \
  --predictions runs/j42_smoke/predictions/test_predictions.tsv \
  --output-dir runs/j42_smoke/eval/prediction_overlays \
  --max-images 2
```

The overlay convention is:

- red open circles: target labels from CryoSPARC,
- cyan crosses: Topaz predictions.

Use `--crop-radius 512` to zoom around the labels/predictions instead of plotting the full micrograph.

The training script prints and records a setup block before handing off to Topaz:

- train/test micrograph counts,
- train/test particle counts,
- `NUM_EPOCHS`,
- Topaz method/model/radius/preload settings,
- target paths and output paths.

The same output is saved to `eval/train.log`. Topaz does not expose per-micrograph preprocessing progress in this install, so the most useful wrapper signal is the count of micrographs that Topaz is about to load before the first epoch output appears.

## Raw Topaz Training Command Shape

The training script wraps this command:

```bash
topaz train \
  --train-images topaz_picker/runs/j42_baseline/metadata/train_images.tsv \
  --train-targets topaz_picker/runs/j42_baseline/metadata/train_particles.txt \
  --test-images topaz_picker/runs/j42_baseline/metadata/test_images.tsv \
  --test-targets topaz_picker/runs/j42_baseline/metadata/test_particles.txt \
  --format coord \
  --method GE-binomial \
  --model resnet8 \
  -n 30 \
  -r 10 \
  --num-epochs 10 \
  --save-prefix topaz_picker/runs/j42_baseline/models/prestin_topaz \
  -o topaz_picker/runs/j42_baseline/eval/train_curve.tsv
```

Before running a full training job, verify the exact installed flags with:

```bash
topaz train --help
```

## Smoke Test Subset

After preparing the baseline metadata, create a small smoke run with:

```bash
topaz_picker/scripts/create_j42_smoke_subset.sh
```

This creates `topaz_picker/runs/j42_smoke/metadata` with 8 train micrographs and 2 test micrographs by default, plus matching `train_particles.txt` and `test_particles.txt` target files.

If creating a subset manually, preserve the particle header with `FNR==1`, not `NR==1`. `NR==1` only matches the header of the first input file and produces headerless particle target files, which Topaz reports as `KeyError: 'image_name'`.

Run a one-epoch smoke training job on a GPU allocation:

```bash
RUN_DIR=/nfs/bartesaghilab2/zy223/topaz_picker_2/runs/j42_smoke \
NUM_EPOCHS=1 \
NUM_PARTICLES=30 \
scripts/train_j42_baseline.sh
```

## Notes

- CryoSPARC particle `.cs` files commonly store coordinates as fractional image positions, e.g. `location/center_x_frac` and `location/center_y_frac`.
- Convert to pixel coordinates using the corresponding micrograph shape:
  - \(x = x_{\mathrm{frac}} \cdot n_x\)
  - \(y = y_{\mathrm{frac}} \cdot n_y\)
- Topaz coordinate tables use extensionless `image_name` values, even when the image paths point to `.mrc`, `.tif`, or other image files.
- If Topaz training uses downsampled/preprocessed images, divide coordinates by the same downsample factor.
