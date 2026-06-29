# cryoEM project map

This repository is a working area for Prestin cryo-EM denoising, denoiser fine-tuning, particle-picking benchmarks, Topaz picker training, and vesicle-mask filtering. Most large datasets and generated outputs live on the HPC under `/nfs/...`; this repo should mainly track scripts, configs, manifests, tests, and lightweight QC artifacts.

## Path conventions

- Local checkout: `/Users/alanye/Documents/CS+ 2026/cryoEM`
- HPC project mirror used by most scripts: `/nfs/bartesaghilab2/zy223`
- Treat CryoSPARC exports, source micrographs, pretrained model weights, and generated run outputs as read-only or generated data unless a script explicitly stages symlinks.
- Many older docs say `zy223/<area>/...`; in this checkout that usually means the corresponding top-level directory, e.g. `zy223/denoising` maps to `denoising/` locally and `/nfs/bartesaghilab2/zy223/denoising` on the HPC.

## Repo Structure

```text
.
  README.md

  denoising/                         # Topaz-Denoise, DRACO, and IsoNet2 SPA smoke-test workflows
    README.md
    configs/
      denoising_test_config.yaml     # small-subset Topaz/DRACO config
      isonet2_spa_smoke.yaml         # IsoNet2 SPA smoke-test config
      draco_venv_requirements.txt    # DRACO environment notes/requirements
    data/
      subset/                        # small symlinked pilot subset on working copies/HPC
      fuller_subset/                 # fuller Prestin subset used by downstream presets
    envs/
      topaz-requirements.txt
    experiments/
      isonet2_spa_smoke/             # generated IsoNet2 smoke-test manifests/QC
    external/
      draco/                         # external DRACO checkout; do not treat as project source
    notebooks/
      denoising_initial_tests.ipynb  # older exploratory notebook
    scripts/
      denoising_initial_tests.py     # cell-style exploratory workflow
      check_env.py                   # dependency/environment check
      make_subset.py                 # symlink/copy small MRC subset
      inspect_mrc_headers.py         # MRC shape/header inspection
      build_commands.py              # print conservative runner commands
      run_topaz_pretrained.py        # Topaz pretrained denoise launcher
      run_topaz_retraining.py        # Topaz Noise2Noise retraining launcher
      setup_draco.py                 # DRACO clone/checkpoint setup
      run_draco_pretrained.py        # DRACO pretrained denoise launcher
      draco_denoise_adapter.py       # local DRACO batch adapter
      compare_topaz_draco_mosaics.py # visual comparison/QC helper
      isonet2_spa/                   # IsoNet2 pseudo-volume SPA smoke-test scripts
    tests/

  finetuning/                        # Prestin-specific denoiser fine-tuning
    README.md
    configs/
      topaz_finetune_pilot.yaml      # pilot Topaz/DRACO fine-tuning config
    data/
      manifests/topaz_pilot/         # generated TSV manifests and split summaries
    docs/
      topaz_finetuning_plan.md
    scripts/
      prepare_pairs.py               # build even/odd pair manifest
      split_pairs.py                 # train/val/test split generation
      run_topaz_baseline.py          # pretrained Topaz baseline on held-out pairs
      run_topaz_finetune.py          # Topaz fine-tuning launcher
      run_draco_finetune.py          # project-local DRACO N2N trainer
      compare_topaz_checkpoints.py   # checkpoint/output comparison
      visualize_*                    # visual QC panels/triplets

  particle_picking/                  # raw vs denoised particle-picking benchmark
    README.md
    particle_picking_benchmark_plan.md
    configs/
      particle_picking_benchmark_pilot.env
    docs/
      particle_picking_benchmark_plan.md
      pyp_container_usage.md
    scripts/
      run_particle_picking_benchmark.sh # stages symlinks, runs DoG, writes nextPYP commands
      evaluate_dog_picking.py           # DoG/NMS benchmark
      export_dog_picks_to_box.py        # `.box` export for inspection/import
      plot_picking_overlays.py          # overlay PNG generation

  segmentation_integration/          # Vesicle Picker/SAM masks, CryoVIA benchmark, pick filtering
    README.md
    configs/
      prestin_vesicle_picker_settings.ini
    scripts/
      prepare_vesicle_picker_run.py              # generate Vesicle Picker configs/run script
      prepare_slurm_pipeline.py                  # SLURM-oriented pipeline prep
      run_local_vesicle_picker.py                # local Vesicle Picker invocation helper
      filter_cryosparc_picks_by_vesicle_masks.py # mask-band filter for CryoSPARC picks
      visualize_filtered_picks.py                # accepted/rejected pick overlays
      visualize_vesicle_mask_filtering.py        # found-vs-filtered mask QC
      visualize_vesicle_masks.py                 # standalone mask rendering
      run_segmentation_visual_benchmark.py       # CryoVIA vs SAM visual benchmark
      generate_vesicle_membrane_picks.py         # membrane-band pick generation helper
    tests/
    *.py                             # library modules for CryoSPARC IO, masks, filtering, QC, presets

  topaz_picker/                      # earlier Topaz picker baseline scaffold
    README.md
    scripts/
      cryosparc_cs_to_topaz.py
      train_topaz_baseline.sh
      run_topaz_extract_eval.py

  topaz_picker_2/                    # current J42 Topaz picker baseline
    README.md
    scripts/
      prepare_j42_baseline.sh        # convert CryoSPARC exports to Topaz metadata
      create_j42_smoke_subset.sh     # small smoke split
      train_j42_baseline.sh          # Topaz picker training wrapper
      evaluate_j42_baseline.sh       # Topaz extract + PR evaluation wrapper
      plot_topaz_overlay.py          # target/prediction overlay QC
      cryosparc_cs_to_topaz.py       # coordinate conversion
    tests/

  cet_pick/                          # local ignored MiLoPYP/CET picker checkout
    README.md
    AGENTS.md
    docs/
    configs/
    cet_pick/                        # upstream package source
    tests/
    .venv/                           # local environment; not project source

  sample_mask_filter_qc/             # small generated/sample mask-filter QC images and stats
```

Ignore `__pycache__/`, `.DS_Store`, local virtual environments, external tool checkouts, and generated run/result folders when reasoning about project source. A few stray files in `topaz_picker_2/` are named like SLURM flags, e.g. `--mem=64G`; they are not pipeline entry points.

## Key Data Locations

### Denoising inputs and outputs

- Accepted Prestin exposures for larger runs:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J80_exposures_accepted/J3/motioncorrected`
- Earlier pilot source MRCs:
  `/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-1fhwPcd8gQ/sp-preprocessing-ghLbktywzmKi35IA/mrc`
- Small denoising subset:
  `/nfs/bartesaghilab2/zy223/denoising/data/subset`
- Fuller subset used by segmentation presets:
  `/nfs/bartesaghilab2/zy223/denoising/data/fuller_subset`
- Topaz pretrained outputs:
  `/nfs/bartesaghilab2/zy223/denoising/results/topaz_pretrained`
- DRACO pretrained outputs:
  `/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained`
- DRACO fuller-subset outputs:
  `/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained_fuller_subset`

### Fine-tuning inputs and manifests

- Prestin accepted even/odd source data:
  `/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-test-WNXj0kBZzgoPdMbE/sp-preprocessing-gkym1IT0yo4s7y1L/mrc`
- Topaz accepted-exposure manifests:
  `/nfs/bartesaghilab2/zy223/data/finetuning_data/topaz_prestin`

### CryoSPARC exports

- J42 Topaz picker micrographs:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs`
- J42 Topaz picker particles:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles`
- Raw segmentation/pick-filter preset particles:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J71_particles`
- DRACO segmentation/pick-filter preset particles:
  `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J70_particles`
- SARS-CoV-2 SPA micrographs/exposures for `CS-other-datasets`, pixel size `1.061 A`:
  `/nfs/bartesaghilab2/zy223/cryosparc/other_datasets/CS-other-datasets/exports/groups/J4_exposures`

### External tools and environments

- Topaz environment:
  `/nfs/bartesaghilab2/zy223/venvs/topaz`
- DRACO Python 3.11 environment:
  `/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python`
- DRACO checkout:
  `/nfs/bartesaghilab2/zy223/denoising/external/draco`
- DRACO Denoise/B checkpoint:
  `/nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt`
- Vesicle Picker checkout:
  `/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker`
- Default SAM weights:
  `/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker/sam_vit_h_4b8939.pth`
- nextPYP/PYP container:
  `/nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif`
- IsoNet2 SPA environment prefix:
  `/nfs/bartesaghilab2/zy223/conda_envs/isonet2_environment`

## Pipeline Entry Points

### 1. Denoising: Topaz, DRACO, and IsoNet2 smoke tests

Primary docs:

- `denoising/README.md`
- `denoising/scripts/isonet2_spa/README.md`
- `denoising/configs/denoising_test_config.yaml`

Inspect the environment and make a small subset:

```bash
python denoising/scripts/check_env.py

python denoising/scripts/make_subset.py \
  --source /nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-1fhwPcd8gQ/sp-preprocessing-ghLbktywzmKi35IA/mrc \
  --dest /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --n 5 \
  --seed 42 \
  --overwrite

python denoising/scripts/inspect_mrc_headers.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset
```

Run dry-run command generation before model execution:

```bash
python denoising/scripts/build_commands.py \
  --subset /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --results /nfs/bartesaghilab2/zy223/denoising/results
```

Topaz pretrained denoising:

```bash
python denoising/scripts/run_topaz_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/topaz_pretrained \
  --model unet \
  --device 0 \
  --suffix _topaz_denoised \
  --dry-run
```

DRACO setup and pretrained denoising:

```bash
python denoising/scripts/setup_draco.py --dry-run

python denoising/scripts/run_draco_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --repo /nfs/bartesaghilab2/zy223/denoising/external/draco \
  --weights /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained \
  --dry-run
```

IsoNet2 SPA qualitative smoke test:

```bash
bash /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/setup_project_env.sh

python /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/run_smoke_pipeline.py \
  /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset \
  --limit 3 \
  --crop-size 1024 \
  --dry-run-predict
```

Use `--run-predict --gpu-id 0` only after the environment and dry-run command are confirmed.

### 2. Denoiser fine-tuning

Primary docs:

- `finetuning/README.md`
- `finetuning/docs/topaz_finetuning_plan.md`
- `finetuning/configs/topaz_finetune_pilot.yaml`

Prepare paired even/odd manifests and splits:

```bash
python finetuning/scripts/prepare_pairs.py \
  --skip-metadata

python finetuning/scripts/split_pairs.py \
  --limit 0 \
  --train-count 11890 \
  --val-count 1486 \
  --test-count 1486 \
  --visual-count 24
```

Run pretrained Topaz baseline on held-out pairs:

```bash
python finetuning/scripts/run_topaz_baseline.py \
  --manifest /nfs/bartesaghilab2/zy223/data/finetuning_data/topaz_prestin/test_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/finetuning/results/topaz_pretrained_pilot \
  --dry-run
```

Prepare Topaz fine-tuning:

```bash
finetuning/scripts/run_topaz_finetune_prestin.sh
```

The wrapper defaults to `DRY_RUN=1`. Start a real run with:

```bash
DRY_RUN=0 OVERWRITE=1 finetuning/scripts/run_topaz_finetune_prestin.sh
```

Prepare DRACO fine-tuning:

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  finetuning/scripts/run_draco_finetune.py \
  --train-manifest /nfs/bartesaghilab2/zy223/data/finetuning_data/topaz_prestin/train_pairs.tsv \
  --val-manifest /nfs/bartesaghilab2/zy223/data/finetuning_data/topaz_prestin/val_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/finetuning/results/draco_finetuned_pilot \
  --dry-run
```

### 3. Particle-picking benchmark: raw vs Topaz vs DRACO

Primary docs:

- `particle_picking/README.md`
- `particle_picking/docs/particle_picking_benchmark_plan.md`
- `particle_picking/docs/pyp_container_usage.md`
- `particle_picking/configs/particle_picking_benchmark_pilot.env`

Run the pilot benchmark from the repo root:

```bash
bash particle_picking/scripts/run_particle_picking_benchmark.sh \
  particle_picking/configs/particle_picking_benchmark_pilot.env
```

The benchmark stages symlinks into a timestamped run directory, runs the DoG/NMS picker for each available image condition, writes `.box` files and overlay PNGs, and writes nextPYP command files. By default `RUN_NEXTPYP=0`, so nextPYP command files are generated but not executed.

Set `RUN_NEXTPYP=1` in the config only after `SCOPE_PIXEL`, `DETECT_RAD`, `DETECT_DIST`, the container path, and `/nfs` bind mounts are confirmed.

### 4. Topaz particle picker baseline

Current baseline docs and scripts live in `topaz_picker_2/`. The older `topaz_picker/` directory has an earlier scaffold.

Primary docs:

- `topaz_picker_2/README.md`
- `topaz_picker_2/scripts/cryosparc_cs_to_topaz.py`
- `topaz_picker_2/scripts/prepare_j42_baseline.sh`
- `topaz_picker_2/scripts/train_j42_baseline.sh`
- `topaz_picker_2/scripts/evaluate_j42_baseline.sh`

Run the J42 baseline in order:

```bash
cd /nfs/bartesaghilab2/zy223/topaz_picker_2

scripts/prepare_j42_baseline.sh
scripts/train_j42_baseline.sh
scripts/evaluate_j42_baseline.sh
```

Useful overrides:

```bash
NUM_PARTICLES=50 RADIUS=15 NUM_EPOCHS=20 scripts/train_j42_baseline.sh
NUM_EPOCHS=20 RADIUS=15 scripts/evaluate_j42_baseline.sh
```

Create and run a smaller smoke subset:

```bash
scripts/create_j42_smoke_subset.sh

RUN_DIR=/nfs/bartesaghilab2/zy223/topaz_picker_2/runs/j42_smoke \
NUM_EPOCHS=1 \
NUM_PARTICLES=30 \
scripts/train_j42_baseline.sh
```

Visual coordinate checks:

```bash
python scripts/plot_topaz_overlay.py \
  --images runs/j42_smoke/metadata/test_images.tsv \
  --targets runs/j42_smoke/metadata/test_particles.txt \
  --output-dir runs/j42_smoke/eval/target_overlays \
  --max-images 2
```

### 5. Segmentation and vesicle-mask pick filtering

Primary docs:

- `segmentation_integration/README.md`
- `segmentation_integration/dataset_presets.py`
- `segmentation_integration/configs/prestin_vesicle_picker_settings.ini`

Standard dataset presets:

- `raw`: particles from `J71_particles`, micrographs from `/nfs/bartesaghilab2/zy223/denoising/data/fuller_subset`
- `draco`: particles from `J70_particles`, micrographs from `/nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained_fuller_subset`

Prepare a standard local-micrograph Vesicle Picker + CryoSPARC filter run:

```bash
python segmentation_integration/scripts/prepare_vesicle_picker_run.py \
  --dataset raw \
  --local-micrographs \
  --output-dir /nfs/bartesaghilab2/zy223/segmentation_runs/raw \
  --segmentation-preset default \
  --filter-output-dir /nfs/bartesaghilab2/zy223/segmentation_runs/raw/filtered_particles
```

Use `--dataset draco` for DRACO-denoised micrographs. Use `--segmentation-preset recall_heavy` for the higher-recall SAM parameter preset.

Run the generated end-to-end script:

```bash
bash /nfs/bartesaghilab2/zy223/segmentation_runs/raw/run_end_to_end.sh
```

If Vesicle Picker masks already exist, run only the CryoSPARC particle filter:

```bash
python segmentation_integration/scripts/filter_cryosparc_picks_by_vesicle_masks.py \
  --dataset raw \
  --vesicle-mask-dir /path/to/filter_vesicles_outputs \
  --output-dir /path/to/filter_output \
  --psize-a 1.068 \
  --downsample 4 \
  --membrane-band-a 40 \
  --max-overlays 10
```

Regenerate overlays from existing filter outputs:

```bash
python segmentation_integration/scripts/visualize_filtered_picks.py \
  --particles-cs /path/to/particles.cs \
  --filtered-particles-cs /path/to/filter_output/filtered_particles.cs \
  --rejected-particles-cs /path/to/filter_output/rejected_particles.cs \
  --vesicle-mask-dir /path/to/filter_vesicles_outputs \
  --micrographs-dir /path/to/micrographs \
  --output-dir /path/to/filter_output/overlays
```

### 6. CryoVIA vs SAM visual benchmark

Run this when comparing CryoVIA pretrained U-Net segmentation to the SAM/Vesicle Picker workflow:

```bash
python segmentation_integration/scripts/run_segmentation_visual_benchmark.py \
  --manifest /path/to/manifest.tsv \
  --output-dir /path/to/segmentation_visual_benchmark \
  --vesicle-picker-dir /nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker \
  --sam-find-parameters /path/to/find_vesicles.ini \
  --sam-filter-parameters /path/to/filter_vesicles.ini \
  --cryovia-model Default_thin \
  --limit 20
```

Use `--dry-run` first to write the scaffold and resolved config without launching CryoVIA, TensorFlow, or SAM.

## Tests

There is no root packaging file yet. Existing tests are per-area pytest suites:

```bash
python -m pytest denoising/tests
python -m pytest segmentation_integration/tests
python -m pytest topaz_picker_2/tests
```

For a quick current-code check from the root:

```bash
python -m pytest segmentation_integration/tests topaz_picker_2/tests denoising/tests
```

## Documentation Maintenance Notes

- Prefer adding current top-level orientation here and keeping detailed command flags in the per-area READMEs.
- Keep generated outputs under each area’s `results/`, `runs/`, or external `/nfs/...` run directories.
- Keep raw data, CryoSPARC exports, model weights, and large denoised MRC outputs out of git.
- When promoting a dry-run command to a real job, first confirm the installed tool flags with commands such as `topaz denoise --help`, `topaz train --help`, or `pyp --help` inside the target HPC environment.
