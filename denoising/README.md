r# Denoising tests: Topaz-Denoise and DRACO

This directory contains a lightweight Python-script workflow for initial cryo-EM denoising tests on a small subset of the Prestin preprocessing MRC files. It is designed for Zed: open the main `.py` workflow and run the `# %%` sections interactively in Zed's Python REPL.

## Source data

```text
/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-1fhwPcd8gQ/sp-preprocessing-ghLbktywzmKi35IA/mrc
```

At setup time this directory contained 35 `*_fractions.mrc` files. The workflow inspects headers before assuming whether these files are 2D micrographs or 3D frame stacks.

## Main workflow

Use this file first:

```text
zy223/denoising/scripts/denoising_initial_tests.py
```

It is split into Zed/VS Code-style cells using `# %%` markers. Run cells in order to:

1. define the project configuration,
2. check availability of `topaz`, DRACO, `mrcfile`, `torch`, `numpy`, and `matplotlib`,
3. create a reproducible small subset under `data/subset`,
4. inspect MRC dimensions without external dependencies,
5. print conservative Topaz/DRACO command templates,
6. optionally preview MRC images if `mrcfile` and `matplotlib` are installed.

The exploratory workflow intentionally does **not** run Topaz or DRACO automatically. Actual model execution is separated into dedicated runner scripts under `scripts/run_*.py`; run those only inside the appropriate environment/GPU allocation.

## Directory layout

```text
zy223/denoising/
  README.md
  configs/denoising_test_config.yaml
  data/subset/                       # symlinks to selected source MRCs
  notebooks/                         # old notebook scaffold; no longer primary
  results/topaz_pretrained/
  results/topaz_retrained/
  results/draco_pretrained/
  scripts/denoising_initial_tests.py # primary Zed REPL workflow
  scripts/check_env.py               # standalone environment check
  scripts/make_subset.py             # standalone subset creation
  scripts/inspect_mrc_headers.py     # standalone MRC header inspection
  scripts/build_commands.py          # standalone command-template printer
  scripts/denoising_run_common.py    # shared helpers for model runners
  scripts/run_topaz_pretrained.py    # actually runs Topaz pretrained inference
  scripts/run_topaz_retraining.py    # actually runs Topaz retraining once pairs exist
  scripts/setup_draco.py             # clones DRACO and downloads Denoise/B weights
  scripts/draco_denoise_adapter.py   # local batch adapter based on DRACO notebook
  scripts/run_draco_pretrained.py    # runs DRACO Denoise/B via the local adapter
```

## Zed REPL usage

Open:

```text
zy223/denoising/scripts/denoising_initial_tests.py
```

Then run the `# %%` cells interactively. The file can also be executed as a regular script:

```bash
python zy223/denoising/scripts/denoising_initial_tests.py
```

## Standalone commands

From the workspace root, the same steps can be run without the REPL:

```bash
python zy223/denoising/scripts/check_env.py

python zy223/denoising/scripts/make_subset.py \
  --source /nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-1fhwPcd8gQ/sp-preprocessing-ghLbktywzmKi35IA/mrc \
  --dest /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --n 5 \
  --seed 42 \
  --overwrite

python zy223/denoising/scripts/inspect_mrc_headers.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset

python zy223/denoising/scripts/build_commands.py \
  --subset /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --results /nfs/bartesaghilab2/zy223/denoising/results
```

## Running the models

The runner scripts are intentionally separate and modular:

- `scripts/run_topaz_pretrained.py` runs pretrained Topaz denoising on input MRCs.
- `scripts/run_topaz_retraining.py` runs Topaz denoising retraining after paired even/odd observations have been prepared.
- `scripts/run_draco_pretrained.py` runs a local DRACO inference adapter or explicit DRACO command template.

Use `--dry-run` first to print the exact command without launching work.

## Topaz-Denoise plan

References: Bepler et al., 2020, Nature Communications; <https://github.com/tbepler/topaz>.

### Pretrained test

After activating/installing Topaz, first verify the CLI flags for the installed version:

```bash
topaz denoise --help
```

The workflow prints a first-pass command template like:

```bash
topaz denoise --model unet --device 0 --patch-size 4000 --patch-padding 500 --suffix _topaz_denoised --output /nfs/bartesaghilab2/zy223/denoising/results/topaz_pretrained <subset .mrc files>
```

By default, the local Topaz runner asks Topaz to write outputs as `raw_stem_topaz_denoised.mrc`, matching DRACO's `raw_stem_draco_denoised.mrc` convention.

The actual separated runner is:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_topaz_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/topaz_pretrained \
  --model unet \
  --device 0 \
  --suffix _topaz_denoised \
  --dry-run
```

Remove `--dry-run` when the command and environment are confirmed. Use `--overwrite` if reusing a non-empty output directory.

### Retraining

Topaz denoising retraining is Noise2Noise-style and generally requires paired noisy observations, commonly even/odd frame sums from movie frames. Do not enable retraining until the `*_fractions.mrc` files are confirmed to be suitable frame stacks or until paired even/odd inputs are prepared.

Once paired files exist, run the separated retraining launcher with either a two-column TSV:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_topaz_retraining.py \
  --pairs-tsv /path/to/even_odd_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/topaz_retrained \
  --device 0 \
  --dry-run
```

or matching filenames in separate directories:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_topaz_retraining.py \
  --even-dir /path/to/even_mrcs \
  --odd-dir /path/to/odd_mrcs \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/topaz_retrained \
  --device 0 \
  --dry-run
```

Check `topaz denoise-train --help` in the installed Topaz environment; if your version uses different training flags, pass them with repeated `--extra-arg` or adjust the runner.

Denoised micrographs should mainly be used for visualization and particle picking. Downstream reconstruction should generally continue from raw data to avoid hallucination/artifact risk.

## DRACO plan

References: DRACO NeurIPS 2024 repo; <https://github.com/Cellverse/draco>.

DRACO's public repo is notebook/API-driven rather than a stable CLI. The local adapter in `scripts/draco_denoise_adapter.py` is based on the official `notebook/draco_denoise.ipynb` and runs batch MRC inference using:

- local DRACO source clone: `/nfs/bartesaghilab2/zy223/denoising/external/draco`,
- Denoise/B checkpoint: `/nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt`,
- official config: `<repo>/notebook/denoise.yaml`.

The Denoise/B checkpoint linked by the DRACO README is Google Drive file ID `1z6DunE6skfKNODC6vOJ_48QXHRBzBPyK`. To set up repo + weights after activating an environment with `git` and `gdown`:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/setup_draco.py
```

If you only want to see what will happen:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/setup_draco.py --dry-run
python /nfs/bartesaghilab2/zy223/denoising/scripts/setup_draco.py --print-install-commands
```

Then run a DRACO dry run:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_draco_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --repo /nfs/bartesaghilab2/zy223/denoising/external/draco \
  --weights /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained \
  --dry-run
```

Remove `--dry-run` after confirming the environment, repo, checkpoint, GPU allocation, and input shapes. The adapter expects one 2D micrograph per MRC; if `*_fractions.mrc` files are frame stacks, first produce summed/even/odd 2D micrographs.

Useful adapter options can be passed through repeated `--extra-arg=...` values. Use the equals form when forwarding options that start with `--`; otherwise `argparse` may interpret the forwarded option as an option to `run_draco_pretrained.py` itself.

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_draco_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --repo /nfs/bartesaghilab2/zy223/denoising/external/draco \
  --weights /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained \
  --extra-arg=--device \
  --extra-arg=cuda \
  --dry-run
```

To pass explicit tiling options to the local DRACO adapter:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/run_draco_pretrained.py \
  /nfs/bartesaghilab2/zy223/denoising/data/subset \
  --repo /nfs/bartesaghilab2/zy223/denoising/external/draco \
  --weights /nfs/bartesaghilab2/zy223/denoising/models/draco_denoise_b.ckpt \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained \
  --extra-arg=--device \
  --extra-arg=cuda \
  --extra-arg=--tile-size \
  --extra-arg=1024 \
  --extra-arg=--tile-overlap \
  --extra-arg=128 \
  --dry-run
```

DRACO source and model parameters are restricted to non-commercial use by non-commercial organizations and should not be redistributed outside approved use. The setup script downloads weights locally but does not make them part of this repository.

## IsoNet2 SPA smoke test

A prediction-only IsoNet2 smoke-test wrapper is available under:

```text
zy223/denoising/scripts/isonet2_spa/
```

It uses the pretrained IsoNet2 model:

```text
/nfs/bartesaghilab/nextpyp/workshop_dhvi/10453/isonet_network_isonet2-n2n_unet-medium_96_full.pt
```

The wrapper does **not** run IsoNet2 training, deconvolution, mask creation, or refinement. It only:

1. inspects checkpoint metadata,
2. repeats each 2D SPA micrograph along fake Z to make a pseudo-volume,
3. writes an IsoNet2-compatible STAR file with duplicated half1/half2 paths,
4. optionally runs `isonet.py predict` with missing-wedge application disabled,
5. extracts center/mean-Z/z-std 2D outputs,
6. writes QC stats and PNG panels.

This is only a qualitative compatibility/readout test. Repeated-Z pseudo-volumes create artificial 3D continuity and should not be treated as a validated SPA denoising method.

Prepare inputs and print the prediction command without launching IsoNet2. By default this center-crops each micrograph to `1024 x 1024` before fake-Z wrapping to avoid huge pseudo-volumes and memory kills:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/run_smoke_pipeline.py \
  /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset \
  --limit 3 \
  --crop-size 1024 \
  --dry-run-predict
```

If the IsoNet2 environment is not installed yet, create it under project storage rather than the HPC home directory:

```bash
bash /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/setup_project_env.sh
conda activate /nfs/bartesaghilab2/zy223/conda_envs/isonet2_environment
```

Run prediction after activating the project-local IsoNet2 environment and getting a GPU allocation:

```bash
python /nfs/bartesaghilab2/zy223/denoising/scripts/isonet2_spa/run_smoke_pipeline.py \
  /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset \
  --limit 3 \
  --crop-size 1024 \
  --run-predict \
  --gpu-id 0
```

By default the prediction command uses:

```text
--apply_mw_x1 False
--isCTFflipped True
```

so IsoNet2's tomographic missing-wedge mask and CTF sign filtering are avoided for this first SPA smoke test. Outputs are written under:

```text
/nfs/bartesaghilab2/zy223/denoising/experiments/isonet2_spa_smoke/
```

Individual steps can also be run directly: `inspect_checkpoint.py`, `make_pseudo_volumes.py`, `make_spa_star.py`, `run_predict.py`, `extract_slices.py`, and `qc_outputs.py`.

## Denoising evaluation plan

Goal: decide whether each denoiser suppresses uncorrelated noise while preserving weak, membrane-associated Prestin-sized signal. Because cryo-EM has no clean per-micrograph ground truth, do **not** treat visual smoothness or global SNR as sufficient. Use denoised micrographs for visualization/picking, but validate particle evidence from **raw extracted particles**.

### Theory of the tests

- Real structural signal should be reproducible across independent observations, e.g. even/odd frame sums; shot noise should not.
- A denoiser that preserves signal should leave mostly noise-like residuals: `residual = raw - matched_denoised`.
- If particle- or membrane-shaped structure appears in the residual, the denoiser may be deleting biological signal.
- If denoised-assisted coordinates produce better 2D/3D results after extracting from raw micrographs, denoising is useful for picking even if denoised images themselves are not reconstruction inputs.

### Evaluation stages

1. **Residual test**: compute residual MRCs and raw/denoised/residual panels for each method.
2. **Coordinate residual test**: if candidate picks exist, compare residual energy at candidate coordinates against nearby background patches.
3. **Even/odd test**: if even/odd frame sums exist, run residual tests separately on even and odd pairs; reproducible particle-like residuals are stronger evidence of deleted signal.
4. **Downstream validation**: pick on denoised images, extract from raw micrographs, then compare 2D classes and eventual 3D refinement against raw-picked controls.
5. **Optional spike-in control**: inject weak synthetic Prestin-sized particles into real backgrounds/membranes and test whether denoising preserves or removes them.

### Run residual evaluation

Use the project DRACO venv for this script because it already has `numpy`, `mrcfile`, and `matplotlib`:

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  /nfs/bartesaghilab2/zy223/denoising/scripts/evaluate_residuals.py \
  --raw-dir /path/to/raw_mrcs \
  --denoised-dir /path/to/topaz_denoised_mrcs \
  --denoised-suffix _topaz_denoised \
  --method topaz_pretrained \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/residual_eval_topaz \
  --normalization linear-match
```

If denoised filenames are `raw_stem + suffix + .mrc`:

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  /nfs/bartesaghilab2/zy223/denoising/scripts/evaluate_residuals.py \
  --raw-dir /path/to/raw_mrcs \
  --denoised-dir /path/to/draco_denoised_mrcs \
  --denoised-suffix _draco_denoised \
  --method draco_pretrained \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/residual_eval_draco
```

For arbitrary filename mappings, use a TSV:

```text
raw	denoised	method
/path/raw_001.mrc	/path/denoised_001.mrc	topaz
/path/raw_002.mrc	/path/denoised_002.mrc	topaz
```

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  /nfs/bartesaghilab2/zy223/denoising/scripts/evaluate_residuals.py \
  --pairs-tsv /path/to/raw_denoised_pairs.tsv \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/residual_eval
```

Outputs:

```text
residuals/*.mrc          # residual maps
panels/*.png             # raw | denoised | residual | abs residual
patches/*.png            # optional coordinate patch montages
residual_metrics.tsv     # image and patch metrics
run_config.json          # command provenance
```

### Run coordinate-centered residual evaluation

Coordinate TSV format:

```text
micrograph	x	y	label
FoilHole_001.mrc	1234	2048	candidate
FoilHole_001.mrc	1876	1650	candidate
```

Then run:

```bash
/nfs/bartesaghilab2/zy223/venvs/draco311/bin/python \
  /nfs/bartesaghilab2/zy223/denoising/scripts/evaluate_residuals.py \
  --raw-dir /path/to/raw_mrcs \
  --denoised-dir /path/to/topaz_denoised_mrcs \
  --denoised-suffix _topaz_denoised \
  --coords-tsv /path/to/candidate_coords.tsv \
  --box-size 128 \
  --method topaz_pretrained \
  --output-dir /nfs/bartesaghilab2/zy223/denoising/results/residual_eval_topaz_coords
```

### Interpret residual results

Inspect `panels/*.png` first, then `residual_metrics.tsv`.

Good signs:

- residual looks texture-like/noise-like, not particle-shaped;
- residual does not preferentially trace membranes;
- coordinate patches show no coherent average particle feature;
- `particle_background_energy_ratio` is near 1 when coordinates are supplied.

Warning signs:

- particle-sized blobs or membrane edges appear in residual panels;
- residual patches centered on candidate particles look structured;
- `particle_background_energy_ratio > 1.2–1.5` consistently across micrographs/methods;
- `particle_background_energy_ratio > 2` is strongly suspicious;
- raw-only picks that vanish after denoising produce good raw 2D classes.

Interpret ratios comparatively, not as absolute truth. Local ice thickness, membranes, contaminants, and imperfect coordinate quality can all inflate residual energy. The strongest residual evidence is reproducible structure in even/odd residuals or coherent residual averages at candidate coordinates.

### Downstream validation criterion

A denoiser should be preferred only if denoised-assisted picking improves raw-data processing:

```text
raw picking              -> raw extraction -> 2D/3D
Topaz-denoised picking   -> raw extraction -> 2D/3D
DRACO-denoised picking   -> raw extraction -> 2D/3D
consensus/union picking  -> raw extraction -> 2D/3D
```

Compare:

- number and fraction of usable 2D classes;
- particle retention after cleanup;
- membrane contamination in classes;
- orientation/class diversity, especially weak views;
- ab initio/refinement success and FSC when enough particles exist.

Do not use denoised particles as the main reconstruction input unless explicitly testing artifacts; neural denoisers can alter noise statistics or hallucinate structure.

## Environment notes

Use project-local `venv` environments under `/nfs/bartesaghilab2/zy223/venvs` when available. Conda/module-based environments are also acceptable for tools like IsoNet2 when the cluster provides them; activate the environment before running the corresponding model command.

Topaz is installed in:

```bash
source /nfs/bartesaghilab2/zy223/venvs/topaz/bin/activate
topaz --help
```

To rebuild the Topaz venv:

```bash
python3 -m venv /nfs/bartesaghilab2/zy223/venvs/topaz
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/python -m pip install --upgrade pip wheel setuptools
/nfs/bartesaghilab2/zy223/venvs/topaz/bin/python -m pip install -r /nfs/bartesaghilab2/zy223/denoising/envs/topaz-requirements.txt
```

DRACO currently uses the Python 3.11 project venv:

```bash
source /nfs/bartesaghilab2/zy223/venvs/draco311/bin/activate
```

An older Python 3.9 DRACO venv may also exist at `/nfs/bartesaghilab2/zy223/venvs/draco`, but prefer `draco311` for the current DRACO scripts.

Topaz currently uses `topaz-em` from PyPI, which pulls PyTorch/torchvision and CUDA runtime wheels. Match GPU-node CUDA expectations when changing Torch versions.
