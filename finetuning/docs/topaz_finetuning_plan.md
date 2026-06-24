# Topaz fine-tuning implementation plan for Prestin SPA denoising

## Objective

Build a reproducible Topaz fine-tuning pipeline for Prestin-specific SPA micrographs using paired even/odd split images. The first implementation should support a pilot subset of about 300 micrograph pairs, then scale to 1000+ pairs without changing the dataset contract or evaluation protocol.

The pipeline should answer one question before any production-scale training:

> Does Prestin-specific fine-tuning improve downstream particle localization while avoiding denoising artifacts and hallucinated structure?

## Guiding constraints

- Keep this work under `zy223/finetuning`.
- Treat raw/preprocessed micrograph directories as read-only.
- Do not copy large image datasets into the repository.
- Store manifests, configs, logs, and small summary plots in this tree; store large denoised outputs/checkpoints under parameterized result paths.
- Split by complete micrograph, not by patch, to avoid leakage.
- Evaluate pretrained Topaz and fine-tuned Topaz on exactly the same held-out test micrographs.
- Confirm installed Topaz CLI flags with `topaz denoise --help` before launching jobs. This install exposes Noise2Noise training through `topaz denoise --dir-a ... --dir-b ...`, not a separate `denoise-train` subcommand.

## Phase 0: Confirm input data contract

### Current sample dataset path

Use this external read-only dataset path for the initial pilot. Do not import or copy the dataset into `zy223/finetuning`:

```text
/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/zy223@duke.edu/projects/Prestin-test-WNXj0kBZzgoPdMbE/sp-preprocessing-gkym1IT0yo4s7y1L/mrc
```

This directory should contain complete micrographs plus even/odd split files. The current split naming convention is:

```text
*_half1.mrc
*_half2.mrc
```

The pair-preparation script should infer paired examples from `_half1` and `_half2` filenames while ignoring complete micrographs unless explicitly requested for baseline/downstream comparisons.

### Required paired data

Each training example is a pair of noisy observations of the same underlying micrograph signal:

```text
micrograph_id    even_path            odd_path             optional_mask_path
prestin_000001   /path/...even.mrc    /path/...odd.mrc     /path/...mask.mrc
prestin_000002   /path/...even.mrc    /path/...odd.mrc     /path/...mask.mrc
```

Required invariants:

- even and odd files are aligned;
- same dimensions;
- same pixel size;
- same preprocessing pipeline;
- same dose-weighting assumptions, or explicitly recorded if not;
- no pair crosses acquisition/session metadata incorrectly;
- file paths point to source data outside this repo or symlinks in a controlled data area.

### Manifest format

Use a TSV manifest as the stable interface between dataset prep, training, and evaluation:

```text
micrograph_id	even_path	odd_path	mask_path	session_id	defocus_group	pixel_size_a	shape_y	shape_x	notes
```

`mask_path` is optional for Topaz, but include the column from the beginning because DRACO will likely need mask-aware preparation/evaluation.

### DRACO masking requirements to preserve now

Even though this plan starts with Topaz, dataset preparation should not discard mask information. The manifest should support:

- membrane masks for mask-biased patch sampling;
- evaluation within membrane vs non-membrane regions;
- exclusion masks for carbon, contaminants, very sharp boundaries, or unusable areas;
- mask provenance: how the mask was produced, thresholded, and aligned;
- mask resolution/shape validation against even/odd MRCs.

For DRACO-specific future work, plan for a mask table or metadata sidecar:

```text
micrograph_id	mask_path	mask_type	mask_source	mask_version	positive_region_label	exclude_region_label
```

Do not make Topaz training depend on masks initially. Use masks first for stratified evaluation and optional patch sampling experiments.

## Phase 1: Dataset preparation

### Initial pilot subset

Target: approximately 300 even/odd micrograph pairs.

Recommended split:

```text
train: 240 pairs
val:    30 pairs
test:   30 pairs
```

If sessions/grids/defocus groups vary substantially, stratify the split so each subset covers the same distribution. Keep the test set untouched across all experiments.

### Scale-up subset

Target: 1000+ even/odd micrograph pairs.

Recommended split:

```text
train: 800+ pairs
val:   100 pairs
test:  100 pairs
```

If more data is available, keep the test set fixed and increase train/val. Avoid repeatedly selecting test micrographs based on visual quality.

### Dataset-prep scripts to implement

1. `scripts/prepare_pairs.py`
   - Inputs: even directory, odd directory, optional mask directory, filename matching rule or explicit input table.
   - Outputs: validated all-pairs TSV.
   - Checks: existence, dimensions, pixel size where readable, unique IDs, missing mates, optional mask alignment.
   - Does not copy source MRCs.

2. `scripts/split_pairs.py`
   - Inputs: all-pairs TSV, split fractions or target counts, random seed, optional stratification columns.
   - Outputs: `train_pairs.tsv`, `val_pairs.tsv`, `test_pairs.tsv`, and `split_summary.json`.
   - Must split by `micrograph_id` only.

3. `scripts/inspect_pairs.py` or add an inspection mode to `prepare_pairs.py`
   - Summarizes dimensions, pixel sizes, sessions, masks, and missing metadata.
   - Flags outliers before training.

## Phase 2: Evaluation first

Before changing model weights, implement evaluation for the pretrained Topaz baseline. This prevents tuning to subjective image appearance.

### Evaluation datasets

Use three dataset roles:

- `val`: guides early stopping and hyperparameter choices;
- `test`: final held-out comparison across pretrained and fine-tuned models;
- optional `visual_panel`: a small frozen list of representative micrographs for side-by-side figures.

The `visual_panel` should include:

- membrane-rich images;
- sparse/empty membrane regions;
- different ice thickness/SNR;
- edge-heavy images to expose boundary artifacts;
- typical and low-quality cases.

### Denoising-level checks

Implement metrics/plots for held-out even/odd pairs:

1. **Validation loss on held-out pairs**
   - Compute the same or comparable pixel loss used by Topaz training where possible.
   - Evaluate both directions when available: `D(even)` vs `odd`, `D(odd)` vs `even`.
   - Report mean, median, and per-micrograph distribution.

2. **Fourier spectra before/after denoising**
   - Radially averaged power spectrum for raw even/odd and denoised outputs.
   - Compare high-frequency attenuation and whether membrane-relevant bands are preserved.
   - Save summary plots per visual-panel image and aggregate curves.

3. **High-frequency noise suppression**
   - Define a configurable high-frequency band in normalized spatial frequency or Angstrom range.
   - Report power reduction after denoising.
   - Avoid rewarding aggressive suppression if consistency/downstream metrics degrade.

4. **Preservation of membrane edges**
   - If masks exist: evaluate gradient magnitude or edge contrast near membrane boundaries before/after denoising.
   - If masks do not exist yet: use conservative image-gradient summaries on the visual panel.
   - Flag sharp artificial rims at patch or image boundaries.

5. **Absence of hallucinated repeating structure**
   - Inspect autocorrelation or Fourier peaks not present in raw inputs.
   - Track structured background texture in non-membrane regions where masks are available.
   - Compare `corr(D(even), D(odd))` against raw split correlation; excessive increases may indicate artificial structural correlation.

6. **Visual inspection**
   - Generate fixed side-by-side panels:

```text
even | odd | pretrained D(even) | pretrained D(odd) | fine-tuned D(even) | fine-tuned D(odd)
```

   - Use identical contrast scaling rules within each panel.
   - Include zoomed membrane regions and background regions.

### Consistency checks

For every held-out pair, run the model on both splits:

```text
D_even = D(even)
D_odd  = D(odd)
```

Report:

- pixel correlation between `D_even` and `D_odd`;
- Fourier shell/ring correlation if implemented for 2D images;
- masked consistency within membrane regions vs background;
- difference maps: `D_even - D_odd`;
- whether consistency gains are accompanied by visible artifact creation.

Important interpretation:

- Higher consistency is good only if it reflects shared biological signal.
- Suspiciously high consistency in background or edge regions may indicate hallucination or shared model bias.

### Downstream checks

Denoising should be judged by whether it helps Prestin localization, not just prettier images.

Minimum downstream comparison:

1. Run MiLoPYP exploration on the same held-out test micrographs for:
   - raw/preprocessed baseline if available;
   - pretrained Topaz denoised outputs;
   - fine-tuned Topaz denoised outputs.

2. Compare:
   - exploration output quality;
   - number of candidate Prestin picks per micrograph;
   - spatial distribution of picks relative to membrane masks;
   - whether picks concentrate on plausible membrane regions rather than edges/background;
   - overlap/consistency of picks between even/odd-derived denoised images;
   - downstream 2D class averages after picking when feasible;
   - whether membrane-associated densities are preserved.

3. Record failure modes:
   - more picks but worse 2D classes;
   - picks concentrated on denoising artifacts;
   - loss of faint membrane-associated density;
   - over-sharpened membranes mistaken for particles.

## Phase 3: Topaz baseline inference

Implement `scripts/run_topaz_baseline.py` as a dry-run-first launcher.

Inputs:

- split manifest or list of MRCs;
- output directory;
- model name/checkpoint;
- device;
- patch size/padding;
- overwrite flag;
- dry-run flag.

Output structure:

```text
results/topaz_pretrained_pilot/
  config_resolved.yaml
  commands.txt
  logs/
  denoised_even/
  denoised_odd/
  evaluation/
```

Run pretrained inference on validation/test before any training. This is the baseline every fine-tuned model must beat.

## Phase 4: Topaz fine-tuning

### Training approach

Use paired even/odd Noise2Noise training:

```text
input:  even patch    target: odd patch
input:  odd patch     target: even patch
```

For this installed Topaz version, training uses paired directories:

```text
topaz denoise --dir-a <training_inputs> --dir-b <training_targets> --method noise2noise ...
```

The launcher creates matched symlink directories from the train manifest and prints the command in dry-run mode first.

### Conservative pilot hyperparameters

Start with a conservative pilot:

```text
subset:          ~300 pairs total
train/val/test:  240/30/30
learning rate:   low; start around 1e-5 if CLI allows
max epochs:      conservative, e.g. 5-20
early stopping:  based on validation loss if supported, otherwise manual checkpoint comparison
patch size:      use Topaz-recommended/default unless memory requires smaller
augmentations:   flips/rotations only if Topaz supports them safely
```

Do not tune aggressively until evaluation scripts and baseline outputs are stable.

### Comparison matrix

Initial experiments:

| Experiment | Purpose |
|---|---|
| pretrained Topaz | baseline |
| fine-tuned Topaz, low LR | primary test |
| fine-tuned Topaz, lower LR | overfitting/artifact control |
| optional shorter training | checks whether early epochs are best |

Later, if the pilot is promising:

| Experiment | Purpose |
|---|---|
| 1000+ pair fine-tune | scale-up performance |
| mask-biased patch sampling | membrane-focused denoising |
| layer freezing/adapters if feasible | reduce overfitting |
| DRACO comparison with masks | cross-model benchmark |

### Early stopping and checkpoint selection

Checkpoint selection should use:

1. validation loss;
2. held-out consistency metrics;
3. visual artifact review on the frozen visual panel;
4. downstream MiLoPYP comparison where feasible.

Do not select a checkpoint only because it looks sharper.

## Phase 5: Reporting and decision gates

Each experiment should produce a compact report:

```text
experiment_id
input manifests and split hash
Topaz version and command
checkpoint used
validation loss summary
Fourier/noise summary
consistency summary
artifact notes
visual panel path
MiLoPYP downstream summary, if run
recommendation: reject / repeat / scale up
```

### Pilot success criteria

A fine-tuned Topaz model is worth scaling beyond ~300 pairs only if it shows:

- validation loss improves or remains comparable without obvious artifacts;
- Fourier/noise metrics improve without destroying membrane edges;
- consistency improves mainly in plausible signal regions, not background artifacts;
- visual panels show no suspicious repeating structure or sharp edge artifacts;
- MiLoPYP exploration is at least not worse, and preferably gives more plausible membrane-localized candidates;
- downstream 2D class averages are unchanged or improved when available.

### Stop conditions

Pause or reject the fine-tuning direction if:

- background develops structured texture;
- denoised even/odd outputs become artificially correlated in empty regions;
- membranes become over-sharpened or rimmed;
- MiLoPYP picks move toward artifacts or image edges;
- 2D classes worsen despite better-looking denoised images.

## Implementation order

1. Create config template and manifest schema.
2. Implement pair preparation/validation.
3. Implement train/val/test splitting.
4. Implement pretrained Topaz inference launcher.
5. Implement denoising-level evaluation on pretrained outputs.
6. Freeze visual panel and baseline report.
7. Implement Topaz fine-tuning launcher.
8. Run pilot fine-tune on ~300 pairs with dry-run and small validation first.
9. Evaluate fine-tuned vs pretrained outputs on the unchanged test split.
10. If successful, scale to 1000+ pairs and add mask-aware/DRACO-specific branches.

## Open questions to resolve before coding launchers

- Confirm the sample dataset path is readable from the intended compute environment.
- Confirm exact complete-micrograph filename convention so scripts can ignore or separately track non-half files.
- Whether masks already exist, and if so, their format and label semantics.
- Installed Topaz version and exact `topaz denoise` training behavior for checkpoint saving/loading.
- Whether Topaz can resume from the desired pretrained checkpoint directly via `--model` or only initialize from named model presets.
- Where large outputs/checkpoints should live on HPC scratch/project storage.
- MiLoPYP command interface and expected input naming for denoised micrographs.
