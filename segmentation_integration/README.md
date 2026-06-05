# CryoSPARC Vesicle-Mask Pick Filtering

This workflow filters CryoSPARC particle-location exports against Vesicle Picker
filtered mask pickles. Vesicle Picker should be run separately through
`find_vesicles.py` and `filter_vesicles.py`.

The filter expects CryoSPARC-style mask filenames:

```text
{location/micrograph_uid}_vesicles_filtered.pkl
```

## End-to-end readout

For the standard Prestin subset runs, use one dataset preset:

```text
raw   -> particles: /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J71_particles
         micrographs: /nfs/bartesaghilab2/zy223/denoising/data/fuller_subset
draco -> particles: /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J70_particles
         micrographs: /nfs/bartesaghilab2/zy223/denoising/results/draco_pretrained_fuller_subset
```

Segmentation run directories live under:

```text
/nfs/bartesaghilab2/zy223/segmentation_runs/{name}
```

The Vesicle Picker checkout is recorded as:

```text
/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker
```

First prepare Vesicle Picker parameter files and a runnable shell script:

```bash
python segmentation_integration/scripts/prepare_vesicle_picker_run.py \
  --dataset raw \
  --local-micrographs \
  --output-dir /path/to/segmentation_run \
  --segmentation-preset default \
  --filter-output-dir /path/to/segmentation_run/filtered_particles
```

Use `--segmentation-preset recall_heavy` to generate a SAM recall-focused
`find_vesicles.ini` with `points_per_side = 64`, `pred_iou_thresh = 0.8`,
`stability_score_thresh = 0.8`, `crop_n_layers = 2`, and
`crop_nms_thresh = 0.5`.

Use `--dataset draco` for the Draco-denoised micrographs. The preset locates
the one `.cs` and one `.csg` file in the corresponding CryoSPARC exported
particle result-group directory. The explicit path flags still exist as
overrides for nonstandard runs.

`--local-micrographs` runs SAM/Vesicle Picker on the preset MRC directory
instead of asking Vesicle Picker to pull micrographs from a CryoSPARC job, so
`--csparc-login`, `--project-id`, `--workspace-id`, and `--job-id` are not
needed. Omit `--local-micrographs` only when you want stock Vesicle Picker
CryoSPARC import behavior.

Default SAM weights:

```text
/nfs/bartesaghilab2/mm1229/segmentation_code/vesicle-picker/sam_vit_h_4b8939.pth
```

This writes:

```text
/path/to/segmentation_run/
  parameters/find_vesicles.ini
  parameters/filter_vesicles.ini
  vesicle_masks_found/
  vesicle_masks_filtered/
  run_end_to_end.sh
```

Then run:

```bash
bash /path/to/segmentation_run/run_end_to_end.sh
```

The shell script runs Vesicle Picker `find_vesicles.py`, Vesicle Picker
`filter_vesicles.py`, upstream found-vs-filtered mask QC, standalone mask
visualization, and then this repo's CryoSPARC particle filter. The upstream QC
PNGs and mask stats are produced under `filtered_particles/mask_filter_qc/`.
Per-vesicle metric QC is rendered for all found segmentations before filtering
as `{uid}_found_mask_metrics.png`, with accepted masks in green and rejected
masks in red.
The final membrane-band mask PNGs are produced under `filtered_particles/masks/`.

To render upstream QC from existing Vesicle Picker outputs:

```bash
python segmentation_integration/scripts/visualize_vesicle_mask_filtering.py \
  --found-mask-dir /path/to/vesicle_masks_found \
  --filtered-mask-dir /path/to/vesicle_masks_filtered \
  --output-dir /path/to/filter_output/mask_filter_qc \
  --particles-cs /path/to/particles.cs \
  --micrographs-dir /path/to/micrographs
```

When `--particles-cs` and `--micrographs-dir` are provided, the QC images overlay
found and filtered masks on the micrograph underlay. Without those inputs, the
script falls back to binary mask-only panels.

## Filter-only command

If Vesicle Picker masks already exist, run:

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

Import the resulting `filtered_particles.csg` with CryoSPARC's Import Result
Group job, then use the imported particle locations for extraction. By default,
QC output includes pick overlays under `overlays/` and standalone membrane-band
mask PNGs under `masks/`.

To regenerate overlays from existing outputs:

```bash
python segmentation_integration/scripts/visualize_filtered_picks.py \
  --particles-cs /path/to/particles.cs \
  --filtered-particles-cs /path/to/filter_output/filtered_particles.cs \
  --rejected-particles-cs /path/to/filter_output/rejected_particles.cs \
  --vesicle-mask-dir /path/to/filter_vesicles_outputs \
  --micrographs-dir /path/to/micrographs \
  --output-dir /path/to/filter_output/overlays
```
