# CryoSPARC Vesicle-Mask Pick Filtering

This workflow filters CryoSPARC particle-location exports against Vesicle Picker
filtered mask pickles. Vesicle Picker should be run separately through
`find_vesicles.py` and `filter_vesicles.py`.

The filter expects CryoSPARC-style mask filenames:

```text
{location/micrograph_uid}_vesicles_filtered.pkl
```

## End-to-end readout

First prepare Vesicle Picker parameter files and a runnable shell script:

```bash
python segmentation_integration/scripts/prepare_vesicle_picker_run.py \
  --output-dir /path/to/segmentation_run \
  --vesicle-picker-dir /path/to/vesicle-picker \
  --csparc-login /path/to/csparc_login.ini \
  --project-id PXX \
  --workspace-id WXX \
  --job-id JXX \
  --job-type curate \
  --model-weights-path /path/to/sam_vit_l_0b3195.pth \
  --particles-cs /path/to/particles.cs \
  --particles-csg /path/to/particles.csg \
  --micrographs-dir /path/to/micrographs \
  --filter-output-dir /path/to/segmentation_run/filtered_particles
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
`filter_vesicles.py`, standalone mask visualization, and then this repo's
CryoSPARC particle filter. The mask PNGs are produced immediately after
Vesicle Picker filtering under `filtered_particles/masks/`.

## Filter-only command

If Vesicle Picker masks already exist, run:

```bash
python segmentation_integration/scripts/filter_cryosparc_picks_by_vesicle_masks.py \
  --particles-cs /path/to/particles.cs \
  --particles-csg /path/to/particles.csg \
  --vesicle-mask-dir /path/to/filter_vesicles_outputs \
  --micrographs-dir /path/to/micrographs \
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
