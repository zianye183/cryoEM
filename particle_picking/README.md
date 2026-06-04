# Prestin particle-picking benchmarks

This directory is for benchmarking whether denoising improves downstream Prestin particle picking.

Canonical notes:

- PYP container usage: `docs/pyp_container_usage.md`
- Pilot config: `configs/particle_picking_benchmark_pilot.env`

The command-line PYP container for this project is:

```text
/nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif
```

Generated benchmark outputs should stay under `results/` and should not include copied raw micrograph datasets.

---

# Denoising-assisted particle-picking benchmark plan

Goal: test whether denoising helps downstream Prestin particle picking, not whether denoised micrographs look smoother. All downstream evidence should be evaluated from coordinates generated on raw / Topaz-denoised / DRACO-denoised images, with extraction/classification ultimately performed from the raw micrographs when feasible.

## Inputs

Use a small, fixed subset first:

- raw/preprocessed micrographs: read-only source directory
- Topaz-denoised micrographs: generated or reused from prior runs
- DRACO-denoised micrographs: generated or reused from prior runs

The benchmark implementation should live under `zy223/particle_picking/`. The benchmark script stages only symlinks into a run directory. It does not move, delete, or copy raw data.

## Methods compared

For each image condition (`raw`, `topaz`, `draco`):

1. **DoG pyramid + NMS picker**
   - Use a dedicated implementation under `zy223/particle_picking/scripts/`, reusing the current `zy223/denoising/scripts/evaluate_dog_picking.py` logic as a starting point if helpful.
   - Same detector parameters across all image conditions.
   - Scores denoised-assisted picks on the raw micrograph DoG response as a label-free proxy.
   - Exports `.box` files for visual inspection/import.

2. **nextPYP 2D particle picking**
   - nextPYP docs list 2D picking methods in the Pre-processing block:
     - coordinate import (`*.box`, `*.spk`),
     - manual,
     - size-based `auto` / `all`,
     - neural-network `pyp-train` / `pyp-eval` and Topaz wrappers.
   - First pilot should use size-based picking (`detect_method=auto` and optionally `all`) on each image condition with identical `detect_rad` and `detect_dist`.
   - The script creates one nextPYP project per image condition and records the exact `pyp` command. If `pyp` is available and `RUN_NEXTPYP=1`, it executes; otherwise it writes runnable command files for the configured environment.

## Outputs

Each benchmark run writes to a timestamped output directory under `zy223/particle_picking/results/`:

```text
zy223/particle_picking/
  README.md
  configs/
    particle_picking_benchmark_pilot.env
  docs/
    particle_picking_benchmark_plan.md
  scripts/
    run_particle_picking_benchmark.sh
    evaluate_dog_picking.py
    export_dog_picks_to_box.py
    plot_picking_overlays.py
  results/
    particle_picking_benchmark/<run_id>/
      config.env                       # resolved run configuration
      logs/                            # command logs
      inputs/<method>/*.mrc            # symlinks only
      denoised/topaz/*.mrc             # optional generated Topaz outputs
      denoised/draco/*.mrc             # optional generated DRACO outputs
      dog/
        dog_summary.tsv
        dog_metrics.tsv
        dog_picks.tsv
        boxes/<method>/*.box
        overlays/*.png
      nextpyp/
        raw/run_nextpyp.sh
        topaz/run_nextpyp.sh
        draco/run_nextpyp.sh
        ... nextPYP outputs if executed ...
```

## Evaluation criteria

### Quick quantitative screen

For DoG/NMS:

- comparable pick count, not just massive overpicking;
- denoised-assisted picks have higher raw-micrograph DoG response than raw picks;
- reasonable overlap/stability with raw picks without collapsing to obvious membrane edges or contaminants;
- no method produces suspiciously uniform artifacts.

### By-eye review

Use overlay PNGs with identical display settings. Inspect:

- whether picks lie on plausible membrane-associated densities;
- whether denoising moves picks away from background/ice/carbon;
- whether picks are over-concentrated on membrane edges;
- whether Topaz/DRACO introduce hallucinated blob patterns.

### nextPYP downstream screen

For each image condition, compare:

- number of picked particles per micrograph;
- by-eye micrograph overlays in nextPYP;
- if enough particles: raw-data extraction followed by a small 2D classification run;
- class-average quality: coherent Prestin-sized signal, membrane contamination, junk class fraction, reproducibility across micrographs.

## Important constraints

- Do not extract/reconstruct from denoised images as the main biological result. Use denoised images to assist picking; validate particles on raw data.
- Keep all parameters identical across raw / Topaz / DRACO except the input image directory.
- Preserve exact commands and config for reproducibility.
- Keep new particle-picking benchmark scripts, configs, docs, and generated outputs under `zy223/particle_picking/`; do not add new benchmark implementation files under `zy223/denoising/`.
- Start with `LIMIT=5` or `LIMIT=10`; do not launch full-scale jobs until the pilot is reviewed.
