# Running command-line PYP from the container

Use this note for the nextPYP/PYP command-line container used by the particle-picking benchmark.

## Container path

```text
/nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif
```

This HPC environment has both `apptainer` and `singularity`; prefer `apptainer`.

## Smoke test

```bash
apptainer exec /nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif pyp --help
```

If project or data paths are not visible inside the container, bind the relevant `/nfs` trees explicitly:

```bash
apptainer exec \
  --bind /nfs/bartesaghilab2:/nfs/bartesaghilab2 \
  --bind /nfs/bartesaghilab:/nfs/bartesaghilab \
  --bind /nfs/bartesaghilab/www:/nfs/bartesaghilab/www \
  /nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif \
  pyp --help
```

## Recommended bash wrapper

Use this in `zy223/particle_picking/scripts/run_particle_picking_benchmark.sh`:

```bash
PYP_SIF=/nfs/bartesaghilab/www/files/pyp/0.7.3/pyp.sif

run_pyp() {
  apptainer exec \
    --bind /nfs/bartesaghilab2:/nfs/bartesaghilab2 \
    --bind /nfs/bartesaghilab:/nfs/bartesaghilab \
    --bind /nfs/bartesaghilab/www:/nfs/bartesaghilab/www \
    "${PYP_SIF}" \
    pyp "$@"
}
```

## Important working-directory rule

Run `pyp` from inside the intended nextPYP project directory. PYP writes `.pyp_config.toml`, `pkl/`, `webp/`, and related outputs relative to the current working directory.

Example for the raw-image branch of a benchmark run:

```bash
mkdir -p "${RUN_DIR}/nextpyp/raw"
cd "${RUN_DIR}/nextpyp/raw"

run_pyp \
  -data_mode spr \
  -scope_voltage 300 \
  -scope_pixel "${SCOPE_PIXEL}" \
  -data_path="${RUN_DIR}/inputs/raw/*.mrc" \
  -detect_rad "${DETECT_RAD}" \
  -detect_method "${DETECT_METHOD}" \
  -detect_dist "${DETECT_DIST}" \
  -slurm_tasks "${SLURM_TASKS}" \
  -slurm_memory "${SLURM_MEMORY}"
```

For the benchmark, repeat the same command template in separate project directories for `raw`, `topaz`, and `draco`, changing only the project working directory and `-data_path`.
