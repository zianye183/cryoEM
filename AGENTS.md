# Agent Instructions

- Output all mathematical formulas in LaTeX rendered in Codex. Do not overly use LaTeX for natural language; restrict it to math.
- Always commit and push after making an edit.
- When creating data processing pipelines, always display progress.
- For CryoSPARC Import Result Group workflows, keep imported result-group files under the CryoSPARC project directory. For project `CS-prestin`, use a path under `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/`, such as `imports/<name>/`. The `.csg` should reference the colocated `.cs` file by local filename, e.g. `filtered_particles.cs`, not an absolute `/nfs/.../filtered_particles.cs` path; otherwise CryoSPARC may prepend the job directory and fail import.

## HPC Runtime Paths

- On the Bartesaghi HPC, keep all runtime outputs, caches, downloaded model weights, temporary generated assets, and environment-specific artifacts under `/nfs/bartesaghilab2/zy223/`. Do not rely on the default `$HOME` because it resolves to `/hpc/home/zy223`, which is space-constrained and has already caused PyTorch model-cache failures.
- For SSH/scp access, use `zy223@hpc-bartesaghi-login-01.oit.duke.edu` as the host. Interactive sessions can use `ssh -Y zy223@hpc-bartesaghi-login-01.oit.duke.edu`.
- Non-interactive SSH probes to that host are default-allowed, including `ssh -o BatchMode=yes -o ConnectTimeout=10 zy223@hpc-bartesaghi-login-01.oit.duke.edu`.
- Request interactive compute with Slurm instead of running heavy jobs on the login node. Typical allocations:
  - GPU: `srun -p bartesaghilab-gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=12:00:00 --pty bash`
  - CPU: `srun -p bartesaghilab-cpu --cpus-per-task=8 --mem=64G --time=12:00:00 --pty bash`
- For Python/PyTorch commands, prefer these defaults unless explicitly overridden:
  - `CET_PICK_SHARED_ROOT=/nfs/bartesaghilab2/zy223`
  - `XDG_CACHE_HOME=/nfs/bartesaghilab2/zy223/.cache`
  - `TORCH_HOME=/nfs/bartesaghilab2/zy223/.cache/torch`
  - `MPLCONFIGDIR=/nfs/bartesaghilab2/zy223/.cache/matplotlib`

## Local CET Pick Environment

- A local `.venv` exists in `cet_pick` with Python 3.10 for fast local experiments. Activate it from the `cet_pick` repo root with `source .venv/bin/activate`.
- The local environment includes the editable `cet_pick` package plus a practical CPU/local experiment stack: PyTorch, torchvision, scipy, scikit-image, scikit-learn, pandas, matplotlib, opencv, mrcfile, torchio, numba, umap, and related dependencies.
- Validation previously passed for `pip check`, `cet_pick` model-path imports, and a synthetic MRC read/write smoke test.

## CET Pick / MiLoPYP Context

- The `cet_pick` repository contains MiLoPYP, a cryo-electron tomography particle exploration and refinement pipeline. The corresponding paper is `/Users/alanye/Documents/CS+ 2026/MiLoPYP.pdf`.
- The pipeline has two high-level stages:
  - Exploration: starts from rough particle or candidate locations, extracts patches, and trains a SimSiam-style self-supervised backbone to organize candidates in embedding space. Users then inspect clustered/embedded candidates and recursively pick out good labels.
  - Refinement: uses the good labels selected from exploration as sparse ground truth for positive-unlabeled learning, then predicts a final particle probability heatmap.
- Current project focus: adapt the exploration stage from cryo-ET to SPA. The original exploration code assumes tomograms, tilt series, 3D coordinates, DoG candidates over volumes, and paired 2D/3D patch inputs. SPA adaptation should primarily change the exploration data ingestion and patch extraction path so it can consume SPA micrographs plus rough 2D particle locations, while reusing the SimSiam training machinery where possible.
- Keep the original cryo-ET/3D functionality intact. Treat SPA/2D support as additive extensions: new dataset keys, classes, scripts, tests, and narrow compatibility shims are preferred over rewriting or weakening existing 3D paths. When a shared file must change, preserve current behavior for existing dataset/task names and add tests or smoke checks that protect the 3D path.
- Important exploration files:
  - `cet_pick/simsiam_main.py`: exploration training entry point.
  - `cet_pick/datasets/tomo_milopyp_2d3d.py`: cryo-ET exploration dataset; loads tomograms/tilts, generates DoG candidates, extracts paired averaged tilt-series and tomogram patches.
  - `cet_pick/datasets/particle_milopyp_2d3d.py`: SimSiam dataset mixin that returns original and augmented positive views.
  - `cet_pick/trains/tomo_simsiam_trainer.py`: SimSiam loss/training wrapper.
  - `cet_pick/simsiam_test_hm_2d3d.py`: inference/export of learned exploration embeddings to `all_output_info.npz`.
  - `cet_pick/plot_2d.py`: UMAP/over-clustering and visualization of embeddings.
  - `cet_pick/select_sublabels.py`: exports selected embedding labels as refinement training coordinates.
- Known SPA exploration risks:
  - If the local `.venv` is not active, the ambient Python environment may not include all exploration dependencies. Use `cet_pick/.venv` for local checks unless explicitly told otherwise.
  - The CLI still defaults `--arch` to `unet_4`. SPA SimSiam exploration should explicitly use a SimSiam architecture such as `--arch simsiam2d_18` until the defaulting/validation path is fixed.
  - `opts.py` currently hardcodes `opt.data_dir` to `<repo>/data`. SPA manifests generated elsewhere must be copied under `data` or passed through paths that resolve from that directory until a configurable `--data_dir` option is added.
- Prestin is the target dataset for the SPA work. Nearby project context exists under `segmentation_integration`, including Prestin CryoSPARC export references and `.cs` utilities.
- A newer non-Prestin SPA dataset is SARS-CoV-2 under CryoSPARC project `CS-other-datasets`. Its ready micrograph/exposure export is `/nfs/bartesaghilab2/zy223/cryosparc/other_datasets/CS-other-datasets/exports/groups/J4_exposures`, and its pixel size is `1.061 A`.

## Local Data Skeleton

- The `cet_pick` repository tracks a lightweight Prestin data skeleton: real large data under `data/` remains ignored, while `.gitkeep` directories and `data_manifest.txt` expose structure and filenames. The referenced manifest records 40,431 entries: 40,392 `.tsv`, 24 `.txt`, 9 `.json`, and 6 `.npz` paths.
- Top-level manifest groups include:
  - `data/prestin_draco_spa_manifest`: full Draco-denoised SPA manifest with `all_images.txt`, `train_images.txt`, `test_images.txt`, `summary.json`, and 16,596 coordinate TSVs.
  - `data/prestin_draco_spa_manifest_3k`: 3k subset with split image lists, `subset_summary.json`, and 2,778 coordinate TSVs.
  - `data/prestin_draco_spa_manifest_with_pkl_masks`: full manifest variant with PKL-mask naming/coverage and 16,596 coordinate TSVs.
  - `data/prestin_draco_spa_manifest_3k_with_pkl_masks`: 3k PKL-mask variant with 2,758 coordinate TSVs plus patch caches `patch_cache_b128_align` and `patch_cache_b192_align`, each containing `all_patches.npz`, `train_patches.npz`, `test_patches.npz`, and `patch_cache_summary.json`.
  - `data/prestin_rounds/round_001_ckpt44_selected`: selected exploration round output containing a `good/manifest` subtree and `round_summary.json`.
  - `data/prestin_normal_field_top_pick_mics`, `data/prestin_one_mic_normal_field`, and `data/prestin_spa_draco_filtered_manifest`: small SPA manifest/test subsets.
