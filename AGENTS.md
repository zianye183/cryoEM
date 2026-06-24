# Agent Instructions

- Output all mathematical formulas in LaTeX rendered in Codex. Do not overly use LaTeX for natural language; restrict it to math.
- Always commit and push after making an edit.
- For CryoSPARC Import Result Group workflows, keep imported result-group files under the CryoSPARC project directory. For project `CS-prestin`, use a path under `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/`, such as `imports/<name>/`. The `.csg` should reference the colocated `.cs` file by local filename, e.g. `filtered_particles.cs`, not an absolute `/nfs/.../filtered_particles.cs` path; otherwise CryoSPARC may prepend the job directory and fail import.

## HPC Runtime Paths

- On the Bartesaghi HPC, keep all runtime outputs, caches, downloaded model weights, temporary generated assets, and environment-specific artifacts under `/nfs/bartesaghilab2/zy223/`. Do not rely on the default `$HOME` because it resolves to `/hpc/home/zy223`, which is space-constrained and has already caused PyTorch model-cache failures.
- For SSH/scp access, use `zy223@hpc-bartesaghi-login-01.oit.duke.edu` as the host.
- For Python/PyTorch commands, prefer these defaults unless explicitly overridden:
  - `CET_PICK_SHARED_ROOT=/nfs/bartesaghilab2/zy223`
  - `XDG_CACHE_HOME=/nfs/bartesaghilab2/zy223/.cache`
  - `TORCH_HOME=/nfs/bartesaghilab2/zy223/.cache/torch`
  - `MPLCONFIGDIR=/nfs/bartesaghilab2/zy223/.cache/matplotlib`

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
  - The active Python environment may not include all exploration dependencies. Recent local verification was blocked because `mrcfile` and `albumentations` were missing, although both are listed in `requirements.txt`.
  - The CLI still defaults `--arch` to `unet_4`. SPA SimSiam exploration should explicitly use a SimSiam architecture such as `--arch simsiam2d_18` until the defaulting/validation path is fixed.
  - `opts.py` currently hardcodes `opt.data_dir` to `<repo>/data`. SPA manifests generated elsewhere must be copied under `data` or passed through paths that resolve from that directory until a configurable `--data_dir` option is added.
- Prestin is the target dataset for the SPA work. Nearby project context exists under `segmentation_integration`, including Prestin CryoSPARC export references and `.cs` utilities.
