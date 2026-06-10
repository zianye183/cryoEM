# CryoSPARC CLI setup for this project

This repository uses CryoSPARC primarily for Prestin project `CS-prestin`.
Large CryoSPARC data live on the HPC under:

```text
/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin
```

Use two command-line paths:

- `cryosparc-tools` from any Python environment that can reach the CryoSPARC
  web/API port.
- `cryosparcm` only on the CryoSPARC master host, or anywhere the
  `cryosparc_master/bin` directory is available.

The current public package for scripting is `cryosparc-tools`; PyPI lists
version `5.0.3`, released on May 4, 2026, with Python `>=3.8` support.

## Configure local credentials

Copy the example config and fill in private values:

```bash
cp configs/cryosparc_cli.example.env .env.cryosparc
chmod 600 .env.cryosparc
```

Then edit `.env.cryosparc`. At minimum set:

```bash
export CRYOSPARC_HOST=...
export CRYOSPARC_BASE_PORT=...
export CRYOSPARC_EMAIL=...
export CRYOSPARC_PASSWORD='...'
export CRYOSPARC_PROJECT_UID=CS-prestin
export CRYOSPARC_WORKSPACE_UID=W1
export CRYOSPARC_PROJECT_DIR=/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin
```

`.env.cryosparc` is ignored by git through the existing `.env.*` rule.

## Install the Python API

Use an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install cryosparc-tools
source .env.cryosparc
```

Check the project-local setup:

```bash
python scripts/cryosparc_cli_helper.py status
```

Check a real CryoSPARC API connection:

```bash
python scripts/cryosparc_cli_helper.py python-snippet | python
```

The helper emits a minimal `cryosparc-tools` script that connects, runs
`cs.test_connection()`, opens project `CS-prestin`, and opens the configured
workspace.

## Use the master-host CLI

On the CryoSPARC master host:

```bash
cryosparcm status
cryosparcm cli 'print(db.projects.find_one({"uid": "CS-prestin"})["uid"])'
```

If `cryosparcm` is not on `PATH`, set:

```bash
export CRYOSPARC_MASTER_DIR=/path/to/cryosparc_master
$CRYOSPARC_MASTER_DIR/bin/cryosparcm status
```

Prefer `cryosparc-tools` for scripts that should run from this repository or
from compute nodes. Prefer `cryosparcm` for master administration and quick
database inspection on the master host.

## Project import/export conventions

For CryoSPARC Import Result Group workflows in `CS-prestin`, keep import files
inside the CryoSPARC project directory:

```bash
mkdir -p /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/imports/my_import
cp filtered_particles.cs filtered_particles.csg \
  /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/imports/my_import/
```

The `.csg` must reference the colocated `.cs` by local filename:

```text
filtered_particles.cs
```

Do not use an absolute `.cs` path inside the `.csg` for this workflow. If the
`.csg` contains an absolute path, CryoSPARC may prepend the job directory and
fail the import.

Current exported groups commonly used by this repo:

```text
J42 micrographs: /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_micrographs
J42 particles:   /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J42_particles
J71 particles:   /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J71_particles
J70 particles:   /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J70_particles
J80 exposures:   /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/J80_exposures_accepted
```

## Common workflow

1. Export a CryoSPARC result group to
   `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/exports/groups/...`.
2. Run the repo script that consumes `.cs/.csg` metadata.
3. Put generated import files under
   `/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/imports/<name>/`.
4. Verify the `.csg` references the local `.cs` filename.
5. Import the `.csg` with CryoSPARC's Import Result Group job.

For the existing vesicle-mask particle filter, the final step is usually:

```bash
python segmentation_integration/scripts/filter_cryosparc_picks_by_vesicle_masks.py \
  --dataset raw \
  --vesicle-mask-dir /path/to/vesicle_masks_filtered \
  --output-dir /nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/imports/my_filtered_particles \
  --psize-a 1.068 \
  --downsample 4 \
  --membrane-band-a 40
```

Then import:

```text
/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin/imports/my_filtered_particles/filtered_particles.csg
```

## Sources

- PyPI `cryosparc-tools`: https://pypi.org/project/cryosparc-tools/
- CryoSPARC tools documentation: https://tools.cryosparc.com/
- CryoSPARC tools repository: https://github.com/cryoem-uoft/cryosparc-tools
