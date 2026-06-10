#!/usr/bin/env python3
"""Project-local CryoSPARC CLI setup helper.

This script does not store credentials or call CryoSPARC. It validates the
environment variables used by this repository and prints copy-pasteable command
templates for either cryosparc-tools or the master-host `cryosparcm` CLI.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
from pathlib import Path


DEFAULT_PROJECT_DIR = Path("/nfs/bartesaghilab2/zy223/cryosparc/Prestin/CS-prestin")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def detect_cryosparcm() -> str | None:
    master_dir = env("CRYOSPARC_MASTER_DIR")
    if master_dir:
        candidate = Path(master_dir) / "bin" / "cryosparcm"
        if candidate.exists():
            return str(candidate)
    return shutil.which("cryosparcm")


def get_cryosparc_tools_version() -> str | None:
    try:
        return importlib.metadata.version("cryosparc-tools")
    except importlib.metadata.PackageNotFoundError:
        return None


def print_status() -> None:
    project_uid = env("CRYOSPARC_PROJECT_UID", "CS-prestin")
    workspace_uid = env("CRYOSPARC_WORKSPACE_UID", "W1")
    project_dir = Path(env("CRYOSPARC_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)) or DEFAULT_PROJECT_DIR)
    cryosparcm = detect_cryosparcm()
    tools_version = get_cryosparc_tools_version()

    print("CryoSPARC project defaults")
    print(f"  project uid:   {project_uid}")
    print(f"  workspace uid: {workspace_uid}")
    print(f"  project dir:   {project_dir}")
    print()
    print("Detected local tools")
    print(f"  cryosparc-tools: {tools_version or 'not installed'}")
    print(f"  cryosparcm:      {cryosparcm or 'not found on PATH or CRYOSPARC_MASTER_DIR'}")
    print()

    missing = [
        name
        for name in ("CRYOSPARC_HOST", "CRYOSPARC_BASE_PORT", "CRYOSPARC_EMAIL", "CRYOSPARC_PASSWORD")
        if not env(name)
    ]
    if missing:
        print("Missing cryosparc-tools environment variables:")
        for name in missing:
            print(f"  {name}")
        print()

    print("Recommended local setup")
    print("  python -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install cryosparc-tools")
    print("  source .env.cryosparc")
    print("  python scripts/cryosparc_cli_helper.py status")
    print()

    print("Connection check with cryosparc-tools")
    print("  python scripts/cryosparc_cli_helper.py python-snippet | python")
    print()

    print("Master-host CLI check")
    if cryosparcm:
        print(f"  {cryosparcm} status")
        print(f"  {cryosparcm} cli 'print(db.projects.find_one({{\"uid\": \"{project_uid}\"}})[\"uid\"])'")
    else:
        print("  export CRYOSPARC_MASTER_DIR=/path/to/cryosparc_master")
        print("  $CRYOSPARC_MASTER_DIR/bin/cryosparcm status")
    print()

    import_dir = project_dir / "imports" / "filtered_particles_example"
    print("Import Result Group path convention")
    print(f"  mkdir -p {import_dir}")
    print(f"  cp filtered_particles.cs filtered_particles.csg {import_dir}/")
    print("  ensure filtered_particles.csg references filtered_particles.cs by local filename")


def print_python_snippet() -> None:
    project_uid = env("CRYOSPARC_PROJECT_UID", "CS-prestin")
    workspace_uid = env("CRYOSPARC_WORKSPACE_UID", "W1")
    print(
        f"""from cryosparc.tools import CryoSPARC
import os

cs = CryoSPARC(
    host=os.environ["CRYOSPARC_HOST"],
    base_port=int(os.environ["CRYOSPARC_BASE_PORT"]),
    email=os.environ["CRYOSPARC_EMAIL"],
    password=os.environ["CRYOSPARC_PASSWORD"],
)
assert cs.test_connection()
project = cs.find_project("{project_uid}")
workspace = project.find_workspace("{workspace_uid}")
print("connected", project.uid, workspace.uid)
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("status", "python-snippet"),
        nargs="?",
        default="status",
        help="Show setup status or emit a cryosparc-tools connection snippet.",
    )
    args = parser.parse_args()
    if args.command == "status":
        print_status()
    else:
        print_python_snippet()


if __name__ == "__main__":
    main()
