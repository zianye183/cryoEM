#!/usr/bin/env python3
"""Set up local DRACO code and Denoise/B checkpoint paths.

The DRACO source and model weights are released for non-commercial use with
redistribution restrictions. This script downloads them into this workspace; do
not commit or redistribute downloaded checkpoint files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from denoising_run_common import DEFAULT_PROJECT, quote_command

DRACO_REPO_URL = "https://github.com/Cellverse/draco.git"
DENOISE_B_DRIVE_ID = "1z6DunE6skfKNODC6vOJ_48QXHRBzBPyK"
DEFAULT_REPO = DEFAULT_PROJECT / "external" / "draco"
DEFAULT_WEIGHTS = DEFAULT_PROJECT / "models" / "draco_denoise_b.ckpt"

INSTALL_COMMANDS = """\
# DRACO already has a project-local venv:
source /nfs/bartesaghilab2/zy223/venvs/draco/bin/activate

# If you need to rebuild it, create a venv and install the project requirements:
python3 -m venv /nfs/bartesaghilab2/zy223/venvs/draco
/nfs/bartesaghilab2/zy223/venvs/draco/bin/python -m pip install --upgrade pip wheel setuptools
/nfs/bartesaghilab2/zy223/venvs/draco/bin/python -m pip install -r /nfs/bartesaghilab2/zy223/denoising/configs/draco_venv_requirements.txt
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=str(DEFAULT_REPO), help="DRACO clone destination."
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Destination for Denoise/B checkpoint downloaded from Google Drive.",
    )
    parser.add_argument(
        "--skip-repo",
        action="store_true",
        help="Do not clone/update the DRACO repository.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Do not download the Denoise/B checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing checkpoint and allow replacing an incomplete repo directory.",
    )
    parser.add_argument(
        "--print-install-commands",
        action="store_true",
        help="Print the recommended venv/pip environment setup commands and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing git/gdown.",
    )
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print(quote_command(command))
    if not dry_run:
        subprocess.run(command, check=True)


def ensure_repo(repo: Path, *, force: bool, dry_run: bool) -> None:
    if repo.exists() and (repo / ".git").is_dir():
        print(f"DRACO repo already exists: {repo}")
        return
    if repo.exists():
        if not force:
            raise SystemExit(
                f"Destination exists but is not a git clone: {repo}. "
                "Use --force to remove it, or pass a different --repo."
            )
        print(f"Removing non-git destination before clone: {repo}")
        if not dry_run:
            if repo.is_dir():
                shutil.rmtree(repo)
            else:
                repo.unlink()

    repo.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", DRACO_REPO_URL, str(repo)], dry_run=dry_run)


def ensure_weights(weights: Path, *, force: bool, dry_run: bool) -> None:
    if weights.exists() and not force:
        print(f"Denoise/B weights already exist: {weights}")
        return

    weights.parent.mkdir(parents=True, exist_ok=True)
    gdown = shutil.which("gdown")
    if gdown:
        run(
            [
                gdown,
                "--id",
                DENOISE_B_DRIVE_ID,
                "--output",
                str(weights),
            ],
            dry_run=dry_run,
        )
        return

    python = sys.executable
    print("gdown is not on PATH; trying `python -m gdown`.")
    run(
        [
            python,
            "-m",
            "gdown",
            "--id",
            DENOISE_B_DRIVE_ID,
            "--output",
            str(weights),
        ],
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()
    if args.print_install_commands:
        print(INSTALL_COMMANDS)
        return

    repo = Path(args.repo).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()

    print(
        "DRACO license reminder: source and weights are non-commercial/restricted-use; do not redistribute checkpoint files."
    )
    if not args.skip_repo:
        ensure_repo(repo, force=args.force, dry_run=args.dry_run)
    if not args.skip_weights:
        ensure_weights(weights, force=args.force, dry_run=args.dry_run)

    print("\nNext DRACO dry run:")
    print(
        quote_command(
            [
                sys.executable,
                str(DEFAULT_PROJECT / "scripts" / "run_draco_pretrained.py"),
                str(DEFAULT_PROJECT / "data" / "subset"),
                "--repo",
                str(repo),
                "--weights",
                str(weights),
                "--dry-run",
            ]
        )
    )


if __name__ == "__main__":
    main()
