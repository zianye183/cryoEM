#!/usr/bin/env python3
"""REPL-friendly initial denoising workflow for Topaz-Denoise and DRACO.

This file is intended to be run cell-by-cell in Zed's Python REPL using the
`# %%` markers, or executed as a regular script. It keeps the workflow safe:

- creates/refreshes a small MRC subset by symlink
- checks which denoising dependencies are available
- inspects MRC dimensions without requiring `mrcfile`
- prints conservative Topaz/DRACO command templates
- optionally previews MRC images/stacks when `mrcfile` and `matplotlib` exist

It does not run Topaz or DRACO automatically because the correct command/API
must be confirmed in the environment where those tools are installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import shutil
import struct
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast


# %% Configuration
@dataclass(frozen=True)
class DenoisingConfig:
    project_root: Path = Path("/nfs/bartesaghilab2/zy223/denoising")
    source_mrc_dir: Path = Path(
        "/nfs/bartesaghilab/micromon/research-bartesaghilab-06/shared/users/"
        + "zy223@duke.edu/projects/Prestin-1fhwPcd8gQ/sp-preprocessing-ghLbktywzmKi35IA/mrc"
    )
    subset_size: int = 5
    random_seed: int = 42
    use_symlinks: bool = True

    @property
    def subset_dir(self) -> Path:
        return self.project_root / "data" / "subset"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def topaz_pretrained_dir(self) -> Path:
        return self.results_dir / "topaz_pretrained"

    @property
    def topaz_retrained_dir(self) -> Path:
        return self.results_dir / "topaz_retrained"

    @property
    def draco_pretrained_dir(self) -> Path:
        return self.results_dir / "draco_pretrained"


CFG = DenoisingConfig()
print(CFG)


# %% Environment checks
def module_status(name: str) -> str:
    return "available" if importlib.util.find_spec(name) is not None else "missing"


def command_status(command: str) -> str:
    found = shutil.which(command)
    return found if found is not None else "not found on PATH"


def command_first_line(command: list[str], timeout: int = 20) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except FileNotFoundError:
        return "not found"
    except subprocess.TimeoutExpired:
        return "found, but command timed out"

    output = completed.stdout.strip()
    first_line = output.splitlines()[0] if output else "no output"
    return f"exit={completed.returncode}: {first_line}"


def check_environment() -> dict[str, str]:
    checks = {
        "topaz": command_status("topaz"),
        "draco command": command_status("draco"),
        "python mrcfile": module_status("mrcfile"),
        "python torch": module_status("torch"),
        "python matplotlib": module_status("matplotlib"),
        "python numpy": module_status("numpy"),
    }

    print("Environment checks")
    print("==================")
    for key, value in checks.items():
        print(f"{key:18s} {value}")

    if checks["topaz"] != "not found on PATH":
        print("\nTopaz help/version probe")
        print(command_first_line(["topaz", "--help"]))

    draco_repo = CFG.project_root / "external" / "draco"
    print(f"\nExpected local DRACO repo: {draco_repo}")
    print("present" if draco_repo.exists() else "not present")
    return checks


ENV_CHECKS = check_environment()


# %% Create a small reproducible subset
def list_source_mrcs(source_dir: Path = CFG.source_mrc_dir) -> list[Path]:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    files = sorted(source_dir.glob("*.mrc"))
    if not files:
        raise FileNotFoundError(f"No .mrc files found in {source_dir}")
    return files


def create_subset(
    source_dir: Path = CFG.source_mrc_dir,
    subset_dir: Path = CFG.subset_dir,
    n: int = CFG.subset_size,
    seed: int = CFG.random_seed,
    use_symlinks: bool = CFG.use_symlinks,
    overwrite: bool = True,
) -> list[Path]:
    files = list_source_mrcs(source_dir)
    rng = random.Random(seed)
    selected = rng.sample(files, k=min(n, len(files)))

    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest = subset_dir / "subset_manifest.tsv"

    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("index\tname\tsource_path\tsubset_path\n")
        for idx, src in enumerate(selected, start=1):
            target = subset_dir / src.name
            if target.exists() or target.is_symlink():
                if not overwrite:
                    raise FileExistsError(f"Destination exists: {target}")
                target.unlink()

            if use_symlinks:
                target.symlink_to(src)
            else:
                _ = shutil.copy2(src, target)

            _ = handle.write(f"{idx}\t{src.name}\t{src}\t{target}\n")

    print(f"Selected {len(selected)} files from {len(files)} available MRCs")
    print(f"Subset directory: {subset_dir}")
    print(f"Manifest: {manifest}")
    return [subset_dir / path.name for path in selected]


SUBSET_FILES = create_subset()
_ = SUBSET_FILES


# %% Inspect MRC headers without external dependencies
def read_mrc_header(path: Path) -> dict[str, int | str]:
    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) < 16:
        raise ValueError("file too small for MRC header")
    nx, ny, nz, mode = cast(tuple[int, int, int, int], struct.unpack("<4i", header))
    interpretation = "2D micrograph" if nz == 1 else "3D stack/volume"
    return {
        "name": path.name,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "mode": mode,
        "interpretation": interpretation,
    }


def inspect_subset_headers(
    paths: Iterable[Path] = SUBSET_FILES,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    print("name\tnx\tny\tnz\tmode\tinterpretation")
    for path in paths:
        try:
            row = read_mrc_header(path)
            _ = rows.append(row)
            print(
                f"{row['name']}\t{row['nx']}\t{row['ny']}\t{row['nz']}\t"
                f"{row['mode']}\t{row['interpretation']}"
            )
        except Exception as exc:  # noqa: BLE001 - inspection should report all failures
            print(f"{path.name}\tERROR\t{exc}")
    return rows


HEADER_ROWS = inspect_subset_headers()
_ = HEADER_ROWS


# %% Build command templates for first denoising tests
def build_topaz_pretrained_command(paths: Iterable[Path] = SUBSET_FILES) -> str:
    CFG.topaz_pretrained_dir.mkdir(parents=True, exist_ok=True)
    file_list = " ".join(str(path) for path in paths)
    return (
        "topaz denoise "
        + "--model unet --device 0 --patch-size 4000 --patch-padding 500 "
        + "--suffix _topaz_denoised "
        + f"--output {CFG.topaz_pretrained_dir} {file_list}"
    )


def print_command_templates(paths: Iterable[Path] = SUBSET_FILES) -> None:
    CFG.topaz_pretrained_dir.mkdir(parents=True, exist_ok=True)
    CFG.topaz_retrained_dir.mkdir(parents=True, exist_ok=True)
    CFG.draco_pretrained_dir.mkdir(parents=True, exist_ok=True)

    print("# Topaz-Denoise pretrained test")
    print("# First confirm exact flags in the active Topaz environment:")
    print("topaz denoise --help")
    print()
    print(build_topaz_pretrained_command(paths))

    print("\n# Topaz-Denoise retraining")
    print("# Do not start retraining until paired noisy observations are prepared.")
    print("# Typical input is even/odd frame sums from the same movies.")
    print("# Check installed CLI help before wiring this up:")
    print("topaz denoise-train --help")

    print("\n# DRACO pretrained inference")
    print("# DRACO appears repo/API-notebook driven rather than a stable CLI.")
    print("# Set up the repo/weights locally, then adapt its Denoise/B inference code.")
    print(f"# Expected output directory: {CFG.draco_pretrained_dir}")
    print(f"# Suggested repo path: {CFG.project_root / 'external' / 'draco'}")


print_command_templates()


# %% Optional visual preview when mrcfile/matplotlib are installed
def preview_mrc(path: Path, frame: int | None = None) -> None:
    """Display one MRC file if optional plotting dependencies are installed.

    For 3D stacks, `frame=None` displays the middle slice. This is intended for
    quick qualitative inspection in the REPL, not publication-quality figures.
    """
    if (
        module_status("mrcfile") == "missing"
        or module_status("matplotlib") == "missing"
    ):
        print("Preview skipped: install/activate mrcfile and matplotlib first.")
        return

    plt = cast(Any, importlib.import_module("matplotlib.pyplot"))
    mrcfile = cast(ModuleType, importlib.import_module("mrcfile"))

    with cast(Any, mrcfile).open(path, permissive=True) as mrc:
        data = cast(Any, mrc).data

    if getattr(data, "ndim", 0) == 3:
        selected_frame = data.shape[0] // 2 if frame is None else frame
        image = data[selected_frame]
        title = f"{path.name} frame {selected_frame}"
    else:
        image = data
        title = path.name

    plt.figure(figsize=(7, 7))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# Uncomment in the REPL after optional dependencies are available:
# preview_mrc(SUBSET_FILES[0])


# %% Script entry point summary
if __name__ == "__main__":
    print("\nInitial denoising workflow scaffold complete.")
    print("Run individual # %% cells in Zed's REPL for interactive inspection.")
