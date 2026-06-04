#!/usr/bin/env python3
"""Check availability of tools needed for initial denoising tests."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except FileNotFoundError:
        return "not found"
    except subprocess.TimeoutExpired:
        return "found, but version command timed out"

    first_line = (
        completed.stdout.strip().splitlines()[0]
        if completed.stdout.strip()
        else "no output"
    )
    return f"exit={completed.returncode}: {first_line}"


def module_status(name: str) -> str:
    spec = importlib.util.find_spec(name)
    return "available" if spec is not None else "missing"


def main() -> None:
    checks = {
        "topaz": shutil.which("topaz") or "not found on PATH",
        "draco command": shutil.which("draco")
        or "not found on PATH (DRACO may be notebook/repo-based)",
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
        print("\nTopaz version/help:")
        print(command_version(["topaz", "--help"]))

    repo = Path("/nfs/bartesaghilab2/zy223/denoising/external/draco")
    print(f"\nExpected local DRACO repo: {repo}")
    print("present" if repo.exists() else "not present")


if __name__ == "__main__":
    main()
