#!/usr/bin/env python3
"""Create a reproducible small MRC subset for denoising experiments.

By default this creates symlinks instead of copying potentially large files.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Directory containing input .mrc files"
    )
    parser.add_argument("--dest", required=True, help="Destination subset directory")
    parser.add_argument("--n", type=int, default=5, help="Number of files to select")
    parser.add_argument("--seed", type=int, default=20260526, help="Random seed")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (not recommended for large MRCs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks/files in destination if names collide",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")

    files = sorted(source.glob("*.mrc"))
    if not files:
        raise SystemExit(f"No .mrc files found in {source}")

    rng = random.Random(args.seed)
    subset = rng.sample(files, k=min(args.n, len(files)))
    dest.mkdir(parents=True, exist_ok=True)

    manifest = dest / "subset_manifest.tsv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("index\tname\tsource_path\tsubset_path\n")
        for idx, src in enumerate(subset, start=1):
            target = dest / src.name
            if target.exists() or target.is_symlink():
                if args.overwrite:
                    target.unlink()
                else:
                    raise SystemExit(f"Destination exists; use --overwrite: {target}")

            if args.copy:
                # Import lazily so the default symlink path has no extra overhead.
                import shutil

                shutil.copy2(src, target)
            else:
                target.symlink_to(src)

            handle.write(f"{idx}\t{src.name}\t{src}\t{target}\n")

    print(f"Selected {len(subset)} files from {len(files)} available MRCs")
    print(f"Subset directory: {dest}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
