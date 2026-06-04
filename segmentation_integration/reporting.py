"""Report writers for segmentation integration workflows."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    path: Path,
    *,
    inputs: dict[str, object],
    outputs: dict[str, object],
    reason_counts: Counter,
    micrographs: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": inputs,
        "outputs": outputs,
        "reason_counts": dict(sorted(reason_counts.items())),
        "micrographs": micrographs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

