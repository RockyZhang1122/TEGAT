"""Sanity-check the bundled extraction_stats.json and dataset_stats.json.

Run from the repository root::

    python scripts/check_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DATA_DIR  # noqa: E402

with open(DATA_DIR / "extraction_stats.json", "r", encoding="utf-8") as f:
    extraction = json.load(f)

with open(DATA_DIR / "dataset_stats.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("=== extraction_stats.json ===")
print(json.dumps(extraction, ensure_ascii=False, indent=2))
print("\n=== dataset_stats.json ===")
print(json.dumps(dataset, ensure_ascii=False, indent=2))
