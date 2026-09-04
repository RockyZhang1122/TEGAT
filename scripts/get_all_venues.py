"""Print every venue category referenced across the dataset stats and the
RQ1 venue-risk result file.

Run from the repository root::

    python scripts/get_all_venues.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402

print("=== dataset_stats.json ===")
with open(DATA_DIR / "dataset_stats.json", "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))

print("\n=== rq1_venue_risk.json ===")
with open(RESULTS_DIR / "rq1_venue_risk.json", "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
