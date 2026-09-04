"""Decode and print the human-readable labels from
``province_stats.json`` (province names) and ``rq1_venue_risk.json`` (venue
names) and write a combined label file under results/.

Run from the repository root::

    python scripts/decode_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402

SOURCES = [
    (DATA_DIR / "province_stats.json",     "province"),
    (RESULTS_DIR / "rq1_venue_risk.json",  "venue"),
]

output_lines = []
for path, kind in SOURCES:
    if not path.exists():
        print(f"[skip] {path} (missing)")
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=== {kind} labels ({path.name}) ===")
    if kind == "province":
        for k, v in (data.get("per_province", data) or {}).items():
            print(f"  {k}: {v}")
            output_lines.append(f"province\t{k}\t{v}")
    else:
        # venue risk output
        venues = data.get("venues", data)
        for k, v in (venues.items() if isinstance(venues, dict) else []):
            print(f"  {k}: {v}")
            output_lines.append(f"venue\t{k}\t{v}")

output_path = RESULTS_DIR / "labels_utf8.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\nWrote {output_path}")
