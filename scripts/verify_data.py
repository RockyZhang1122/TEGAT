"""Compact verifier for the bundled structured dataset & ablation results.

Run from the repository root::

    python scripts/verify_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402

with open(RESULTS_DIR / "ablation_results.json", "r", encoding="utf-8") as f:
    r = json.load(f)

print("=== tau_sensitivity ===")
for k, v in r["tau_sensitivity"].items():
    print(f"  {k}: AUC={v['AUC-ROC']:.4f} AP={v['AP']:.4f} F1={v['F1']:.4f} "
          f"Acc={v['Accuracy']:.4f} Recall={v['Recall']:.4f}")

print("\n=== feature_group ablation ===")
for k, v in r["ablation_feature_groups"].items():
    print(f"  {k}: AUC={v['AUC-ROC']:.4f} Acc={v['Accuracy']:.4f} F1={v['F1']:.4f} Recall={v['Recall']:.4f}")

print("\n=== component ablation ===")
for k, v in r["ablation_components"].items():
    print(f"  {k}: AUC={v['AUC-ROC']:.4f} Acc={v['Accuracy']:.4f} F1={v['F1']:.4f} Recall={v['Recall']:.4f}")

# Check event type distribution
with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
    events = json.load(f)

from collections import Counter
types = Counter()
for ev in events:
    t = ev.get("event_type", "")
    types[t] += 1

total = sum(types.values())
print(f"\n=== Event type distribution ===")
print(f"  Total events: {total}")
for t, c in types.most_common():
    pct = c / total * 100 if total > 0 else 0
    print(f"  {t}: {c} ({pct:.1f}%)")

# Also check the extraction_stats.json
with open(DATA_DIR / "extraction_stats.json", "r", encoding="utf-8") as f:
    stats = json.load(f)
print(f"\n=== extraction_stats.json ===")
print(json.dumps(stats, ensure_ascii=False, indent=2))
