"""Quick visual check of RQ2 and RQ3 numerical outputs.

Run from the repository root::

    python scripts/verify_rq2_rq3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import RESULTS_DIR  # noqa: E402

with open(RESULTS_DIR / "rq2_rq3_results.json", "r", encoding="utf-8") as f:
    r = json.load(f)

print(json.dumps(r, ensure_ascii=False, indent=2))
