"""Verify the RQ2 centrality distributions and the betweenness distribution
printed in the paper.

Run from the repository root::

    python scripts/verify_betweenness.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import DATA_DIR, RESULTS_DIR, REPO_ROOT, RAW_TEXT_DIR  # noqa: E402

import numpy as np  # noqa: E402

# -----------------------------------------------------------------------
# 1. RQ2 distribution summary from the cached result file
# -----------------------------------------------------------------------
with open(RESULTS_DIR / "rq2_rq3_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== RQ2: 各中心性指标精确分布 ===")
for metric in ["degree", "page_rank", "betweenness", "clustering"]:
    asym = data["rq2"][metric]["asymptomatic_mean"]
    asym_std = data["rq2"][metric]["asymptomatic_std"]
    sym = data["rq2"][metric]["symptomatic_mean"]
    sym_std = data["rq2"][metric]["symptomatic_std"]
    print(f"  {metric}:")
    print(f"    Asym: mean={asym:.6f}, std={asym_std:.6f}, "
          f"range estimated from std: [{asym-asym_std:.4f}, {asym+asym_std:.4f}]")
    print(f"    Sym:  mean={sym:.6f}, std={sym_std:.6f}")

# -----------------------------------------------------------------------
# 2. Recompute betweenness from raw cases + events
# -----------------------------------------------------------------------
print("\n=== Betweenness 详细分析 ===")

with open(DATA_DIR / "cases_metadata.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
    events = json.load(f)

print(f"  Loaded {len(cases)} cases and {len(events)} events")

# 重建网络
connections = defaultdict(set)
location_time_cases = defaultdict(list)
for event in events:
    if event.get("date") and event.get("venues"):
        for v in event["venues"]:
            try:
                d = datetime.strptime(event["date"], "%Y-%m-%d")
                for offset in range(-3, 4):
                    key = (event["province"], v, (d + timedelta(days=offset)).strftime("%Y-%m-%d"))
                    location_time_cases[key].append(event["case_id"])
            except Exception:
                pass

for key, case_list in location_time_cases.items():
    for c in case_list:
        for other in case_list:
            if c != other:
                connections[c].add(other)

case_ids = [c["case_id"] for c in cases]
total_edges = sum(len(v) for v in connections.values()) // 2
print(f"  Total nodes: {len(case_ids)}")
print(f"  Total edges (undirected): {total_edges}")

asym_cases_set = {c["case_id"] for c in cases if c["is_asymptomatic"]}
sym_cases_set = {c["case_id"] for c in cases if not c["is_asymptomatic"]}


def bfs_count(s, connections, all_cases):
    """Count shortest paths from s."""
    sigma = {c: 0 for c in all_cases}
    sigma[s] = 1
    dist = {c: -1 for c in all_cases}
    dist[s] = 0
    pred = {c: [] for c in all_cases}
    queue = [s]
    head = 0
    while head < len(queue):
        v = queue[head]
        head += 1
        for w in connections.get(v, set()):
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)
    return pred, sigma


# Brandes for all nodes
bc = {c: 0.0 for c in case_ids}
for s in case_ids:
    pred, sigma = bfs_count(s, connections, case_ids)
    delta = {c: 0.0 for c in case_ids}
    stack = []
    visited = {c: False for c in case_ids}
    visited[s] = True
    q = [s]
    head = 0
    while head < len(q):
        v = q[head]
        head += 1
        stack.append(v)
        for w in connections.get(v, set()):
            if not visited.get(w, False):
                visited[w] = True
                q.append(w)
    while stack:
        w = stack.pop()
        for v in pred.get(w, []):
            delta[v] += (sigma[v] / max(sigma[w], 1)) * (1 + delta[w])
        if w != s:
            bc[w] += delta[w]

# 报告 betweenness 分布
asym_bc = [bc[c] for c in case_ids if c in asym_cases_set]
sym_bc = [bc[c] for c in case_ids if c in sym_cases_set]

print("\n  Betweenness distribution:")
print(f"    All cases: min={min(bc.values()):.6f}, max={max(bc.values()):.6f}")
print(f"    Non-zero cases: {sum(1 for v in bc.values() if v > 0)}/{len(bc)}")
print(f"    Asym: min={min(asym_bc):.6f}, max={max(asym_bc):.6f}, mean={np.mean(asym_bc):.6f}")
print(f"    Sym:  min={min(sym_bc):.6f}, max={max(sym_bc):.6f}, mean={np.mean(sym_bc):.6f}")

# Top-10 betweenness
sorted_bc = sorted(bc.items(), key=lambda x: -x[1])[:10]
print("\n  Top-10 betweenness:")
for cid, val in sorted_bc:
    is_asym = cid in asym_cases_set
    print(f"    {cid}: {val:.6f} ({'asym' if is_asym else 'sym'})")
