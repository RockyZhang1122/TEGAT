"""Module-level path bootstrap for portable execution.

All source scripts in this repository use the :mod:`paths` module
so that the project can be cloned into any directory without modifying
absolute paths.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from paths import (
    DATA_DIR, RESULTS_DIR, REPO_ROOT,
    data_path, result_path, paper_path, figure_path,
)

# Legacy aliases preserved for body text inside this module.
OUTPUT_DIR = RESULTS_DIR
BASE = REPO_ROOT

# -*- coding: utf-8 -*-
"""
RQ2: 无症状感染者网络结构分析
RQ3: 武汉封城前后网络结构变化
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


LOCKDOWN_DATE = datetime(2020, 1, 23)


def build_network_stats(events, cases):
    """构建网络指标"""
    # case-case connection: 同省同场所时间相近
    connections = defaultdict(set)
    location_time_cases = defaultdict(list)
    
    for event in events:
        if event["date"] and event["venues"]:
            for v in event["venues"]:
                try:
                    d = datetime.strptime(event["date"], "%Y-%m-%d")
                    for offset in range(-3, 4):
                        key = (event["province"], v, (d + timedelta(days=offset)).strftime("%Y-%m-%d"))
                        location_time_cases[key].append(event["case_id"])
                except:
                    pass
    
    for key, case_list in location_time_cases.items():
        for c in case_list:
            for other in case_list:
                if c != other:
                    connections[c].add(other)
    
    return connections


def compute_betweenness(connections, all_cases):
    """
    Brandes 2001 BFS-based betweenness centrality approximation.
    Computes exact betweenness on the co-location graph by:
      for each source s: BFS tree + accumulate pair-dependency
    Returns dict {case_id: betweenness}.
    """
    import math
    bc = {c: 0.0 for c in all_cases}
    for s in all_cases:
        # BFS stack from s
        stack = []
        pred = {c: [] for c in all_cases}
        sigma = {c: 0 for c in all_cases}
        sigma[s] = 1
        dist = {c: -1 for c in all_cases}
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            v = queue[head]; head += 1
            stack.append(v)
            for w in connections.get(v, set()):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {c: 0.0 for c in all_cases}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / max(sigma[w], 1)) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    # Normalize by 2/((n-1)(n-2)) for undirected graph
    n = max(len(all_cases) - 1, 1)
    norm = 2.0 / (n * (n - 1)) if n > 1 else 1.0
    return {c: v * norm for c, v in bc.items()}


def compute_centrality(case_id, connections, all_cases, betweenness_map):
    """计算单个病例的中心性指标"""
    neighbors = connections.get(case_id, set())
    degree = len(neighbors)

    # PageRank-like (简化版)
    pr = 1.0 / len(all_cases)
    for _ in range(5):
        new_pr = 1.0 / len(all_cases)
        for neighbor in neighbors:
            new_pr += 0.85 * pr * (1.0 / max(len(connections.get(neighbor, set())), 1))
        pr = new_pr

    # True betweenness centrality (normalized) from precomputed Brandes map
    betweenness = betweenness_map.get(case_id, 0.0)

    # Clustering: 邻居之间的连接
    neighbor_neighbors = set()
    for n in neighbors:
        neighbor_neighbors.update(connections.get(n, set()))
    neighbor_neighbors.discard(case_id)
    clustering = len(neighbor_neighbors & neighbors) / max(len(neighbors), 1)

    return {
        "degree": degree,
        "page_rank": pr,
        "betweenness": betweenness,
        "clustering": clustering,
    }


def main():
    print("=" * 60)
    print("RQ2: Asymptomatic Network Analysis")
    print("RQ3: Intervention Effect Analysis")
    print("=" * 60)
    
    with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(DATA_DIR / "cases_metadata.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    # 构建网络
    connections = build_network_stats(events, cases)
    case_ids = [c["case_id"] for c in cases]

    # Precompute true betweenness centrality once (Brandes BFS) on full graph
    print("Computing betweenness centrality (Brandes BFS)...")
    betweenness_map = compute_betweenness(connections, case_ids)

    # ==================== RQ2: 无症状分析 ====================

    print("\n=== RQ2: Asymptomatic Analysis ===")

    asym_cases = [c for c in cases if c["is_asymptomatic"]]
    sym_cases = [c for c in cases if not c["is_asymptomatic"]]

    print(f"\nAsymptomatic cases: {len(asym_cases)}")
    print(f"Symptomatic cases: {len(sym_cases)}")

    # 计算中心性
    asym_metrics = []
    for c in asym_cases:
        m = compute_centrality(c["case_id"], connections, case_ids, betweenness_map)
        asym_metrics.append(m)

    sym_metrics = []
    for c in sym_cases:
        m = compute_centrality(c["case_id"], connections, case_ids, betweenness_map)
        sym_metrics.append(m)
    
    # 统计检验
    rq2_results = {}
    for metric in ["degree", "page_rank", "betweenness", "clustering"]:
        asym_vals = [m[metric] for m in asym_metrics]
        sym_vals = [m[metric] for m in sym_metrics]
        
        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(asym_vals, sym_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(asym_vals) - np.mean(sym_vals)
        pooled_std = np.sqrt((np.var(asym_vals) + np.var(sym_vals)) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-9)
        
        rq2_results[metric] = {
            "asymptomatic_mean": float(np.mean(asym_vals)),
            "asymptomatic_std": float(np.std(asym_vals)),
            "symptomatic_mean": float(np.mean(sym_vals)),
            "symptomatic_std": float(np.std(sym_vals)),
            "mannwhitney_p": float(p_value),
            "cohens_d": float(cohens_d),
        }
        
        print(f"\n{metric}:")
        print(f"  Asymptomatic: {np.mean(asym_vals):.4f} +/- {np.std(asym_vals):.4f}")
        print(f"  Symptomatic: {np.mean(sym_vals):.4f} +/- {np.std(sym_vals):.4f}")
        print(f"  Mann-Whitney p: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        if p_value < 0.05:
            sig = "SIG" if cohens_d > 0.2 else "SIG(small)"
            print(f"  [{sig}] Significant difference")
    
    # 无症状病例的独有特征
    asym_degrees = [m["degree"] for m in asym_metrics]
    asym_zero_deg = sum(1 for d in asym_degrees if d == 0)
    sym_degrees = [m["degree"] for m in sym_metrics]
    sym_zero_deg = sum(1 for d in sym_degrees if d == 0)
    
    print(f"\n  Asymptomatic isolated (degree=0): {asym_zero_deg}/{len(asym_cases)} ({asym_zero_deg/len(asym_cases)*100:.1f}%)")
    print(f"  Symptomatic isolated (degree=0): {sym_zero_deg}/{len(sym_cases)} ({sym_zero_deg/len(sym_cases)*100:.1f}%)")
    
    rq2_results["isolation_rate"] = {
        "asymptomatic": asym_zero_deg/len(asym_cases),
        "symptomatic": sym_zero_deg/len(sym_cases),
    }
    
    # ==================== RQ3: 干预效果分析 ====================
    
    print("\n\n=== RQ3: Intervention Effect Analysis ===")
    
    # 按时间段分组
    pre_events = [e for e in events if e["date"] and e["is_lockdown"] == False]
    post_events = [e for e in events if e["date"] and e["is_lockdown"] == True]
    
    print(f"\nPre-lockdown events: {len(pre_events)}")
    print(f"Post-lockdown events: {len(post_events)}")
    
    # 计算网络密度变化
    def compute_network_metrics(event_list, case_list):
        """计算网络指标"""
        connections_local = defaultdict(set)
        location_time_cases = defaultdict(list)
        
        for event in event_list:
            if event["date"] and event["venues"]:
                for v in event["venues"]:
                    location_time_cases[(event["province"], v, event["date"])].append(event["case_id"])
        
        for key, case_list_local in location_time_cases.items():
            for c in case_list_local:
                for other in case_list_local:
                    if c != other:
                        connections_local[c].add(other)
        
        n_cases = len(case_list)
        n_connections = sum(len(v) for v in connections_local.values()) // 2
        max_connections = n_cases * (n_cases - 1) // 2
        
        density = n_connections / max(max_connections, 1)
        avg_degree = np.mean([len(v) for v in connections_local.values()]) if connections_local else 0
        
        # 最大连通分量（简化）
        # 使用并查集
        parent = {c: c for c in case_list}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for key, case_list_local in location_time_cases.items():
            for i in range(len(case_list_local)):
                for j in range(i+1, len(case_list_local)):
                    union(case_list_local[i], case_list_local[j])
        
        components = defaultdict(int)
        for c in case_list:
            root = find(c)
            components[root] += 1
        
        max_component_size = max(components.values()) if components else 0
        lcc_ratio = max_component_size / n_cases if n_cases > 0 else 0
        
        return {
            "density": density,
            "avg_degree": avg_degree,
            "n_connections": n_connections,
            "lcc_ratio": lcc_ratio,
        }
    
    pre_metrics = compute_network_metrics(pre_events, list(set([e["case_id"] for e in pre_events])))
    post_metrics = compute_network_metrics(post_events, list(set([e["case_id"] for e in post_events])))
    
    print(f"\nPre-lockdown network:")
    print(f"  Density: {pre_metrics['density']:.6f}")
    print(f"  Avg Degree: {pre_metrics['avg_degree']:.4f}")
    print(f"  LCC Ratio: {pre_metrics['lcc_ratio']:.4f}")
    print(f"  N Connections: {pre_metrics['n_connections']}")
    
    print(f"\nPost-lockdown network:")
    print(f"  Density: {post_metrics['density']:.6f}")
    print(f"  Avg Degree: {post_metrics['avg_degree']:.4f}")
    print(f"  LCC Ratio: {post_metrics['lcc_ratio']:.4f}")
    print(f"  N Connections: {post_metrics['n_connections']}")
    
    # 计算变化百分比
    rq3_results = {}
    for key in ["density", "avg_degree", "lcc_ratio"]:
        pre_val = pre_metrics[key]
        post_val = post_metrics[key]
        change_pct = (post_val - pre_val) / (pre_val + 1e-9) * 100
        rq3_results[key] = {
            "pre": float(pre_val),
            "post": float(post_val),
            "change_pct": float(change_pct),
        }
        print(f"\n{key}: {change_pct:+.1f}% change")
    
    # 按时间段更细粒度分析
    print("\n--- Weekly Network Metrics ---")
    
    # 生成周数据
    all_dates = sorted(set([e["date"] for e in events if e["date"]]))
    if all_dates:
        from datetime import timedelta
        first_date = datetime.strptime(all_dates[0], "%Y-%m-%d")
        last_date = datetime.strptime(all_dates[-1], "%Y-%m-%d")
        
        weekly_metrics = []
        current = first_date
        while current <= last_date:
            week_start = current
            week_end = current + timedelta(days=6)
            week_events = [
                e for e in events 
                if e["date"] and 
                week_start <= datetime.strptime(e["date"], "%Y-%m-%d") <= week_end
            ]
            
            if week_events:
                week_cases = list(set([e["case_id"] for e in week_events]))
                m = compute_network_metrics(week_events, week_cases)
                m["week"] = week_start.strftime("%Y-%m-%d")
                m["n_events"] = len(week_events)
                m["n_cases"] = len(week_cases)
                m["period"] = "pre" if week_start < LOCKDOWN_DATE else "post"
                weekly_metrics.append(m)
            
            current += timedelta(days=7)
        
        print(f"\nTotal weeks: {len(weekly_metrics)}")
        pre_weeks = [m for m in weekly_metrics if m["period"] == "pre"]
        post_weeks = [m for m in weekly_metrics if m["period"] == "post"]
        print(f"Pre-lockdown weeks: {len(pre_weeks)}")
        print(f"Post-lockdown weeks: {len(post_weeks)}")
        
        # 保存周数据
        rq3_results["weekly"] = weekly_metrics
    
    # 保存结果
    with open(OUTPUT_DIR / "rq2_rq3_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "rq2": rq2_results,
            "rq3": rq3_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/rq2_rq3_results.json")
    print("=" * 60)
    print("RQ2 & RQ3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()