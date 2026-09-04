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
RQ3 因果推断增强分析：
  1. 中断时间序列（ITS）分段回归 - 基于活跃周（排除零事件周）
  2. 格兰杰因果检验
  3. 春节效应鲁棒性检验
  4. 事件密度ITS（事件数加权的网络连接强度）
"""
import json, warnings, sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
warnings.filterwarnings('ignore')

# stdout UTF-8
sys.stdout.reconfigure(encoding='utf-8')

LOCKDOWN_DATE  = datetime(2020, 1, 23)
SPRING_FEST    = datetime(2020, 1, 24)   # 2020春节（正月初一）
np.random.seed(42)

# ============================================================
# 1. 数据加载
# ============================================================
with open(DATA_DIR / "events_extracted.json",  "r", encoding="utf-8") as f:
    EVENTS = json.load(f)
with open(DATA_DIR / "cases_metadata.json",     "r", encoding="utf-8") as f:
    CASES = json.load(f)

dated_events = sorted([e for e in EVENTS if e.get("date")], key=lambda x: x["date"])

# ============================================================
# 2. 生成周度指标（3种：edges, density, avg_degree）
# ============================================================
def compute_week_metrics(events_in_week):
    lt = defaultdict(list)
    for ev in events_in_week:
        for v in ev.get("venues", []):
            lt[(ev["province"], v, ev["date"])].append(ev["case_id"])

    conn = defaultdict(set)
    for key, cl in lt.items():
        for c in cl:
            for other in cl:
                if c != other:
                    conn[c].add(other)

    n_conn = sum(len(v) for v in conn.values()) // 2   # 无向边数
    n_nodes = len(conn)                                  # 有连接的节点数

    # LCC: union-find across all nodes in window (including isolated)
    all_nodes = list({e["case_id"] for e in events_in_week})
    if not all_nodes:
        return {"n_edges": 0, "n_nodes": 0, "density": 0.0,
                "avg_degree": 0.0, "lcc_ratio": 0.0, "n_events": 0, "n_cases": 0}

    parent = {c: c for c in all_nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py

    for cl in lt.values():
        for i in range(len(cl)):
            for j in range(i+1, len(cl)):
                union(cl[i], cl[j])

    comps = defaultdict(int)
    for c in all_nodes:
        comps[find(c)] += 1
    lcc = max(comps.values()) if comps else 0
    lcc_ratio = lcc / len(all_nodes)

    # 活跃节点密度（活跃节点数/总节点数）
    n_total = len(all_nodes)
    density = n_conn / max(n_total * (n_total - 1) // 2, 1)
    avg_deg = np.mean([len(v) for v in conn.values()]) if conn else 0.0

    return {
        "n_edges": n_conn, "n_nodes": n_conn, "density": density,
        "avg_degree": avg_deg, "lcc_ratio": lcc_ratio,
        "n_events": len(events_in_week),
        "n_cases": n_total,
    }


def build_weekly_series(metric="density", gap=7):
    """gap=7天一周"""
    dates = sorted(set(e["date"] for e in dated_events))
    if not dates: return []
    first, last = datetime.strptime(dates[0],"%Y-%m-%d"), datetime.strptime(dates[-1],"%Y-%m-%d")
    series = []
    t = 0
    cur = first
    while cur <= last:
        window = [e for e in dated_events
                  if cur <= datetime.strptime(e["date"],"%Y-%m-%d") < cur + timedelta(days=gap)]
        m = compute_week_metrics(window)
        lock = cur >= LOCKDOWN_DATE
        sf   = cur >= SPRING_FEST
        # 距封城第几周（封城前=0）
        t_since = (cur - LOCKDOWN_DATE).days // gap if lock else 0
        series.append({
            "week_start":   cur.strftime("%Y-%m-%d"),
            "week_num":     t,
            metric:         m[metric],
            "post":         1 if lock else 0,
            "spring_festival": 1 if sf else 0,
            "post_weeks":   t_since,
            "n_events":     m["n_events"],
            "n_cases":      m["n_cases"],
            "n_edges":      m["n_edges"],
            "avg_degree":   m["avg_degree"],
            "lcc_ratio":    m["lcc_ratio"],
        })
        t += 1
        cur += timedelta(days=gap)
    return series


# ============================================================
# 3. ITS 分段回归
# ============================================================
def its_regression(series, metric="density"):
    """
    Segmented regression:
      Y_t = β0 + β1*time + β2*post + β3*time_since_post + ε
    β2: 封城即刻效应（截距跳跃）
    β3: 封城后趋势变化（斜率改变）
    仅在活跃周（n_edges > 0）上回归
    """
    active = [s for s in series if s["n_edges"] > 0]
    if len(active) < 6:
        return {"note": f"活跃周不足（{len(active)}个），ITS不可靠", "active_weeks": len(active),
                "n_pre": sum(1 for s in active if s["post"]==0),
                "n_post": sum(1 for s in active if s["post"]==1)}

    y   = np.array([s[metric] for s in active], dtype=float)
    t   = np.arange(len(active), dtype=float)
    post = np.array([s["post"] for s in active], dtype=float)
    t_since = np.array([s["post_weeks"] for s in active], dtype=float)

    X = np.column_stack([np.ones(len(t)), t, post, t_since])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    n, p = len(y), X.shape[1]
    mse = ss_res / max(n - p, 1)
    se = np.sqrt(mse * np.diag(np.linalg.pinv(X.T @ X + 1e-12*np.eye(4))))
    t_vals = beta / (se + 1e-12)
    p_vals = [2*(1 - stats.t.cdf(abs(ts), df=n-p)) for ts in t_vals]

    names = ["Intercept", "Time-trend", "Lockdown-Level", "Lockdown-Trend"]
    res = {n: {"coef": float(beta[i]), "se": float(se[i]),
                "t": float(t_vals[i]), "p": float(p_vals[i]),
                "sig": "***" if p_vals[i]<.001 else "**" if p_vals[i]<.01 else "*" if p_vals[i]<.05 else ""}
           for i, n in enumerate(names)}
    res["R2"] = float(r2)
    res["active_weeks"] = len(active)
    res["n_pre"] = int(post.sum() == 0 and len(active) > 0 and sum(1 for s in active if s["post"]==0))
    res["n_post"] = int(sum(1 for s in active if s["post"]==1))

    # 封城前自身趋势
    pre_mask = np.array([s["post"]==0 for s in active])
    post_mask = np.array([s["post"]==1 for s in active])
    if pre_mask.sum() > 2:
        tp = np.arange(len(active))[pre_mask]
        res["pre_trend"] = float(np.cov(tp, y[pre_mask])[0,1] / np.var(tp))
    if post_mask.sum() > 2:
        tp2 = np.arange(len(active))[post_mask]
        res["post_trend"] = float(np.cov(tp2, y[post_mask])[0,1] / np.var(tp2))

    return res


# ============================================================
# 4. 格兰杰因果检验
# ============================================================
def granger_test(series, metric="density"):
    """Granger causality: 封城(post) 是否能改善对 Y 的预测"""
    y  = np.array([s[metric] for s in series], dtype=float)
    x  = np.array([s["post"]  for s in series], dtype=float)
    n  = len(y)
    results = {}
    for lag in [1, 2, 3]:
        if n <= 2*lag + 2:
            results[f"lag{lag}"] = {"f": np.nan, "p": np.nan, "note": "数据不足"}
            continue
        yr = y[lag:]
        # Restricted: Y_t = a + sum_k Y_{t-k}
        Xr = np.ones((len(yr), 1+lag))
        for k in range(1, lag+1):
            Xr[:, k] = y[lag-k : n-k]
        beta_r = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        ss_r = np.sum((yr - Xr @ beta_r)**2)
        df_r = len(yr) - Xr.shape[1]
        # Unrestricted: + X_{t-k}
        Xu = np.ones((len(yr), 1+2*lag))
        Xu[:, :1+lag] = Xr
        for k in range(1, lag+1):
            Xu[:, lag+k] = x[lag-k : n-k]
        beta_u = np.linalg.lstsq(Xu, yr, rcond=None)[0]
        ss_u = np.sum((yr - Xu @ beta_u)**2)
        df_u = len(yr) - Xu.shape[1]
        f_stat = ((ss_r - ss_u) / lag) / (ss_u / max(df_u, 1)) if ss_u > 0 else 0.0
        p_val = 1 - stats.f.cdf(f_stat, lag, df_u) if df_u > 0 else np.nan
        sig = "***" if p_val<.001 else "**" if p_val<.01 else "*" if p_val<.05 else ""
        results[f"lag{lag}"] = {
            "f": float(f_stat), "p": float(p_val), "lag": lag,
            "sig": sig, "significant": bool(p_val < 0.05),
            "df_num": lag, "df_den": int(df_u),
        }
    return results


# ============================================================
# 5. 春节效应鲁棒性检验（封城前线性趋势）
# ============================================================
def pre_trend_check(series, metric="density"):
    """检验封城前是否存在线性趋势（若有显著下降趋势，结果应更保守）"""
    pre = [s for s in series if s["post"] == 0 and s["n_edges"] > 0]
    if len(pre) < 3:
        return {"note": "封城前活跃周不足", "pre_weeks": len(pre)}
    t_pre = np.arange(len(pre), dtype=float)
    y_pre = np.array([s[metric] for s in pre], dtype=float)
    slope, intercept, r, p, se = stats.linregress(t_pre, y_pre)
    return {
        "slope": float(slope), "p": float(p), "r2": float(r**2),
        "pre_weeks": len(pre),
        "interpretation": ("封城前无显著趋势 → 下降更可能是封城即时效应"
                           if p > 0.05 else "封城前存在显著趋势 → 需控制该趋势后解读"),
    }


# ============================================================
# 6. ITS + 春节协变量
# ============================================================
def its_with_sf(series, metric="density"):
    """加入春节哑变量的ITS，控制季节效应"""
    active = [s for s in series if s["n_edges"] > 0]
    if len(active) < 6:
        return {"note": "活跃周不足"}

    y  = np.array([s[metric] for s in active], dtype=float)
    t  = np.arange(len(active), dtype=float)
    post = np.array([s["post"] for s in active], dtype=float)
    t_since = np.array([s["post_weeks"] for s in active], dtype=float)
    sf = np.array([s["spring_festival"] for s in active], dtype=float)

    X = np.column_stack([np.ones(len(t)), t, post, t_since, sf])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    n, p = len(y), X.shape[1]
    mse = ss_res / max(n-p, 1)
    se = np.sqrt(mse * np.diag(np.linalg.pinv(X.T @ X + 1e-12*np.eye(5))))
    t_vals = beta / (se + 1e-12)
    p_vals = [2*(1 - stats.t.cdf(abs(ts), df=n-p)) for ts in t_vals]
    names = ["Intercept", "Time-trend", "Lockdown-Level", "Lockdown-Trend", "SpringFest"]
    res = {n: {"coef": float(beta[i]), "se": float(se[i]),
                "t": float(t_vals[i]), "p": float(p_vals[i]),
                "sig": "***" if p_vals[i]<.001 else "**" if p_vals[i]<.01 else "*" if p_vals[i]<.05 else ""}
           for i, n in enumerate(names)}
    res["R2"] = float(r2)
    return res


# ============================================================
# 主程序
# ============================================================
print("=" * 60)
print("RQ3 Enhanced Causal Inference Analysis")
print("=" * 60)

# --- 生成时间序列 ---
print("\n[1] Building weekly time series...")
weekly_dens = build_weekly_series(metric="density", gap=7)
weekly_edges = build_weekly_series(metric="n_edges", gap=7)
weekly_deg   = build_weekly_series(metric="avg_degree", gap=7)

print(f"\n  Total weeks: {len(weekly_dens)}")
pre_w  = [w for w in weekly_dens if w["post"]==0]
post_w = [w for w in weekly_dens if w["post"]==1]
pre_active  = [w for w in pre_w  if w["n_edges"]>0]
post_active = [w for w in post_w if w["n_edges"]>0]
print(f"  Pre-lockdown total weeks:  {len(pre_w)}  (active: {len(pre_active)})")
print(f"  Post-lockdown total weeks: {len(post_w)} (active: {len(post_active)})")

# 打印序列
print("\n  Weekly network metrics:")
print(f"  {'Week':>4} {'Date':>10} {'n_edges':>8} {'density':>10} {'avg_deg':>8} {'period':>5}")
print(f"  {'-'*55}")
for w in weekly_dens:
    m = "***" if w["post"]==1 else "pre"
    print(f"  {w['week_num']:>4} {w['week_start']:>10} {w['n_edges']:>8} "
          f"{w['density']:>10.6f} {w['avg_degree']:>8.3f} {m}")

# --- Pre/post summary ---
print("\n[2] Pre vs. Post-lockdown summary (active weeks only)...")
metrics = ["density", "n_edges", "avg_degree"]
summary = {}
for m in metrics:
    pre_vals  = [w[m] for w in pre_active]
    post_vals = [w[m] for w in post_active]
    pre_mean  = np.mean(pre_vals) if pre_vals else 0.0
    post_mean = np.mean(post_vals) if post_vals else 0.0
    pct_chg   = (post_mean - pre_mean) / (pre_mean + 1e-9) * 100
    summary[m] = {"pre_mean": float(pre_mean), "post_mean": float(post_mean),
                   "pct_change": float(pct_chg),
                   "n_pre": len(pre_vals), "n_post": len(post_vals)}
    print(f"  {m}: pre={pre_mean:.4f}, post={post_mean:.4f}, change={pct_chg:+.1f}%")

# --- ITS ---
print("\n[3] ITS Segmented Regression (active weeks)...")
print("  Model: Y_t = β0 + β1*time + β2*post + β3*time_since_post")
for m in ["density", "avg_degree"]:
    print(f"\n  --- Metric: {m} ---")
    res = its_regression(weekly_dens if m=="density" else weekly_deg, metric=m)
    if "note" in res:
        print(f"    {res['note']}")
        continue
    print(f"    R2 = {res['R2']:.4f}, active weeks={res['active_weeks']}")
    for name in ["Intercept","Time-trend","Lockdown-Level","Lockdown-Trend"]:
        r = res[name]
        print(f"    {name:18s}: coef={r['coef']:.6f}, p={r['p']:.4f} {r['sig']}")
    if "pre_trend" in res:
        print(f"    Pre-lockdown weekly trend:  {res['pre_trend']:.6f}")
    if "post_trend" in res:
        print(f"    Post-lockdown weekly trend: {res['post_trend']:.6f}")

# --- Granger ---
print("\n[4] Granger Causality Test...")
for m in ["density", "avg_degree"]:
    print(f"\n  --- Metric: {m} ---")
    gr = granger_test(weekly_dens if m=="density" else weekly_deg, metric=m)
    for lag_name, r in gr.items():
        if r.get("note"):
            print(f"  Lag {r['lag']}: {r['note']}")
        else:
            print(f"  Lag {r['lag']}: F={r['f']:.4f}, p={r['p']:.4f} {r['sig']}"
                  + (" (Significant)" if r.get("significant") else ""))

# --- Pre-trend robustness ---
print("\n[5] Pre-lockdown Trend Robustness Check...")
for m in ["density", "avg_degree"]:
    print(f"\n  --- Metric: {m} ---")
    rob = pre_trend_check(weekly_dens if m=="density" else weekly_deg, metric=m)
    if "note" in rob:
        print(f"    {rob['note']}")
    else:
        print(f"    Pre-lockdown slope: {rob['slope']:.6f} per week, p={rob['p']:.4f}")
        print(f"    R2: {rob['r2']:.4f}")
        print(f"    {rob['interpretation']}")

# --- ITS + SF ---
print("\n[6] ITS with Spring Festival Covariate...")
for m in ["density"]:
    print(f"\n  --- Metric: {m} ---")
    cov = its_with_sf(weekly_dens, metric=m)
    if "note" in cov:
        print(f"    {cov['note']}")
    else:
        print(f"    R2 = {cov['R2']:.4f}")
        for name in ["Intercept","Time-trend","Lockdown-Level","Lockdown-Trend","SpringFest"]:
            r = cov[name]
            print(f"    {name:18s}: coef={r['coef']:.6f}, p={r['p']:.4f} {r['sig']}")

# --- 汇总保存 ---
out = {
    "weekly_density": [{k:v for k,v in w.items() if k!='spring_festival'} for w in weekly_dens],
    "pre_post_summary": summary,
    "its_density": its_regression(weekly_dens, metric="density"),
    "its_avg_degree": its_regression(weekly_deg, metric="avg_degree"),
    "granger_density": granger_test(weekly_dens, metric="density"),
    "granger_avg_degree": granger_test(weekly_deg, metric="avg_degree"),
    "pre_trend_density": pre_trend_check(weekly_dens, metric="density"),
    "pre_trend_avg_degree": pre_trend_check(weekly_deg, metric="avg_degree"),
    "its_sf_density": its_with_sf(weekly_dens, metric="density"),
}

with open(OUTPUT_DIR / "rq3_causal_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Saved: {OUTPUT_DIR / 'rq3_causal_results.json'}")
print("=" * 60)
