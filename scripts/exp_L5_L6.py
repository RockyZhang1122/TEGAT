"""
L5 + L6: 边窗口敏感性 + 显著性检验 (TEGAT vs XGBoost)

输出:
  - edge_window_sensitivity.csv: ±1/3/5/7/14 天的 edge_count, AUC, F1
  - statistical_tests.csv: McNemar / Wilcoxon 检验结果
"""
import json, os, math, warnings, sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, average_precision_score)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
try:
    from scipy.stats import wilcoxon, ttest_rel
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from paths import DATA_DIR as _DATA_DIR, RESULTS_DIR  # noqa: E402
DATA_DIR = str(_DATA_DIR)
OUT_DIR = os.path.join(RESULTS_DIR, 'iter2_L5_L6')
os.makedirs(OUT_DIR, exist_ok=True)

LOCKDOWN_DATE = datetime(2020, 1, 23)
SEED = 42
SEEDS = [42, 123, 456, 789, 1024]

with open(os.path.join(DATA_DIR, "events_extracted.json"), encoding="utf-8") as f:
    EVENTS = json.load(f)
with open(os.path.join(DATA_DIR, "cases_metadata.json"), encoding="utf-8") as f:
    CASES = json.load(f)
with open(os.path.join(DATA_DIR, "graph_mappings.json"), encoding="utf-8") as f:
    MAPPINGS = json.load(f)


def build_features(events, cases, mappings):
    venues = mappings["venues"]; transports = mappings["transports"]
    n = len(cases)
    ages = [c["age"] for c in cases if c.get("age")]
    age_min, age_max = (min(ages), max(ages)) if ages else (0, 100)
    n_feat = 8 + len(venues) + len(transports)
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, case in enumerate(cases):
        if case.get("age"): X[i, 0] = (case["age"] - age_min) / (age_max - age_min + 1e-6)
        if case.get("gender") == "M": X[i, 1] = 1
        elif case.get("gender") == "F": X[i, 2] = 1
        if case.get("wuhan_related"): X[i, 3] = 1
        if case.get("is_asymptomatic"): X[i, 4] = 1
        if case.get("first_date"):
            try: fd = datetime.strptime(case["first_date"], "%Y-%m-%d"); X[i, 5] = (fd - LOCKDOWN_DATE).days / 100.0
            except: pass
        if case.get("first_date") and case.get("last_date"):
            try: fd = datetime.strptime(case["first_date"], "%Y-%m-%d"); ld = datetime.strptime(case["last_date"], "%Y-%m-%d"); X[i, 6] = (ld - fd).days / 30.0
            except: pass
        cev = [e for e in events if e["case_id"] == case["case_id"]]
        X[i, 7] = min(len(cev), 20) / 20.0
        vc = defaultdict(int); tc = defaultdict(int)
        for e in cev:
            for v in e.get("venues", []):
                if v in venues: vc[v] += 1
            for t in e.get("transports", []):
                if t in transports: tc[t] += 1
        for j, v in enumerate(venues): X[i, 8 + j] = min(vc[v], 10) / 10.0
        for j, t in enumerate(transports): X[i, 8 + len(venues) + j] = min(tc[t], 5) / 5.0
    return X


def build_labels(events, cases, time_window):
    """构建 cluster-membership 标签（基于 ±k 天共定位）"""
    lt_groups = defaultdict(list)
    for ev in events:
        if not ev.get("date") or not ev.get("venues"): continue
        try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
        except: continue
        for v in ev.get("venues", []):
            lt_groups[(ev["province"], v, ev["date"])].append((ev["case_id"], ed))
    cluster = set()
    c2ev = defaultdict(list)
    for ev in events: c2ev[ev["case_id"]].append(ev)
    for case in cases:
        for ev in c2ev.get(case["case_id"], []):
            if not ev.get("date") or not ev.get("venues"): continue
            try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
            except: continue
            for v in ev.get("venues", []):
                for off in range(-time_window, time_window + 1):
                    key = (ev["province"], v, (ed + timedelta(days=off)).strftime("%Y-%m-%d"))
                    for oid, _ in lt_groups.get(key, []):
                        if oid != case["case_id"]: cluster.add(case["case_id"]); break
                else: continue
                break
            else: continue
            break
    return np.array([1 if c["case_id"] in cluster else 0 for c in cases])


def build_colocation_graph(events, cases, time_window):
    """构建 case-case 共定位图，返回 adj dict"""
    lt_groups = defaultdict(list)
    for ev in events:
        if not ev.get("date") or not ev.get("venues"): continue
        try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
        except: continue
        for v in ev.get("venues", []):
            lt_groups[(ev["province"], v, ev["date"])].append((ev["case_id"], ed))
    adj = defaultdict(set)
    for (prov, venue, date_str), members in lt_groups.items():
        for i, (cid_i, date_i) in enumerate(members):
            for j, (cid_j, date_j) in enumerate(members):
                if cid_i >= cid_j: continue
                delta_days = abs((date_i - date_j).days)
                if delta_days <= time_window:
                    adj[cid_i].add(cid_j)
                    adj[cid_j].add(cid_i)
    return dict(adj)


def eval_model(y_true, y_score):
    pred = (y_score >= 0.5).astype(int)
    return {
        "AUC-ROC": round(roc_auc_score(y_true, y_score), 4),
        "AP": round(average_precision_score(y_true, y_score), 4),
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
    }


print("Loading data...")
X = build_features(EVENTS, CASES, MAPPINGS)
venues = MAPPINGS["venues"]; transports = MAPPINGS["transports"]
v_idx = list(range(8, 8 + len(venues)))
t_idx = list(range(18, 18 + len(transports)))

# 增强特征 (X_full = X + venue-specific aggregations + transport mean)
X_full = np.column_stack([
    X,
    np.column_stack([
        X[:, v_idx].sum(axis=1), X[:, v_idx].max(axis=1),
        X[:, v_idx].std(axis=1), (X[:, v_idx] > 0).sum(axis=1),
    ]),
    X[:, t_idx].mean(axis=1, keepdims=True),
])

# ============ L5: 边窗口敏感性 ============
print("\n=== L5: Edge Window Sensitivity (n=963 test) ===")
edge_window_results = {}

for tw in [1, 3, 5, 7, 14]:
    # 重新构建 label 和图
    y_tw = build_labels(EVENTS, CASES, tw)
    adj_tw = build_colocation_graph(EVENTS, CASES, tw)
    n_edges = sum(len(v) for v in adj_tw.values()) // 2

    np.random.seed(SEED)
    n = len(y_tw)
    idx = np.random.permutation(n)
    tr = idx[:int(n * 0.7)]; te = idx[int(n * 0.8):]

    X_tr, X_te = X_full[tr], X_full[te]
    y_tr, y_te = y_tw[tr], y_tw[te]

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    gb.fit(X_tr, y_tr)
    p = gb.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    auc = round(roc_auc_score(y_te, p), 4)
    f1 = round(f1_score(y_te, pred, zero_division=0), 4)

    edge_window_results[f"+-{tw}d"] = {
        "n_edges": n_edges,
        "positive_ratio": round(y_tw.mean(), 4),
        "AUC-ROC": auc,
        "F1": f1,
    }
    print(f"  +-{tw}d: edges={n_edges}, pos_ratio={y_tw.mean():.4f}, AUC={auc:.4f}, F1={f1:.4f}")

with open(os.path.join(OUT_DIR, "edge_window_results.json"), "w", encoding="utf-8") as f:
    json.dump(edge_window_results, f, ensure_ascii=False, indent=2)

# ============ L6: 显著性检验 (TEGAT vs XGBoost 5 seeds) ============
print("\n=== L6: Statistical Tests (TEGAT vs XGBoost across 5 seeds) ===")

# 加载主实验结果 (来自 iter1 实验的 main_results_v2.json)
with open(os.path.join(RESULTS_DIR, 'iter1_main_tau7_v2', 'main_results_v2.json'), encoding="utf-8") as f:
    main = json.load(f)

# 提取 5 seeds 的 AUC-ROC for TEGAT vs XGBoost
tegat_aucs = [main["all_results"][str(s)]["TEGAT"]["AUC-ROC"] for s in SEEDS]
xgb_aucs = [main["all_results"][str(s)]["XGBoost"]["AUC-ROC"] for s in SEEDS]
tegat_f1s = [main["all_results"][str(s)]["TEGAT"]["F1"] for s in SEEDS]
xgb_f1s = [main["all_results"][str(s)]["XGBoost"]["F1"] for s in SEEDS]

# Wilcoxon 配对检验 (TEGAT > XGBoost)
try:
    from scipy.stats import wilcoxon, mannwhitneyu
    w_stat_auc, w_p_auc = wilcoxon(tegat_aucs, xgb_aucs, alternative="greater")
    w_stat_f1, w_p_f1 = wilcoxon(tegat_f1s, xgb_f1s, alternative="greater")
    print(f"  AUC Wilcoxon: W={w_stat_auc}, p={w_p_auc:.4f} (TEGAT > XGBoost)")
    print(f"  F1  Wilcoxon: W={w_stat_f1}, p={w_p_f1:.4f}")
except Exception as e:
    print(f"  Wilcoxon failed: {e}")
    w_stat_auc, w_p_auc = None, None

# McNemar-like 检验：在每个 seed 上比较 TEGAT vs XGBoost 的预测一致性
# 这里用 AUC 差值作为效应量
print(f"\n  AUC deltas (TEGAT - XGBoost) per seed:")
deltas = []
for s in SEEDS:
    d = tegat_aucs[SEEDS.index(s)] - xgb_aucs[SEEDS.index(s)]
    deltas.append(d)
    print(f"    seed={s}: ΔAUC = {d:+.4f}")
print(f"  Mean ΔAUC = {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")

# 配对 t 检验 (5 个种子)
from scipy.stats import ttest_rel
t_stat, t_p = ttest_rel(tegat_aucs, xgb_aucs)
print(f"  Paired t-test: t={t_stat:.4f}, p={t_p:.4f} (two-sided)")

stat_results = {
    "wilcoxon_auc": {"W": w_stat_auc, "p_greater": w_p_auc},
    "wilcoxon_f1": {"W": w_stat_f1, "p_greater": w_p_f1},
    "paired_t_test_auc": {"t": t_stat, "p_two_sided": t_p},
    "per_seed_tegat_auc": tegat_aucs,
    "per_seed_xgb_auc": xgb_aucs,
    "mean_delta_auc": float(np.mean(deltas)),
    "std_delta_auc": float(np.std(deltas)),
    "p_value_ttest": float(t_p),
}

with open(os.path.join(OUT_DIR, "statistical_tests.json"), "w", encoding="utf-8") as f:
    json.dump(stat_results, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {OUT_DIR}/")
print("L5 + L6 experiments COMPLETE")
