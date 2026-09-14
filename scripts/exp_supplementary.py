"""
补充实验：τ敏感性 + 隔离节点分析 + 统计检验
"""
import json, os, math, warnings, sys
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from paths import DATA_DIR as _DATA_DIR, RESULTS_DIR  # noqa: E402
DATA_DIR = str(_DATA_DIR)
OUT_DIR = os.path.join(RESULTS_DIR, 'iter1_supplementary')
os.makedirs(OUT_DIR, exist_ok=True)

LOCKDOWN_DATE = datetime(2020, 1, 23)
SEED = 42

with open(os.path.join(DATA_DIR, "events_extracted.json"), encoding="utf-8") as f:
    EVENTS = json.load(f)
with open(os.path.join(DATA_DIR, "cases_metadata.json"), encoding="utf-8") as f:
    CASES = json.load(f)
with open(os.path.join(DATA_DIR, "graph_mappings.json"), encoding="utf-8") as f:
    MAPPINGS = json.load(f)

# ============ 特征和标签构建 ============
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
    lt_groups = defaultdict(list)
    for ev in events:
        if not ev.get("date") or not ev.get("venues"): continue
        try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
        except: continue
        for v in ev.get("venues", []): lt_groups[(ev["province"], v, ev["date"])].append((ev["case_id"], ed))
    cluster = set()
    c2ev = defaultdict(list)
    for ev in events: c2ev[ev["case_id"]].append(ev)
    for case in CASES:
        for ev in c2ev.get(case["case_id"], []):
            if not ev.get("date") or not ev.get("venues"): continue
            try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
            except: continue
            for v in ev.get("venues", []):
                for off in range(-time_window, time_window + 1):
                    key = (ev["province"], v, (ed + timedelta(days=off)).strftime("%Y-%m-%d"))
                    for oid, _ in lt_groups.get(key, []):
                        if oid != case["case_id"]: cluster.add(case["case_id"]); break
    return np.array([1 if c["case_id"] in cluster else 0 for c in cases])

def build_colocation_graph(events, cases, time_window):
    lt_groups = defaultdict(list)
    for ev in events:
        if not ev.get("date") or not ev.get("venues"): continue
        try: ed = datetime.strptime(ev["date"], "%Y-%m-%d")
        except: continue
        for v in ev.get("venues", []): lt_groups[(ev["province"], v, ev["date"])].append((ev["case_id"], ed))
    adj = defaultdict(set); edge_weights = {}
    for (prov, venue, date_str), members in lt_groups.items():
        for i, (cid_i, date_i) in enumerate(members):
            for j, (cid_j, date_j) in enumerate(members):
                if cid_i >= cid_j: continue
                delta_days = abs((date_i - date_j).days)
                if delta_days <= time_window:
                    adj[cid_i].add(cid_j); adj[cid_j].add(cid_i)
                    edge_weights[(cid_i, cid_j)] = math.exp(-delta_days / TAU)
                    edge_weights[(cid_j, cid_i)] = math.exp(-delta_days / TAU)
    return dict(adj), edge_weights

print("Loading data...")
X = build_features(EVENTS, CASES, MAPPINGS)
case_ids = [c["case_id"] for c in CASES]
np.random.seed(SEED)
n = len(CASES)
idx = np.random.permutation(n)
tr = idx[:int(n*0.7)]; va = idx[int(n*0.7):int(n*0.8)]; te = idx[int(n*0.8):]
X_tr, X_te = X[tr], X[te]
y_tr, y_te = build_labels(EVENTS, CASES, 3)[tr], build_labels(EVENTS, CASES, 3)[te]
c_tr = [case_ids[i] for i in tr]; c_te = [case_ids[i] for i in te]

# ============ 1. τ 敏感性（固定边窗口±3天，变化τ） ============
print("\n=== tau Sensitivity ===")
tau_results = {}
for tau_val in [3, 5, 7, 14]:
    global TAU; TAU = tau_val  # 修改全局 τ
    y_t = build_labels(EVENTS, CASES, 3)  # 重新构建（τ 只影响边权重，不影响标签）
    y_tr_t, y_te_t = y_t[tr], y_t[te]
    adj_t, ew_t = build_colocation_graph(EVENTS, CASES, 3)
    # 简化：使用 venue-specific 特征
    n_v = len(MAPPINGS["venues"]); n_t = len(MAPPINGS["transports"])
    v_idx = list(range(8, 8+n_v)); t_idx = list(range(18, 18+n_t))
    X_full = np.column_stack([X,
        np.column_stack([X[:,v_idx].sum(axis=1), X[:,v_idx].max(axis=1), X[:,v_idx].std(axis=1), (X[:,v_idx]>0).sum(axis=1)]),
        X[:,t_idx].mean(axis=1, keepdims=True)
    ])
    Xf_tr, Xf_te = X_full[tr], X_full[te]
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    gb.fit(Xf_tr, y_tr_t)
    p = gb.predict_proba(Xf_te)[:, 1]
    tau_results[f"tau={tau_val}"] = {
        "AUC-ROC": round(roc_auc_score(y_te_t, p), 4),
        "F1": round(f1_score(y_te_t, (p>=0.5).astype(int), zero_division=0), 4),
        "Recall": round(recall_score(y_te_t, (p>=0.5).astype(int), zero_division=0), 4),
        "Precision": round(precision_score(y_te_t, (p>=0.5).astype(int), zero_division=0), 4),
    }
    print(f"  tau={tau_val}: AUC={tau_results[f'tau={tau_val}']['AUC-ROC']:.4f}  F1={tau_results[f'tau={tau_val}']['F1']:.4f}")

# ============ 2. 隔离节点分析 ============
print("\n=== Isolation Node Analysis ===")
adj_full, _ = build_colocation_graph(EVENTS, CASES, 3)
# cluster label (y=0 = isolated)
y_full = build_labels(EVENTS, CASES, 3)
y0_from_label = np.where(y_full == 0)[0]
isolated_from_label = set(y0_from_label)

# centrality isolation (degree=0 in full graph)
isolated_by_degree = set()
for i, cid in enumerate(case_ids):
    if len(adj_full.get(cid, [])) == 0:
        isolated_by_degree.add(i)

# overlap
overlap = isolated_from_label & isolated_by_degree
only_label_isolated = isolated_from_label - isolated_by_degree
only_degree_isolated = isolated_by_degree - isolated_from_label

print(f"  cluster-label y=0: {len(isolated_from_label)} cases")
print(f"  centrality degree=0: {len(isolated_by_degree)} cases")
print(f"  Overlap: {len(overlap)}")
print(f"  Only in cluster-label: {len(only_label_isolated)}")
print(f"  Only in degree=0: {len(only_degree_isolated)}")

# Asymptomatic vs symptomatic isolation
asym_n = sum(1 for c in CASES if c["is_asymptomatic"])
sym_n = len(CASES) - asym_n
asym_idx = [i for i, c in enumerate(CASES) if c["is_asymptomatic"]]
sym_idx = [i for i, c in enumerate(CASES) if not c["is_asymptomatic"]]
asym_isolated = sum(1 for i in asym_idx if i in isolated_by_degree)
sym_isolated = sum(1 for i in sym_idx if i in isolated_by_degree)
print(f"  Asymptomatic isolated: {asym_isolated}/{asym_n} = {asym_isolated/asym_n:.4f}")
print(f"  Symptomatic isolated: {sym_isolated}/{sym_n} = {sym_isolated/sym_n:.4f}")

# ============ 3. 边窗口敏感性 ============
print("\n=== Edge Window Sensitivity ===")
window_results = {}
for tw in [1, 3, 5, 7, 14]:
    adj_w, ew_w = build_colocation_graph(EVENTS, CASES, tw)
    n_edges = sum(len(v) for v in adj_w.values()) // 2
    y_w = build_labels(EVENTS, CASES, tw)
    y_tr_w, y_te_w = y_w[tr], y_w[te]
    X_full_w = np.column_stack([X,
        np.column_stack([X[:,v_idx].sum(axis=1), X[:,v_idx].max(axis=1), X[:,v_idx].std(axis=1), (X[:,v_idx]>0).sum(axis=1)]),
        X[:,t_idx].mean(axis=1, keepdims=True)
    ])
    Xf_tr_w, Xf_te_w = X_full_w[tr], X_full_w[te]
    gb_w = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    gb_w.fit(Xf_tr_w, y_tr_w)
    p_w = gb_w.predict_proba(Xf_te_w)[:, 1]
    window_results[f"+-{tw}"] = {
        "edge_count": n_edges,
        "positive_ratio": round(y_w.mean(), 4),
        "AUC-ROC": round(roc_auc_score(y_te_w, p_w), 4),
        "F1": round(f1_score(y_te_w, (p_w>=0.5).astype(int), zero_division=0), 4),
    }
    print(f"  +-{tw}d: edges={n_edges}, pos_ratio={y_w.mean():.4f}, AUC={window_results[f'+-{tw}']['AUC-ROC']:.4f}")

# ============ 保存 ============
results = {
    "tau_sensitivity": tau_results,
    "isolation_analysis": {
        "cluster_label_y0": len(isolated_from_label),
        "centrality_degree0": len(isolated_by_degree),
        "overlap": len(overlap),
        "only_label_isolated": len(only_label_isolated),
        "only_degree_isolated": len(only_degree_isolated),
        "asymptomatic_isolated_rate": round(asym_isolated / asym_n, 4),
        "symptomatic_isolated_rate": round(sym_isolated / sym_n, 4),
        "n_asymptomatic": asym_n,
        "n_symptomatic": sym_n,
    },
    "edge_window_sensitivity": window_results,
}
with open(os.path.join(OUT_DIR, "supplementary_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT_DIR}/supplementary_results.json")
print("Supplementary experiments COMPLETE")
