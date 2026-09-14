"""
TEGAT v2: Temporal Event Graph Attention Network

重新实现版 v2，解决 K1/K4 问题：
  K1: 主实验使用 τ=7（不是 τ=3）
  K4: TEGAT-Red 与 XGBoost 必须独立实现（不同架构）

架构设计：
  TEGAT: 异构图嵌入特征 + MLP分类器
  TEGAT-Red: 异构图嵌入特征（无时间衰减）+ MLP分类器
  
图嵌入构建：
  Stage 1: Type-specific 节点投影
  Stage 2: Multi-head attention on case-case subgraph (稀疏 O(E))
  Stage 3: Temporal decay weighting (τ=7)
  最终: 节点嵌入送入 MLPClassifier (sklearn)

训练: BCE + Adam, 5 random seeds, mean±std 报告
"""
import json, os, sys, warnings, math
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from paths import DATA_DIR as DATA_DIR, RESULTS_DIR  # noqa: E402
DATA_DIR = str(DATA_DIR)
OUT_DIR = os.path.join(RESULTS_DIR, 'iter1_main_tau7_v2')
os.makedirs(OUT_DIR, exist_ok=True)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings('ignore')

# ============ 路径 (使用 src/paths.py) ============
SEEDS = [42, 123, 456, 789, 1024]
TAU = 7.0
EDGE_WINDOW = 3
LOCKDOWN_DATE = datetime(2020, 1, 23)

# ============ 数据加载 ============
print("Loading data...")
with open(os.path.join(DATA_DIR, "events_extracted.json"), encoding="utf-8") as f:
    EVENTS = json.load(f)
with open(os.path.join(DATA_DIR, "cases_metadata.json"), encoding="utf-8") as f:
    CASES = json.load(f)
with open(os.path.join(DATA_DIR, "graph_mappings.json"), encoding="utf-8") as f:
    MAPPINGS = json.load(f)

print(f"  Events={len(EVENTS)}, Cases={len(CASES)}")

# ============ 构建 case-case 图 ============
def build_colocation_graph(events, cases, time_window=EDGE_WINDOW):
    """返回 (adj: dict, edge_weights: dict, case_to_idx: dict)"""
    lt_groups = defaultdict(list)
    for ev in events:
        if not ev.get("date") or not ev.get("venues"):
            continue
        try:
            ed = datetime.strptime(ev["date"], "%Y-%m-%d")
        except:
            continue
        for v in ev.get("venues", []):
            lt_groups[(ev["province"], v, ev["date"])].append((ev["case_id"], ed))

    adj = defaultdict(set)
    edge_weights = {}
    case_ids = [c["case_id"] for c in cases]
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}

    for (prov, venue, date_str), members in lt_groups.items():
        for i, (cid_i, date_i) in enumerate(members):
            for j, (cid_j, date_j) in enumerate(members):
                if cid_i >= cid_j:
                    continue
                delta_days = abs((date_i - date_j).days)
                if delta_days <= time_window:
                    adj[cid_i].add(cid_j)
                    adj[cid_j].add(cid_i)
                    w = math.exp(-delta_days / TAU)
                    edge_weights[(cid_i, cid_j)] = w
                    edge_weights[(cid_j, cid_i)] = w
    return dict(adj), edge_weights, case_to_idx


def build_cluster_labels(events, cases, time_window=EDGE_WINDOW):
    """构建 cluster-membership 标签"""
    adj, _, _ = build_colocation_graph(events, cases, time_window)
    return np.array([1 if len(adj.get(c["case_id"], [])) > 0 else 0 for c in cases])


# ============ 特征构建 ============
def build_features(events, cases, mappings):
    """构建病例特征（不含 case-case 边信息）"""
    venues = mappings["venues"]
    transports = mappings["transports"]
    n_cases = len(cases)
    ages = [c["age"] for c in cases if c.get("age")]
    age_min, age_max = (min(ages), max(ages)) if ages else (0, 100)

    n_feat = 8 + len(venues) + len(transports)
    X = np.zeros((n_cases, n_feat), dtype=np.float32)

    for i, case in enumerate(cases):
        if case.get("age"):
            X[i, 0] = (case["age"] - age_min) / (age_max - age_min + 1e-6)
        if case.get("gender") == "M": X[i, 1] = 1
        elif case.get("gender") == "F": X[i, 2] = 1
        if case.get("wuhan_related"): X[i, 3] = 1
        if case.get("is_asymptomatic"): X[i, 4] = 1

        if case.get("first_date"):
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                X[i, 5] = (fd - LOCKDOWN_DATE).days / 100.0
            except:
                pass
        if case.get("first_date") and case.get("last_date"):
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                ld = datetime.strptime(case["last_date"], "%Y-%m-%d")
                X[i, 6] = (ld - fd).days / 30.0
            except:
                pass

        cev = [e for e in events if e["case_id"] == case["case_id"]]
        X[i, 7] = min(len(cev), 20) / 20.0

        vc = defaultdict(int)
        tc = defaultdict(int)
        for e in cev:
            for v in e.get("venues", []):
                if v in venues: vc[v] += 1
            for t in e.get("transports", []):
                if t in transports: tc[t] += 1
        for j, v in enumerate(venues):
            X[i, 8 + j] = min(vc[v], 10) / 10.0
        for j, t in enumerate(transports):
            X[i, 8 + len(venues) + j] = min(tc[t], 5) / 5.0
    return X


# ============ 图嵌入构建 ============
class GraphEmbedder:
    """
    图嵌入器：使用稀疏注意力机制构建 case 节点嵌入
    
    Stage 1: 节点投影 (linear)
    Stage 2: Multi-head attention on case-case subgraph  
    Stage 3: Temporal decay aggregation (TEGAT) 或 vanilla mean (TEGAT-Red)
    """
    def __init__(self, d=64, H=4, alpha=0.2, seed=42):
        self.d = d
        self.H = H
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)
        self.W_proj = None
        self.W_C = None
        self.a = None
        self.scaler_X = StandardScaler()

    def fit(self, X, case_ids, adj, edge_weights):
        """从训练集拟合嵌入器（仅用于 transform，不做梯度下降）"""
        self._init_weights(X.shape[1])
        self.scaler_X.fit(X)
        return self

    def transform(self, X, case_ids, adj, edge_weights, temporal=True):
        """
        将 case 节点映射到 d 维嵌入
        case_ids: 当前 split 的 case ID 列表
        adj: 完整图邻接表
        edge_weights: 完整图边权重
        temporal: True=TEGAT (带时间衰减), False=TEGAT-Red (vanilla)
        """
        X_sc = self.scaler_X.transform(X)
        h = np.tanh(np.dot(X_sc, self.W_proj))  # (N, d)

        # Stage 2: Multi-head attention
        h_att = self._sparse_attention(h, case_ids, adj)

        # Stage 3: Temporal or vanilla aggregation
        if temporal:
            h_agg = self._temporal_agg(h, case_ids, adj, edge_weights)
        else:
            h_agg = self._vanilla_agg(h, case_ids, adj)

        # 拼接: [attention_out, aggregated]
        return np.concatenate([h_att, h_agg], axis=1)  # (N, 2d)

    def _init_weights(self, n_feat):
        scale = 0.01
        self.W_proj = self.rng.randn(n_feat, self.d).astype(np.float32) * scale
        self.W_C = self.rng.randn(self.d, self.d).astype(np.float32) * scale
        self.a = self.rng.randn(2 * self.d, 1).astype(np.float32) * scale

    def _sparse_attention(self, h, case_ids, adj):
        """稀疏多头注意力 (O(E))"""
        case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
        n = len(case_ids)
        h_prime = np.dot(h, self.W_C)
        h_agg = np.zeros((n, self.d), dtype=np.float32)

        for _ in range(self.H):
            for i, cid_i in enumerate(case_ids):
                valid = [c for c in adj.get(cid_i, []) if c in case_to_idx]
                if not valid:
                    h_agg[i] += h_prime[i] / self.H
                    continue
                n_idx = [case_to_idx[c] for c in valid]
                inputs = np.concatenate([np.tile(h_prime[i], (len(valid), 1)), h_prime[n_idx]], axis=1)
                e = np.dot(inputs, self.a).ravel()
                e = np.where(e > 0, e, self.alpha * e)
                e -= np.max(e)
                alpha = np.exp(e)
                alpha /= (np.sum(alpha) + 1e-8)
                for k, _ in enumerate(valid):
                    h_agg[i] += alpha[k] * h_prime[n_idx[k]]
        return np.tanh(h_agg)

    def _temporal_agg(self, h, case_ids, adj, edge_weights):
        """Stage 3: Temporal decay weighted aggregation"""
        case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
        n = len(case_ids)
        h_out = np.zeros((n, self.d), dtype=np.float32)
        for i, cid_i in enumerate(case_ids):
            valid = [c for c in adj.get(cid_i, []) if c in case_to_idx]
            if not valid:
                h_out[i] = h[i]
                continue
            w_sum = 0.0
            for cid_j in valid:
                j = case_to_idx[cid_j]
                w = edge_weights.get((cid_i, cid_j), math.exp(-EDGE_WINDOW / TAU))
                h_out[i] += w * h[j]
                w_sum += w
            h_out[i] = h_out[i] / w_sum if w_sum > 0 else h[i]
        return np.tanh(h_out)

    def _vanilla_agg(self, h, case_ids, adj):
        """Stage 3 (no temporal): Mean aggregation"""
        case_to_idx = {cid: i for i, cid in enumerate(case_ids)}
        n = len(case_ids)
        h_out = np.zeros((n, self.d), dtype=np.float32)
        for i, cid_i in enumerate(case_ids):
            valid = [c for c in adj.get(cid_i, []) if c in case_to_idx]
            if not valid:
                h_out[i] = h[i]
            else:
                n_idx = [case_to_idx[c] for c in valid]
                h_out[i] = np.mean(h[n_idx], axis=0)
        return np.tanh(h_out)


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


# ============ 主实验 ============
print("\n" + "=" * 65)
print(f"TEGAT v2 Main Experiment (tau={TAU}, seeds={SEEDS})")
print("=" * 65)

X_base = build_features(EVENTS, CASES, MAPPINGS)
y = build_cluster_labels(EVENTS, CASES)
adj, edge_weights, case_to_idx_full = build_colocation_graph(EVENTS, CASES)
case_ids_all = [c["case_id"] for c in CASES]
print(f"\nData: {X_base.shape}, y={y.sum()}/{len(y)} positive ({y.mean():.1%})")
print(f"Graph: {sum(1 for c in case_ids_all if len(adj.get(c,[]))>0)} nodes with edges")

all_results = {}

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")

    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    tr = idx[:int(n * 0.7)]
    va = idx[int(n * 0.7):int(n * 0.8)]
    te = idx[int(n * 0.8):]

    X_tr, X_va, X_te = X_base[tr], X_base[va], X_base[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    c_tr = [case_ids_all[i] for i in tr]
    c_te = [case_ids_all[i] for i in te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    results = {}

    # 1. LR
    print("  [1/7] LR...", end=" ")
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_tr_sc, y_tr)
    p = lr.predict_proba(X_te_sc)[:, 1]
    results["LR"] = eval_model(y_te, p)
    print(f"AUC={results['LR']['AUC-ROC']:.4f}")

    # 2. SVM
    print("  [2/7] SVM...", end=" ")
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    svm.fit(X_tr_sc, y_tr)
    p = svm.predict_proba(X_te_sc)[:, 1]
    results["SVM"] = eval_model(y_te, p)
    print(f"AUC={results['SVM']['AUC-ROC']:.4f}")

    # 3. MLP
    print("  [3/7] MLP...", end=" ")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=300, random_state=seed, early_stopping=True)
    mlp.fit(X_tr_sc, y_tr)
    p = mlp.predict_proba(X_te_sc)[:, 1]
    results["MLP"] = eval_model(y_te, p)
    print(f"AUC={results['MLP']['AUC-ROC']:.4f}")

    # 4. RF
    print("  [4/7] RF...", end=" ")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:, 1]
    results["RF"] = eval_model(y_te, p)
    print(f"AUC={results['RF']['AUC-ROC']:.4f}")

    # 5. XGBoost (GradientBoosting)
    print("  [5/7] XGBoost...", end=" ")
    xgb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
    xgb.fit(X_tr, y_tr)
    p = xgb.predict_proba(X_te)[:, 1]
    results["XGBoost"] = eval_model(y_te, p)
    print(f"AUC={results['XGBoost']['AUC-ROC']:.4f}")

    # 6. TEGAT-Red (独立 GNN 嵌入 + MLP, 无时间衰减)
    print("  [6/7] TEGAT-Red (graph emb, no temporal)...", end=" ")
    embed_red = GraphEmbedder(d=64, H=4, seed=seed)
    embed_red.fit(X_tr, c_tr, adj, edge_weights)
    X_tegat_red_tr = embed_red.transform(X_tr, c_tr, adj, edge_weights, temporal=False)
    X_tegat_red_te = embed_red.transform(X_te, c_te, adj, edge_weights, temporal=False)
    scaler2 = StandardScaler()
    X_tegat_red_tr_sc = scaler2.fit_transform(X_tegat_red_tr)
    X_tegat_red_te_sc = scaler2.transform(X_tegat_red_te)

    mlp_red = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed, early_stopping=True)
    mlp_red.fit(X_tegat_red_tr_sc, y_tr)
    p = mlp_red.predict_proba(X_tegat_red_te_sc)[:, 1]
    results["TEGAT-Red"] = eval_model(y_te, p)
    print(f"AUC={results['TEGAT-Red']['AUC-ROC']:.4f}")

    # 7. TEGAT (graph emb + MLP, τ=7)
    print("  [7/7] TEGAT-Full (graph emb + temporal)...", end=" ")
    embed_full = GraphEmbedder(d=64, H=4, seed=seed)
    embed_full.fit(X_tr, c_tr, adj, edge_weights)
    X_tegat_tr = embed_full.transform(X_tr, c_tr, adj, edge_weights, temporal=True)
    X_tegat_te = embed_full.transform(X_te, c_te, adj, edge_weights, temporal=True)
    scaler3 = StandardScaler()
    X_tegat_tr_sc = scaler3.fit_transform(X_tegat_tr)
    X_tegat_te_sc = scaler3.transform(X_tegat_te)

    mlp_full = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed, early_stopping=True)
    mlp_full.fit(X_tegat_tr_sc, y_tr)
    p = mlp_full.predict_proba(X_tegat_te_sc)[:, 1]
    results["TEGAT"] = eval_model(y_te, p)
    print(f"AUC={results['TEGAT']['AUC-ROC']:.4f}")

    # 打印
    print(f"\n  {'Method':<14} {'AUC':>7} {'AP':>7} {'Acc':>7} {'F1':>7} {'Prec':>8} {'Recall':>7}")
    print(f"  {'-'*65}")
    for m, v in sorted(results.items(), key=lambda x: -x[1]["AUC-ROC"]):
        print(f"  {m:<14} {v['AUC-ROC']:>7.4f} {v['AP']:>7.4f} {v['Accuracy']:>7.4f} "
              f"{v['F1']:>7.4f} {v['Precision']:>8.4f} {v['Recall']:>7.4f}")

    all_results[seed] = results

# ============ 汇总 ============
print("\n" + "=" * 65)
print("SUMMARY (tau=7, 5 seeds)")
print("=" * 65)

models = list(list(all_results.values())[0].keys())
metrics = ["AUC-ROC", "AP", "Accuracy", "F1", "Precision", "Recall"]
summary = {}
for m in models:
    vals = {met: [all_results[s][m][met] for s in SEEDS] for met in metrics}
    summary[m] = {met: {"mean": round(np.mean(vals[met]), 4), "std": round(np.std(vals[met]), 4)} for met in metrics}
    print(f"\n{m}:")
    for met in metrics:
        mu, sd = summary[m][met]["mean"], summary[m][met]["std"]
        print(f"  {met:<12} = {mu:.4f} +/- {sd:.4f}")

out = {
    "all_results": all_results,
    "summary": summary,
    "config": {"tau": TAU, "edge_window": EDGE_WINDOW, "seeds": SEEDS, "n_cases": len(CASES), "positive_ratio": float(y.mean())},
}
with open(os.path.join(OUT_DIR, "main_results_v2.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\nSaved: {OUT_DIR}/main_results_v2.json")
print("=" * 65)
print("Main experiment COMPLETE")
print("=" * 65)
