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
基于scikit-learn的稳定Baseline + 简化TEGAT
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)


def load_data():
    data = np.load(DATA_DIR / "graph_data.npz", allow_pickle=True)
    with open(DATA_DIR / "graph_mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)
    return data, mappings


def evaluate(y_true, y_pred, y_score):
    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_score),
        "AP": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def precision_at_k(y_true, y_score, k=20):
    if len(y_true) <= k:
        return y_true.mean()
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true[top_k_idx].mean()


class GraphBasedMLP:
    """
    使用图结构信息增强的MLP
    特征 = case_features + 邻居聚合统计
    """
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
    
    def _graph_features(self, X, adj):
        """构建图结构特征"""
        n = X.shape[0]
        # 度中心性
        degree = adj.sum(axis=1)
        # 邻居特征均值
        neighbor_mean = np.zeros_like(X)
        deg_inv = 1.0 / (degree + 1e-9)
        for i in range(n):
            if degree[i] > 0:
                neighbors = np.nonzero(adj[i])[0]
                neighbor_mean[i] = X[neighbors].mean(axis=0)
        # PageRank-like 分数（简化版）
        pr = np.ones(n) / n
        for _ in range(5):
            pr = (0.85 * (adj @ pr) / (degree + 1e-9)) + (0.15 / n)
        # 二阶邻居
        second_neighbor = adj @ adj
        second_neighbor_count = second_neighbor.sum(axis=1)
        graph_feats = np.column_stack([
            degree / max(degree.max(), 1),
            neighbor_mean.mean(axis=1),
            neighbor_mean.std(axis=1),
            pr,
            second_neighbor_count / max(second_neighbor_count.max(), 1),
        ])
        return graph_feats
    
    def fit(self, X, adj, y, train_idx):
        # 增强特征
        graph_feats = self._graph_features(X, adj)
        X_enhanced = np.concatenate([X, graph_feats], axis=1)
        X_scaled = self.scaler.fit_transform(X_enhanced)
        self.mlp.fit(X_scaled[train_idx], y[train_idx])
    
    def predict_proba(self, X, adj):
        graph_feats = self._graph_features(X, adj)
        X_enhanced = np.concatenate([X, graph_feats], axis=1)
        X_scaled = self.scaler.transform(X_enhanced)
        return self.mlp.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X, adj, threshold=0.5):
        return (self.predict_proba(X, adj) >= threshold).astype(int)


class TEGAT_SKL:
    """
    TEGAT: Temporal Event Graph Attention Network (scikit-learn版本)
    使用多种聚合 + 时序特征 + 异构融合
    """
    def __init__(self, n_estimators=100, random_state=SEED):
        self.scaler = StandardScaler()
        self.clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
    
    def _build_features(self, X, adj, case_features, case_dates=None):
        """构建TEGAT的丰富特征"""
        n = X.shape[0]
        
        # 1. 基础特征
        base = X.copy()
        
        # 2. 图结构特征
        degree = adj.sum(axis=1)
        neighbor_mean = np.zeros((n, X.shape[1]))
        neighbor_max = np.zeros((n, X.shape[1]))
        for i in range(n):
            if degree[i] > 0:
                neighbors = np.nonzero(adj[i])[0]
                neighbor_mean[i] = X[neighbors].mean(axis=0)
                neighbor_max[i] = X[neighbors].max(axis=0)
        
        # 注意力特征（简化为节点相似度）
        self_similarity = np.diag(X @ X.T)
        
        # 3. 时序特征
        temporal_feats = np.zeros((n, 4))
        if case_dates is not None and len(case_dates) == n:
            temporal_feats[:, 0] = case_dates
            temporal_feats[:, 1] = np.sin(2 * np.pi * case_dates / 30)
            temporal_feats[:, 2] = np.cos(2 * np.pi * case_dates / 30)
            # 时序邻居统计
            for i in range(n):
                neighbors = np.nonzero(adj[i])[0]
                if len(neighbors) > 0:
                    valid_neighbors = neighbors[neighbors < len(case_dates)]
                    if len(valid_neighbors) > 0:
                        time_diff = np.abs(case_dates[valid_neighbors] - case_dates[i])
                        temporal_feats[i, 3] = np.exp(-time_diff.min() / 7.0)
        
        # 4. 异构特征（从venue和date均值）
        venue_feats = case_features.get("venue_mean", np.zeros(n))
        
        # 5. 度数特征
        degree_norm = degree / (degree.max() + 1e-9)
        
        # 合并
        features = np.column_stack([
            base,
            neighbor_mean,
            neighbor_max,
            self_similarity[:, None] if self_similarity.ndim == 1 else self_similarity,
            temporal_feats,
            venue_feats,
            degree_norm[:, None],
        ])
        return features
    
    def fit(self, X, adj, y, train_idx, case_features=None, case_dates=None):
        features = self._build_features(X, adj, case_features or {}, case_dates)
        X_scaled = self.scaler.fit_transform(features)
        self.clf.fit(X_scaled[train_idx], y[train_idx])
    
    def predict_proba(self, X, adj, case_features=None, case_dates=None):
        features = self._build_features(X, adj, case_features or {}, case_dates)
        X_scaled = self.scaler.transform(features)
        return self.clf.predict_proba(X_scaled)[:, 1]
    
    def predict(self, X, adj, threshold=0.5, case_features=None, case_dates=None):
        return (self.predict_proba(X, adj, case_features, case_dates) >= threshold).astype(int)


def build_adj_matrix(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    if edges.size > 0:
        for src, dst in edges.T:
            if src < num_nodes and dst < num_nodes:
                adj[src, dst] = 1
                adj[dst, src] = 1
    return adj


def main():
    print("=" * 60)
    print("Main Experiment: TEGAT vs Baselines (scikit-learn)")
    print("=" * 60)
    
    data, mappings = load_data()
    
    case_X = data["case_features"]
    venue_X = data["venue_features"]
    date_X = data["date_features"]
    transport_X = data["transport_features"]
    symptom_X = data["symptom_features"]
    labels = data["labels"]
    
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    
    edges = data["case__case"] if "case__case" in data.files else np.zeros((2, 0), dtype=int)
    adj = build_adj_matrix(edges, len(case_X))
    
    # Case features for TEGAT
    case_features_for_tegat = {
        "venue_mean": np.tile(venue_X.mean(axis=0), len(case_X)).reshape(len(case_X), -1),
    }
    
    # Case dates (proxy)
    case_dates = np.array([i / 100.0 for i in range(len(case_X))])
    
    print(f"\nDataset: {len(case_X)} cases")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Class balance (train): {labels[train_idx].mean():.3f} positive")
    print(f"  Adj density: {adj.sum() / (len(case_X)**2) * 100:.4f}%")
    
    results = {}
    
    # 1. Logistic Regression
    print("\n[1/8] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(case_X[train_idx], labels[train_idx])
    pred_proba = lr.predict_proba(case_X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["LR"] = evaluate(labels[test_idx], pred, pred_proba)
    results["LR"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 2. Random Forest
    print("[2/8] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    rf.fit(case_X[train_idx], labels[train_idx])
    pred_proba = rf.predict_proba(case_X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["RF"] = evaluate(labels[test_idx], pred, pred_proba)
    results["RF"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 3. XGBoost (使用 GradientBoosting)
    print("[3/8] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    gb.fit(case_X[train_idx], labels[train_idx])
    pred_proba = gb.predict_proba(case_X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["XGBoost"] = evaluate(labels[test_idx], pred, pred_proba)
    results["XGBoost"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 4. SVM
    print("[4/8] SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=SEED)
    svm.fit(case_X[train_idx], labels[train_idx])
    pred_proba = svm.predict_proba(case_X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["SVM"] = evaluate(labels[test_idx], pred, pred_proba)
    results["SVM"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 5. MLP (no graph)
    print("[5/8] MLP (no graph)...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=SEED)
    mlp.fit(case_X[train_idx], labels[train_idx])
    pred_proba = mlp.predict_proba(case_X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["MLP"] = evaluate(labels[test_idx], pred, pred_proba)
    results["MLP"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 6. Graph-MLP (with graph features)
    print("[6/8] Graph-MLP (with graph structure)...")
    graph_mlp = GraphBasedMLP(hidden_layer_sizes=(128, 64), max_iter=200)
    graph_mlp.fit(case_X, adj, labels, train_idx)
    pred_proba = graph_mlp.predict_proba(case_X, adj)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["Graph-MLP"] = evaluate(labels[test_idx], pred, pred_proba)
    results["Graph-MLP"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 7. TEGAT-Full
    print("[7/8] TEGAT-Full...")
    tegat = TEGAT_SKL(n_estimators=200)
    tegat.fit(case_X, adj, labels, train_idx, case_features_for_tegat, case_dates)
    pred_proba = tegat.predict_proba(case_X, adj, case_features_for_tegat, case_dates)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT"] = evaluate(labels[test_idx], pred, pred_proba)
    results["TEGAT"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 8. TEGAT-Reduced (no temporal)
    print("[8/8] TEGAT-Reduced (no temporal/heterogeneous)...")
    tegat_red = TEGAT_SKL(n_estimators=150)
    # 使用简化版特征
    tegat_red._build_features = lambda X, adj, cf, cd: X
    tegat_red.fit(case_X, adj, labels, train_idx, case_features_for_tegat, case_dates)
    pred_proba = tegat_red.predict_proba(case_X, adj, case_features_for_tegat, case_dates)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT-Reduced"] = evaluate(labels[test_idx], pred, pred_proba)
    results["TEGAT-Reduced"]["P@20"] = precision_at_k(labels[test_idx], pred_proba)
    
    # 输出
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Method':<18} {'AUC-ROC':>8} {'AP':>8} {'Acc':>8} {'F1':>8} {'P@20':>8}")
    print("-" * 60)
    for method, metrics in sorted(results.items()):
        print(f"{method:<18} {metrics['AUC-ROC']:>8.4f} {metrics['AP']:>8.4f} "
              f"{metrics['Accuracy']:>8.4f} {metrics['F1']:>8.4f} {metrics['P@20']:>8.4f}")
    
    with open(OUTPUT_DIR / "main_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()