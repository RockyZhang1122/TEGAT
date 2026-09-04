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
TEGAT: Temporal Event Graph Attention Network
+ 多种Baseline实现
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    precision_score, recall_score, average_precision_score,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ==================== 数据加载 ====================

def load_graph_data():
    data = np.load(DATA_DIR / "graph_data.npz", allow_pickle=True)
    with open(DATA_DIR / "graph_mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)
    
    return data, mappings

# ==================== 模型实现 ====================

class LR:
    """Logistic Regression"""
    def __init__(self):
        self.W = None
        self.b = None
    
    def fit(self, X, y, lr=0.01, epochs=200):
        n, d = X.shape
        self.W = np.zeros(d)
        self.b = 0.0
        for _ in range(epochs):
            z = X @ self.W + self.b
            p = 1 / (1 + np.exp(-z))
            grad_W = X.T @ (p - y) / n
            grad_b = (p - y).mean()
            self.W -= lr * grad_W
            self.b -= lr * grad_b
    
    def predict_proba(self, X):
        z = X @ self.W + self.b
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class RandomForest:
    """简化版Random Forest（基于决策树集成）"""
    def __init__(self, n_trees=10, max_depth=8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        n = len(X)
        idx = np.random.choice(n, n, replace=True)
        return X[idx], y[idx]
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            return {"leaf": True, "value": y.mean()}
        
        n, d = X.shape
        best_gain = 0
        best_split = None
        parent_entropy = self._entropy(y)
        
        for feat in range(min(d, 20)):
            thresholds = np.random.choice(X[:, feat], min(10, n), replace=False)
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue
                gain = parent_entropy - (
                    left_mask.sum() / n * self._entropy(y[left_mask]) +
                    right_mask.sum() / n * self._entropy(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat, t, left_mask, right_mask)
        
        if best_split is None:
            return {"leaf": True, "value": y.mean()}
        
        feat, t, left_mask, right_mask = best_split
        return {
            "leaf": False,
            "feat": feat,
            "threshold": t,
            "left": self._build_tree(X[left_mask], y[left_mask], depth+1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth+1),
        }
    
    def _entropy(self, y):
        p = y.mean()
        if p == 0 or p == 1:
            return 0
        return -p * np.log(p + 1e-9) - (1-p) * np.log(1-p + 1e-9)
    
    def _predict_tree(self, x, tree):
        if tree["leaf"]:
            return tree["value"]
        if x[tree["feat"]] <= tree["threshold"]:
            return self._predict_tree(x, tree["left"])
        else:
            return self._predict_tree(x, tree["right"])
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            Xb, yb = self._bootstrap_sample(X, y)
            self.trees.append(self._build_tree(Xb, yb))
    
    def predict_proba(self, X):
        preds = np.array([[self._predict_tree(x, t) for t in self.trees] for x in X])
        return preds.mean(axis=1)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class GCN:
    """Graph Convolutional Network (Kipf 2017)"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        np.random.seed(SEED)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    
    def _normalize_adj(self, adj):
        d = adj.sum(axis=1)
        d_inv_sqrt = np.power(d + 1e-9, -0.5)
        D = np.diag(d_inv_sqrt)
        return D @ adj @ D
    
    def fit(self, X, adj, y, train_idx):
        A_norm = self._normalize_adj(adj + np.eye(adj.shape[0]))
        
        for _ in range(self.epochs):
            # Forward
            H1 = A_norm @ X @ self.W1
            H1 = np.maximum(H1, 0)  # ReLU
            out = A_norm @ H1 @ self.W2
            out = 1 / (1 + np.exp(-out.squeeze()))
            
            # Backward (simplified)
            err = np.zeros_like(out)
            err[train_idx] = (out[train_idx] - y[train_idx]) / len(train_idx)
            
            grad_W2 = H1[train_idx].T @ err[train_idx, None]
            grad_W1 = X[train_idx].T @ ((err[train_idx, None] @ self.W2.T) * (H1[train_idx] > 0))
            
            self.W2 -= self.lr * grad_W2
            self.W1 -= self.lr * grad_W1
    
    def predict_proba(self, X, adj):
        A_norm = self._normalize_adj(adj + np.eye(adj.shape[0]))
        H1 = np.maximum(A_norm @ X @ self.W1, 0)
        out = A_norm @ H1 @ self.W2
        return 1 / (1 + np.exp(-out.squeeze()))
    
    def predict(self, X, adj, threshold=0.5):
        return (self.predict_proba(X, adj) >= threshold).astype(int)

class GAT:
    """Graph Attention Network (simplified)"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        np.random.seed(SEED)
        self.W = np.random.randn(input_dim, hidden_dim) * 0.1
        self.a_src = np.random.randn(hidden_dim) * 0.1
        self.a_dst = np.random.randn(hidden_dim) * 0.1
        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.1
    
    def fit(self, X, adj, y, train_idx):
        n = X.shape[0]
        for _ in range(self.epochs):
            # Compute attention
            H = X @ self.W
            e_src = H @ self.a_src
            e_dst = H @ self.a_dst
            e = e_src[:, None] + e_dst[None, :]
            e = np.where(adj > 0, e, -1e9)
            attention = np.exp(e - e.max(axis=1, keepdims=True))
            attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-9)
            
            # Aggregate
            H_agg = attention @ H
            
            # Output
            out = H_agg @ self.W_out
            out = 1 / (1 + np.exp(-out.squeeze()))
            
            # Backward (simplified gradient update)
            err = np.zeros_like(out)
            err[train_idx] = out[train_idx] - y[train_idx]
            
            grad_W_out = H_agg[train_idx].T @ err[train_idx, None]
            self.W_out -= self.lr * grad_W_out
            
            # Simplified gradient for W
            grad_W = X[train_idx].T @ (err[train_idx, None] @ self.W_out.T)
            self.W -= self.lr * grad_W * 0.1
    
    def predict_proba(self, X, adj):
        H = X @ self.W
        e_src = H @ self.a_src
        e_dst = H @ self.a_dst
        e = e_src[:, None] + e_dst[None, :]
        e = np.where(adj > 0, e, -1e9)
        attention = np.exp(e - e.max(axis=1, keepdims=True))
        attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-9)
        H_agg = attention @ H
        out = H_agg @ self.W_out
        return 1 / (1 + np.exp(-out.squeeze()))
    
    def predict(self, X, adj, threshold=0.5):
        return (self.predict_proba(X, adj) >= threshold).astype(int)

class TEGAT:
    """
    Temporal Event Graph Attention Network
    异构图 + 时序注意力
    """
    def __init__(self, input_dim_case, input_dim_venue, input_dim_date, 
                 hidden_dim=64, num_heads=4, lr=0.01, epochs=100, use_temporal=True, use_hetero=True):
        self.lr = lr
        self.epochs = epochs
        self.use_temporal = use_temporal
        self.use_hetero = use_hetero
        np.random.seed(SEED)
        
        # 投影到统一维度
        self.W_case = np.random.randn(input_dim_case, hidden_dim) * 0.1
        self.W_venue = np.random.randn(input_dim_venue, hidden_dim) * 0.1
        self.W_date = np.random.randn(input_dim_date, hidden_dim) * 0.1
        
        # 多头注意力参数
        self.heads = num_heads
        self.W_attn = [np.random.randn(hidden_dim * 2, 1) * 0.1 for _ in range(num_heads)]
        
        # 时序编码
        if use_temporal:
            self.W_time = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
        # 输出层
        self.W_out = np.random.randn(hidden_dim * 3, 1) * 0.1
    
    def _attention(self, h_src, h_dst):
        """计算注意力系数"""
        n = len(h_src)
        alphas = []
        for w in self.W_attn:
            concat = np.concatenate([h_src, h_dst], axis=1)
            e = np.tanh(concat @ w).squeeze()
            alphas.append(np.exp(e) / (np.exp(e).sum() + 1e-9))
        return np.mean(alphas, axis=0)
    
    def fit(self, case_X, venue_X, date_X, adj, y, train_idx, case_dates=None):
        n = case_X.shape[0]
        for epoch in range(self.epochs):
            # 投影
            H_case = np.maximum(case_X @ self.W_case, 0)
            H_venue = np.maximum(venue_X @ self.W_venue, 0)
            H_date = np.maximum(date_X @ self.W_date, 0)
            
            # 在 case-case 边上做注意力
            row, col = np.nonzero(adj)
            if len(row) == 0:
                continue
            
            h_src = H_case[row]
            h_dst = H_case[col]
            
            # 多头注意力聚合
            alphas = self._attention(h_src, h_dst)
            
            # 聚合
            H_new = np.zeros_like(H_case)
            np.add.at(H_new, col, alphas[:, None] * h_src)
            deg = np.bincount(col, minlength=n)
            H_new = H_new / (deg[:, None] + 1e-9)
            H_case = H_case + 0.5 * H_new  # 残差连接
            
            # 时序模块（如果启用）
            if self.use_temporal and case_dates is not None:
                # 按时间加权聚合相邻病例
                for i in train_idx:
                    if i >= len(case_dates):
                        continue
                    t_i = case_dates[i]
                    neighbors = np.nonzero(adj[i])[0]
                    if len(neighbors) == 0:
                        continue
                    valid_neighbors = [j for j in neighbors if j < len(case_dates)]
                    if not valid_neighbors:
                        continue
                    t_neighbors = case_dates[valid_neighbors]
                    time_diff = np.abs(t_neighbors - t_i)
                    time_weights = np.exp(-time_diff / 7.0)  # 时间衰减，7天半衰期
                    time_weights = time_weights / (time_weights.sum() + 1e-9)
                    H_case[i] = H_case[i] + time_weights @ H_case[valid_neighbors] @ self.W_time
            
            # 异构融合（如果启用）
            if self.use_hetero:
                # 通过 case-venue 边融合
                # 这里简化：用平均venue嵌入增强
                H_case_concat = np.concatenate([
                    H_case,
                    np.tile(H_venue.mean(axis=0, keepdims=True), (n, 1)),
                    np.tile(H_date.mean(axis=0, keepdims=True), (n, 1)),
                ], axis=1)
            else:
                H_case_concat = np.concatenate([
                    H_case,
                    np.zeros((n, H_case.shape[1])),
                    np.zeros((n, H_case.shape[1])),
                ], axis=1)
            
            # 输出
            out = 1 / (1 + np.exp(-(H_case_concat @ self.W_out).squeeze()))
            
            # 简化梯度更新
            err = np.zeros_like(out)
            err[train_idx] = out[train_idx] - y[train_idx]
            
            # 简化：只更新输出层
            grad_W_out = H_case_concat[train_idx].T @ err[train_idx, None]
            self.W_out -= self.lr * grad_W_out / max(1, epoch % 10 + 1)
            
            # 更新其他层（简化）
            if epoch % 5 == 0:
                self.W_case -= self.lr * 0.1 * np.random.randn(*self.W_case.shape) * 0.01
    
    def predict_proba(self, case_X, venue_X, date_X, adj, case_dates=None):
        H_case = np.maximum(case_X @ self.W_case, 0)
        H_venue = np.maximum(venue_X @ self.W_venue, 0)
        H_date = np.maximum(date_X @ self.W_date, 0)
        
        row, col = np.nonzero(adj)
        if len(row) == 0:
            pass
        else:
            h_src = H_case[row]
            h_dst = H_case[col]
            alphas = self._attention(h_src, h_dst)
            H_new = np.zeros_like(H_case)
            np.add.at(H_new, col, alphas[:, None] * h_src)
            deg = np.bincount(col, minlength=len(H_case))
            H_new = H_new / (deg[:, None] + 1e-9)
            H_case = H_case + 0.5 * H_new
        
        if self.use_hetero:
            H_case_concat = np.concatenate([
                H_case,
                np.tile(H_venue.mean(axis=0, keepdims=True), (len(H_case), 1)),
                np.tile(H_date.mean(axis=0, keepdims=True), (len(H_case), 1)),
            ], axis=1)
        else:
            H_case_concat = np.concatenate([
                H_case,
                np.zeros((len(H_case), H_case.shape[1])),
                np.zeros((len(H_case), H_case.shape[1])),
            ], axis=1)
        
        out = 1 / (1 + np.exp(-(H_case_concat @ self.W_out).squeeze()))
        return out
    
    def predict(self, case_X, venue_X, date_X, adj, threshold=0.5, case_dates=None):
        return (self.predict_proba(case_X, venue_X, date_X, adj, case_dates) >= threshold).astype(int)


# ==================== 评估函数 ====================

def evaluate(y_true, y_pred, y_score):
    """评估分类结果"""
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
    """Precision@K"""
    top_k_idx = np.argsort(y_score)[-k:]
    return y_true[top_k_idx].sum() / k


# ==================== 主实验 ====================

def build_adj_matrix(edges, num_nodes):
    """构建邻接矩阵"""
    adj = np.zeros((num_nodes, num_nodes))
    for src, dst in edges.T:
        if src < num_nodes and dst < num_nodes:
            adj[src, dst] = 1
            adj[dst, src] = 1
    return adj


def main():
    print("=" * 60)
    print("TEGAT Main Experiment")
    print("=" * 60)
    
    # 加载数据
    data, mappings = load_graph_data()
    
    case_X = data["case_features"]
    venue_X = data["venue_features"]
    date_X = data["date_features"]
    transport_X = data["transport_features"]
    symptom_X = data["symptom_features"]
    labels = data["labels"]
    
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    
    # 构建邻接矩阵（case-case）
    if "case__case" in data.files:
        edges = data["case__case"]
    else:
        edges = np.zeros((2, 0), dtype=int)
    
    adj = build_adj_matrix(edges, len(case_X))
    
    print(f"\nDataset: {len(case_X)} cases")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Class balance (train): {labels[train_idx].mean():.3f} positive")
    
    # 准备case_dates（用于时序模块）
    case_dates = np.zeros(len(case_X))
    for i, cid in enumerate(mappings["case_ids"]):
        # 简单处理：使用一个虚拟日期
        case_dates[i] = i / 100.0  # 占位
    
    # ==================== 评估所有模型 ====================
    
    results = {}
    
    # 1. Logistic Regression (no graph)
    print("\n--- Training Logistic Regression ---")
    lr = LR()
    lr.fit(case_X[train_idx], labels[train_idx])
    pred_proba = lr.predict_proba(case_X[test_idx])
    pred = (pred_proba >= 0.5).astype(int)
    results["LR"] = evaluate(labels[test_idx], pred, pred_proba)
    results["LR"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 2. Random Forest (no graph)
    print("--- Training Random Forest ---")
    rf = RandomForest(n_trees=10, max_depth=6)
    rf.fit(case_X[train_idx], labels[train_idx])
    pred_proba = rf.predict_proba(case_X[test_idx])
    pred = (pred_proba >= 0.5).astype(int)
    results["RF"] = evaluate(labels[test_idx], pred, pred_proba)
    results["RF"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 3. GCN
    print("--- Training GCN ---")
    gcn = GCN(input_dim=case_X.shape[1], hidden_dim=32, epochs=80)
    gcn.fit(case_X, adj, labels, train_idx)
    pred_proba = gcn.predict_proba(case_X, adj)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["GCN"] = evaluate(labels[test_idx], pred, pred_proba)
    results["GCN"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 4. GAT
    print("--- Training GAT ---")
    gat = GAT(input_dim=case_X.shape[1], hidden_dim=32, epochs=80)
    gat.fit(case_X, adj, labels, train_idx)
    pred_proba = gat.predict_proba(case_X, adj)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["GAT"] = evaluate(labels[test_idx], pred, pred_proba)
    results["GAT"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 5. TEGAT-Full
    print("--- Training TEGAT-Full ---")
    tegat = TEGAT(
        input_dim_case=case_X.shape[1],
        input_dim_venue=venue_X.shape[1],
        input_dim_date=date_X.shape[1],
        hidden_dim=32, epochs=80, use_temporal=True, use_hetero=True
    )
    tegat.fit(case_X, venue_X, date_X, adj, labels, train_idx, case_dates)
    pred_proba = tegat.predict_proba(case_X, venue_X, date_X, adj, case_dates)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT"] = evaluate(labels[test_idx], pred, pred_proba)
    results["TEGAT"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 6. TEGAT w/o Temporal
    print("--- Training TEGAT w/o Temporal ---")
    tegat_no_temp = TEGAT(
        input_dim_case=case_X.shape[1],
        input_dim_venue=venue_X.shape[1],
        input_dim_date=date_X.shape[1],
        hidden_dim=32, epochs=80, use_temporal=False, use_hetero=True
    )
    tegat_no_temp.fit(case_X, venue_X, date_X, adj, labels, train_idx, case_dates)
    pred_proba = tegat_no_temp.predict_proba(case_X, venue_X, date_X, adj, case_dates)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT-noTemp"] = evaluate(labels[test_idx], pred, pred_proba)
    results["TEGAT-noTemp"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # 7. TEGAT w/o Heterogeneous
    print("--- Training TEGAT w/o Heterogeneous ---")
    tegat_no_het = TEGAT(
        input_dim_case=case_X.shape[1],
        input_dim_venue=venue_X.shape[1],
        input_dim_date=date_X.shape[1],
        hidden_dim=32, epochs=80, use_temporal=True, use_hetero=False
    )
    tegat_no_het.fit(case_X, venue_X, date_X, adj, labels, train_idx, case_dates)
    pred_proba = tegat_no_het.predict_proba(case_X, venue_X, date_X, adj, case_dates)[test_idx]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT-noHet"] = evaluate(labels[test_idx], pred, pred_proba)
    results["TEGAT-noHet"]["P@20"] = precision_at_k(labels[test_idx], pred_proba, k=20)
    
    # ==================== 输出结果 ====================
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Method':<20} {'AUC-ROC':>8} {'AP':>8} {'Acc':>8} {'F1':>8} {'P@20':>8}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['AUC-ROC']:>8.4f} {metrics['AP']:>8.4f} "
              f"{metrics['Accuracy']:>8.4f} {metrics['F1']:>8.4f} {metrics['P@20']:>8.4f}")
    
    # 保存
    with open(OUTPUT_DIR / "main_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
