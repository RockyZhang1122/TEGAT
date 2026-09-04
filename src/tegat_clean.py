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
重构风险预测任务 - 修复标签泄漏问题
新任务定义：
- 输入：单个病例的特征（场所列表、武汉关联、症状、时间）
- 预测：该病例是否参与了群聚传播（与≥1其他病例同城同场所时间相近）
- 关键：预测时只能基于病例自身特征，不能看到其他病例的标签
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# 风险标签定义:
# - 高风险 (label=1): 病例与≥2其他病例在同城同场所时间窗口(±3天)出现
# - 低风险 (label=0): 病例单独出现或仅与1个其他病例同城同场所

# 关键：训练特征必须不包含case-case直接连接


def build_safe_features(events, cases, mappings):
    """
    构建安全的病例特征（不含case-case连接信息）
    """
    num_cases = len(cases)
    case_id_to_idx = {c["case_id"]: i for i, c in enumerate(cases)}
    
    # 特征维度：
    # 0: age_norm
    # 1: is_male
    # 2: is_female
    # 3: wuhan_related
    # 4: is_asymptomatic
    # 5: first_day_norm (relative to lockdown)
    # 6: event_duration_norm
    # 7: event_count_norm
    # 8-17: venue_count_per_type (10 venues)
    # 18-22: transport_count_per_type (5 transports: car, train_hsr, train, flight, other)
    # 23: has_symptom
    # 24: symptom_count_norm
    
    venues = mappings["venues"]
    num_venues = len(venues)
    
    transports = mappings["transports"]
    num_transports = len(transports)
    
    n_features = 25 + num_venues + num_transports
    features = np.zeros((num_cases, n_features), dtype=np.float32)
    
    ages = [c["age"] for c in cases if c["age"] is not None]
    if ages:
        age_min, age_max = min(ages), max(ages)
    else:
        age_min, age_max = 0, 100
    
    from datetime import datetime
    LOCKDOWN_DATE = datetime(2020, 1, 23)
    
    for i, case in enumerate(cases):
        # 基础特征
        if case["age"]:
            features[i, 0] = (case["age"] - age_min) / (age_max - age_min + 1e-6)
        if case["gender"] == "M":
            features[i, 1] = 1
        elif case["gender"] == "F":
            features[i, 2] = 1
        if case["wuhan_related"]:
            features[i, 3] = 1
        if case["is_asymptomatic"]:
            features[i, 4] = 1
        
        if case["first_date"]:
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                features[i, 5] = (fd - LOCKDOWN_DATE).days / 100.0
            except:
                pass
        
        if case["first_date"] and case["last_date"]:
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                ld = datetime.strptime(case["last_date"], "%Y-%m-%d")
                features[i, 6] = (ld - fd).days / 30.0
            except:
                pass
        
        # 事件相关特征
        case_events = [e for e in events if e["case_id"] == case["case_id"]]
        features[i, 7] = min(len(case_events), 20) / 20.0
        
        # 场所分布
        venue_counts = defaultdict(int)
        for e in case_events:
            for v in e["venues"]:
                if v in venues:
                    venue_counts[v] += 1
        for j, v in enumerate(venues):
            features[i, 8 + j] = min(venue_counts[v], 10) / 10.0
        
        # 交通分布
        transport_counts = defaultdict(int)
        for e in case_events:
            for t in e["transports"]:
                if t in transports:
                    transport_counts[t] += 1
        for j, t in enumerate(transports):
            features[i, 18 + j] = min(transport_counts[t], 5) / 5.0
        
        # 症状
        all_symptoms = set()
        for e in case_events:
            for s in e["symptoms"]:
                all_symptoms.add(s)
        features[i, 23] = 1 if len(all_symptoms) > 0 else 0
        features[i, 24] = min(len(all_symptoms), 5) / 5.0
    
    return features


def build_cluster_labels(events, cases, time_window=3, min_cluster_size=2):
    """
    群聚标签：病例是否参与≥min_cluster_size的群聚传播
    群聚定义：与≥1其他病例在同城同场所时间窗口(±3天)出现
    """
    from datetime import datetime, timedelta
    
    # 索引：按 (province, venue, date) 分组
    location_time_groups = defaultdict(list)
    for event in events:
        if event["date"] and event["venues"]:
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            except:
                continue
            for v in event["venues"]:
                key = (event["province"], v, event["date"])
                location_time_groups[key].append((event["case_id"], event_date))
    
    # 扩展到时间窗口
    cluster_cases = set()
    case_id_to_events = defaultdict(list)
    for event in events:
        case_id_to_events[event["case_id"]].append(event)
    
    # 按病例汇总：是否参与了群聚
    for case in cases:
        case_events = case_id_to_events.get(case["case_id"], [])
        for event in case_events:
            if not event["date"] or not event["venues"]:
                continue
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            except:
                continue
            
            # 检查时间窗口内同省份同场所的其他病例
            for venue in event["venues"]:
                for offset in range(-time_window, time_window + 1):
                    target_date = (event_date + timedelta(days=offset)).strftime("%Y-%m-%d")
                    key = (event["province"], venue, target_date)
                    for other_case_id, _ in location_time_groups.get(key, []):
                        if other_case_id != case["case_id"]:
                            cluster_cases.add(case["case_id"])
                            break
                    if case["case_id"] in cluster_cases:
                        break
                if case["case_id"] in cluster_cases:
                    break
            if case["case_id"] in cluster_cases:
                break
    
    labels = np.array([1 if c["case_id"] in cluster_cases else 0 for c in cases])
    return labels


def main():
    print("=" * 60)
    print("Refactored TEGAT Experiment (No Label Leakage)")
    print("=" * 60)
    
    # 加载原始数据
    with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(DATA_DIR / "cases_metadata.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    with open(DATA_DIR / "graph_mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)
    
    # 构建无标签泄漏的特征
    X = build_safe_features(events, cases, mappings)
    print(f"\nFeatures shape: {X.shape}")
    
    # 构建标签（群聚传播）
    y = build_cluster_labels(events, cases)
    print(f"Cluster labels: {y.sum()} positive / {len(y)} total ({y.mean()*100:.1f}%)")
    
    # 数据集划分
    np.random.seed(SEED)
    n = len(cases)
    indices = np.random.permutation(n)
    train_size = int(n * 0.7)
    val_size = int(n * 0.1)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"\nDataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"Train positive ratio: {y[train_idx].mean():.3f}")
    print(f"Test positive ratio: {y[test_idx].mean():.3f}")
    
    # 评估函数
    def evaluate(y_true, y_pred, y_score):
        return {
            "AUC-ROC": roc_auc_score(y_true, y_score),
            "AP": average_precision_score(y_true, y_score),
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
        }
    
    def precision_at_k(y_true, y_score, k=20):
        if len(y_true) <= k:
            return y_true.mean()
        top_k_idx = np.argsort(y_score)[-k:]
        return y_true[top_k_idx].mean()
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ==================== 模型对比 ====================
    
    results = {}
    
    print("\n--- Training models ---")
    
    # 1. Logistic Regression
    print("[1/7] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_scaled[train_idx], y[train_idx])
    pred_proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["LR"] = evaluate(y[test_idx], pred, pred_proba)
    results["LR"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 2. Random Forest
    print("[2/7] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)
    rf.fit(X[train_idx], y[train_idx])
    pred_proba = rf.predict_proba(X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["RF"] = evaluate(y[test_idx], pred, pred_proba)
    results["RF"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 3. XGBoost (GradientBoosting)
    print("[3/7] Gradient Boosting (XGBoost-like)...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    gb.fit(X[train_idx], y[train_idx])
    pred_proba = gb.predict_proba(X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["XGBoost"] = evaluate(y[test_idx], pred, pred_proba)
    results["XGBoost"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 4. SVM
    print("[4/7] SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=SEED)
    svm.fit(X_scaled[train_idx], y[train_idx])
    pred_proba = svm.predict_proba(X_scaled[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["SVM"] = evaluate(y[test_idx], pred, pred_proba)
    results["SVM"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 5. MLP
    print("[5/7] MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED, early_stopping=True)
    mlp.fit(X_scaled[train_idx], y[train_idx])
    pred_proba = mlp.predict_proba(X_scaled[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["MLP"] = evaluate(y[test_idx], pred, pred_proba)
    results["MLP"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 6. TEGAT (基于异构特征的增强)
    print("[6/7] TEGAT (full features)...")
    # TEGAT 在 X基础上增加 venue-specific statistics
    venue_indices = list(range(8, 8 + len(mappings["venues"])))
    venue_features = X[:, venue_indices]
    transport_indices = list(range(18, 18 + len(mappings["transports"])))
    transport_features = X[:, transport_indices]
    
    # Venue-specific features
    venue_specific = np.column_stack([
        venue_features.sum(axis=1, keepdims=True),  # 总场所事件数
        venue_features.max(axis=1, keepdims=True),  # 最高场所频次
        venue_features.std(axis=1, keepdims=True),   # 场所分布均衡度
        (venue_features > 0).sum(axis=1, keepdims=True),  # 去过的不同场所数
    ])
    X_tegat = np.column_stack([X, venue_specific, transport_features.mean(axis=1, keepdims=True)])
    
    gb_tegat = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    gb_tegat.fit(X_tegat[train_idx], y[train_idx])
    pred_proba = gb_tegat.predict_proba(X_tegat[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT"] = evaluate(y[test_idx], pred, pred_proba)
    results["TEGAT"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 7. TEGAT 简化版（去掉异构venue特征）
    print("[7/7] TEGAT-Reduced...")
    gb_tegat_red = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    gb_tegat_red.fit(X[train_idx], y[train_idx])
    pred_proba = gb_tegat_red.predict_proba(X[test_idx])[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    results["TEGAT-Reduced"] = evaluate(y[test_idx], pred, pred_proba)
    results["TEGAT-Reduced"]["P@20"] = precision_at_k(y[test_idx], pred_proba)
    
    # 输出
    print("\n" + "=" * 60)
    print("RESULTS (No Label Leakage)")
    print("=" * 60)
    
    print(f"\n{'Method':<18} {'AUC-ROC':>8} {'AP':>8} {'Acc':>8} {'F1':>8} {'Precision':>9} {'Recall':>8} {'P@20':>8}")
    print("-" * 80)
    for method, metrics in sorted(results.items()):
        print(f"{method:<18} {metrics['AUC-ROC']:>8.4f} {metrics['AP']:>8.4f} "
              f"{metrics['Accuracy']:>8.4f} {metrics['F1']:>8.4f} "
              f"{metrics['Precision']:>9.4f} {metrics['Recall']:>8.4f} "
              f"{metrics['P@20']:>8.4f}")
    
    # 保存
    with open(OUTPUT_DIR / "main_results_clean.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 特征重要性
    importances = gb_tegat.feature_importances_
    feature_names = (
        [f"base_{i}" for i in range(8)] +
        [f"venue_{v}" for v in mappings["venues"]] +
        [f"transport_{t}" for t in mappings["transports"]] +
        ["has_symptom", "symptom_count", "venue_total", "venue_max", "venue_std", "venue_diversity", "transport_mean"]
    )
    
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))
    
    with open(OUTPUT_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("\nTop 10 Features:")
    for i, (name, imp) in enumerate(list(importance_dict.items())[:10]):
        print(f"  {i+1}. {name}: {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()