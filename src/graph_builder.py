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
事件图构建：将抽取的事件构建为异构图
- 节点类型：case, venue, date, transport
- 边类型：case-venue, case-date, case-transport, case-case (同城/同时)
"""
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ==================== 配置 ====================
OUTPUT_DIR = DATA_DIR

LOCKDOWN_DATE = datetime(2020, 1, 23)

# ==================== 数据加载 ====================

def load_data():
    with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(DATA_DIR / "cases_metadata.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    return events, cases

# ==================== 异构图构建 ====================

def build_heterogeneous_graph(events, cases):
    """
    构建异构图：
    - case nodes (N_c)
    - venue nodes (N_v) - 10 types
    - date nodes (N_d) - by day
    - transport nodes (N_t) - 7 types
    - symptom nodes (N_s) - 6 types
    
    边：
    - case -> venue (去过该场所)
    - case -> date (在那天)
    - case -> transport (使用该交通)
    - case -> case (同场所同时段，传播嫌疑)
    - case -> symptom (出现该症状)
    """
    # 创建映射
    case_ids = [c["case_id"] for c in cases]
    case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)}
    
    venues = ["医院", "餐厅", "超市", "交通工具", "家庭", "工作场所", "酒店", "公共场所", "学校", "药店"]
    venue_to_idx = {v: i for i, v in enumerate(venues)}
    
    # 日期范围
    all_dates = sorted(set([e["date"] for e in events if e["date"]]))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    transports = list(set([t for e in events for t in e["transports"]]))
    transport_to_idx = {t: i for i, t in enumerate(transports)}
    
    symptoms_list = ["发热", "咳嗽", "乏力", "咽痛", "流涕", "肺炎"]
    symptom_to_idx = {s: i for i, s in enumerate(symptoms_list)}
    
    # 构建边
    edges = defaultdict(list)  # (src_type, dst_type) -> [(src, dst, weight)]
    
    for event in events:
        case_idx = case_id_to_idx.get(event["case_id"])
        if case_idx is None:
            continue
        
        # case -> venue
        for v in event["venues"]:
            if v in venue_to_idx:
                edges[("case", "venue")].append((case_idx, venue_to_idx[v], 1))
        
        # case -> date
        if event["date"] in date_to_idx:
            edges[("case", "date")].append((case_idx, date_to_idx[event["date"]], 1))
        
        # case -> transport
        for t in event["transports"]:
            if t in transport_to_idx:
                edges[("case", "transport")].append((case_idx, transport_to_idx[t], 1))
        
        # case -> symptom
        for s in event["symptoms"]:
            if s in symptom_to_idx:
                edges[("case", "symptom")].append((case_idx, symptom_to_idx[s], 1))
    
    # case-case 边：同城同场所同时段（潜在传播嫌疑）
    # 按(province, date, venue)分组
    location_time_groups = defaultdict(list)
    for event in events:
        if event["date"] and event["venues"]:
            case_idx = case_id_to_idx.get(event["case_id"])
            if case_idx is None:
                continue
            for v in event["venues"]:
                key = (event["province"], event["date"], v)
                location_time_groups[key].append(case_idx)
    
    case_case_edges = []
    for key, case_list in location_time_groups.items():
        case_list = list(set(case_list))
        if len(case_list) > 1 and len(case_list) < 50:  # 避免过大完全图
            for i in range(len(case_list)):
                for j in range(i+1, len(case_list)):
                    case_case_edges.append((case_list[i], case_list[j], 1))
    
    edges[("case", "case")] = case_case_edges
    
    # 统计
    graph_stats = {
        "num_cases": len(case_ids),
        "num_venues": len(venues),
        "num_dates": len(date_to_idx),
        "num_transports": len(transport_to_idx),
        "num_symptoms": len(symptom_to_idx),
        "edges_case_venue": len(edges[("case", "venue")]),
        "edges_case_date": len(edges[("case", "date")]),
        "edges_case_transport": len(edges[("case", "transport")]),
        "edges_case_case": len(edges[("case", "case")]),
        "edges_case_symptom": len(edges[("case", "symptom")]),
    }
    
    return edges, {
        "case_to_idx": case_id_to_idx,
        "venue_to_idx": venue_to_idx,
        "date_to_idx": date_to_idx,
        "transport_to_idx": transport_to_idx,
        "symptom_to_idx": symptom_to_idx,
        "case_ids": case_ids,
        "venues": venues,
        "all_dates": all_dates,
        "transports": transports,
        "symptoms": symptoms_list,
    }, graph_stats

# ==================== 节点特征 ====================

def build_node_features(events, cases, mappings):
    """构建节点特征"""
    num_cases = len(mappings["case_ids"])
    num_venues = len(mappings["venues"])
    num_dates = len(mappings["all_dates"])
    num_transports = len(mappings["transports"])
    num_symptoms = len(mappings["symptoms"])
    
    # case features: [age_norm, gender_M, gender_F, wuhan_related, asymptomatic, first_day_norm]
    case_features = np.zeros((num_cases, 8), dtype=np.float32)
    
    ages = [c["age"] for c in cases if c["age"] is not None]
    if ages:
        age_min, age_max = min(ages), max(ages)
    else:
        age_min, age_max = 0, 100
    
    for i, case in enumerate(cases):
        if case["age"]:
            case_features[i, 0] = (case["age"] - age_min) / (age_max - age_min + 1e-6)
        if case["gender"] == "M":
            case_features[i, 1] = 1
        elif case["gender"] == "F":
            case_features[i, 2] = 1
        if case["wuhan_related"]:
            case_features[i, 3] = 1
        if case["is_asymptomatic"]:
            case_features[i, 4] = 1
        # 病例首发日期（相对于封城日）
        if case["first_date"]:
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                case_features[i, 5] = (fd - LOCKDOWN_DATE).days / 100.0  # 归一化
            except:
                pass
        # 病例活动时长
        if case["first_date"] and case["last_date"]:
            try:
                fd = datetime.strptime(case["first_date"], "%Y-%m-%d")
                ld = datetime.strptime(case["last_date"], "%Y-%m-%d")
                case_features[i, 6] = (ld - fd).days / 30.0
            except:
                pass
        # event count
        case_events = [e for e in events if e["case_id"] == case["case_id"]]
        case_features[i, 7] = min(len(case_events), 20) / 20.0
    
    # venue features: [type_embedding (one-hot of 10)]
    venue_features = np.eye(num_venues, dtype=np.float32)
    
    # date features: [day_of_lockdown_norm, day_of_week_sin, day_of_week_cos]
    date_features = np.zeros((num_dates, 4), dtype=np.float32)
    for i, d in enumerate(mappings["all_dates"]):
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            days_since = (dt - LOCKDOWN_DATE).days / 100.0
            date_features[i, 0] = days_since
            dow = dt.weekday()
            date_features[i, 1] = np.sin(2 * np.pi * dow / 7)
            date_features[i, 2] = np.cos(2 * np.pi * dow / 7)
            date_features[i, 3] = 1.0  # 通常属性
        except:
            pass
    
    # transport features: one-hot
    transport_features = np.eye(num_transports, dtype=np.float32)
    
    # symptom features: one-hot
    symptom_features = np.eye(num_symptoms, dtype=np.float32)
    
    features = {
        "case": case_features,
        "venue": venue_features,
        "date": date_features,
        "transport": transport_features,
        "symptom": symptom_features,
    }
    
    return features

# ==================== 风险标签 ====================

def build_risk_labels(events, cases, mappings):
    """
    构建风险预测标签
    风险事件定义：
    - 病例与≥1其他病例在同场所同时段出现 → label=1
    - 其他 → label=0
    """
    case_idx_to_id = {i: cid for i, cid in enumerate(mappings["case_ids"])}
    
    # 高风险病例：与≥1其他病例同场所同时段
    high_risk_cases = set()
    for event in events:
        if event["date"] and event["venues"]:
            # 查找同城同场所同时段的其他病例
            same_location_time = [
                e2 for e2 in events 
                if e2["date"] == event["date"] 
                and e2["province"] == event["province"]
                and set(e2["venues"]) & set(event["venues"])
                and e2["case_id"] != event["case_id"]
            ]
            if same_location_time:
                high_risk_cases.add(event["case_id"])
    
    labels = np.zeros(len(cases), dtype=np.int64)
    for i, case in enumerate(cases):
        if case["case_id"] in high_risk_cases:
            labels[i] = 1
    
    return labels

# ==================== 数据集划分 ====================

def split_dataset(cases, labels, train_ratio=0.7, val_ratio=0.1, seed=42):
    """随机划分数据集"""
    np.random.seed(seed)
    n = len(cases)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx

# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("Event Graph Construction")
    print("=" * 60)
    
    events, cases = load_data()
    print(f"\nLoaded {len(cases)} cases, {len(events)} events")
    
    # 构建异构图
    edges, mappings, graph_stats = build_heterogeneous_graph(events, cases)
    print("\nGraph Statistics:")
    for k, v in graph_stats.items():
        print(f"  {k}: {v}")
    
    # 构建节点特征
    features = build_node_features(events, cases, mappings)
    print("\nFeature Shapes:")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")
    
    # 构建标签
    labels = build_risk_labels(events, cases, mappings)
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    print(f"\nRisk labels:")
    print(f"  Positive (high risk): {pos_count} ({pos_count/len(labels)*100:.1f}%)")
    print(f"  Negative (low risk): {neg_count} ({neg_count/len(labels)*100:.1f}%)")
    
    # 数据集划分
    train_idx, val_idx, test_idx = split_dataset(cases, labels)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(cases)*100:.1f}%)")
    print(f"  Val: {len(val_idx)} ({len(val_idx)/len(cases)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} ({len(test_idx)/len(cases)*100:.1f}%)")
    
    # 保存
    print("\nSaving graph data...")
    
    # 保存边索引（按类型）
    edge_index_dict = {}
    for edge_type, edge_list in edges.items():
        if edge_list:
            edge_index_dict[f"{edge_type[0]}__{edge_type[1]}"] = np.array(edge_list)[:, :2].T  # (2, E)
    
    np.savez(
        OUTPUT_DIR / "graph_data.npz",
        case_features=features["case"],
        venue_features=features["venue"],
        date_features=features["date"],
        transport_features=features["transport"],
        symptom_features=features["symptom"],
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        **{k: v for k, v in edge_index_dict.items()},
    )
    
    # 保存映射
    with open(OUTPUT_DIR / "graph_mappings.json", "w", encoding="utf-8") as f:
        json.dump({
            "case_ids": mappings["case_ids"],
            "venues": mappings["venues"],
            "all_dates": mappings["all_dates"],
            "transports": mappings["transports"],
            "symptoms": mappings["symptoms"],
        }, f, ensure_ascii=False, indent=2)
    
    # 保存统计
    with open(OUTPUT_DIR / "graph_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            **graph_stats,
            "feature_shapes": {k: list(v.shape) for k, v in features.items()},
            "label_distribution": {
                "positive": int(pos_count),
                "negative": int(neg_count),
                "positive_ratio": float(pos_count/len(labels)),
            },
            "data_split": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)
    print("Graph Construction Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
