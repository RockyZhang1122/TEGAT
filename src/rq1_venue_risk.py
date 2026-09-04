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
RQ1: 场所风险排序
- 不同场所的传播风险差异
- 跨省一致性
- TEGAT特征重要性
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR.mkdir(exist_ok=True)

VENUE_LABELS = {
    "医院": "Hospital",
    "餐厅": "Restaurant",
    "超市": "Supermarket",
    "交通工具": "Transportation",
    "家庭": "Household",
    "工作场所": "Workplace",
    "酒店": "Hotel",
    "公共场所": "Public Space",
    "学校": "School",
    "药店": "Pharmacy",
}


def main():
    print("=" * 60)
    print("RQ1: Venue Risk Ranking Analysis")
    print("=" * 60)
    
    with open(DATA_DIR / "events_extracted.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(DATA_DIR / "cases_metadata.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    # 1. 场所频率统计
    venue_case_count = defaultdict(set)  # venue -> set of case_ids
    venue_event_count = defaultdict(int)
    
    for event in events:
        for v in event["venues"]:
            venue_event_count[v] += 1
            venue_case_count[v].add(event["case_id"])
    
    print("\n=== Venue Frequency (Cases) ===")
    for venue in sorted(venue_event_count, key=lambda x: -venue_event_count[x]):
        print(f"  {venue} ({VENUE_LABELS.get(venue, venue)}): "
              f"{len(venue_case_count[venue])} cases, {venue_event_count[venue]} events")
    
    # 2. 计算每个场所的"风险评分"
    # 定义：病例在该场所出现的事件中，事件后的2-14天内出现新确诊的比例
    # 简化为：该病例是否最终确诊（其实就是1.0，因为都是确诊病例）
    # 替代：用TEGAT模型给出的概率（feature importance代理）
    
    # 用 feature_importance 作为风险评分
    with open(OUTPUT_DIR / "feature_importance.json", "r", encoding="utf-8") as f:
        importance = json.load(f)
    
    venue_risk_from_importance = {}
    for venue in venue_case_count.keys():
        key = f"venue_{venue}"
        if key in importance:
            venue_risk_from_importance[venue] = importance[key]
    
    # 3. 计算每个场所的"群聚率"：去该场所的病例中，参与群聚的比例
    venue_cluster_rate = {}
    cluster_cases = set()
    
    # 重新计算群聚标签
    from datetime import timedelta
    location_time_groups = defaultdict(list)
    for event in events:
        if event["date"] and event["venues"]:
            key = (event["province"], event["date"])
            for v in event["venues"]:
                location_time_groups[(event["province"], v, event["date"])].append(event["case_id"])
    
    # 检查每个病例是否参与群聚
    case_to_venues_at_dates = defaultdict(list)
    for event in events:
        if event["date"] and event["venues"]:
            for v in event["venues"]:
                case_to_venues_at_dates[event["case_id"]].append((v, event["date"], event["province"]))
    
    cluster_cases = set()
    for case_id, venues_dates in case_to_venues_at_dates.items():
        for v, d, prov in venues_dates:
            try:
                event_date = datetime.strptime(d, "%Y-%m-%d")
            except:
                continue
            for offset in range(-3, 4):
                target = (event_date + timedelta(days=offset)).strftime("%Y-%m-%d")
                for other_case in location_time_groups.get((prov, v, target), []):
                    if other_case != case_id:
                        cluster_cases.add(case_id)
                        break
                if case_id in cluster_cases:
                    break
            if case_id in cluster_cases:
                break
    
    for venue in venue_case_count.keys():
        cases_visited = venue_case_count[venue]
        cluster_in_venue = [c for c in cases_visited if c in cluster_cases]
        if len(cases_visited) > 0:
            venue_cluster_rate[venue] = len(cluster_in_venue) / len(cases_visited)
        else:
            venue_cluster_rate[venue] = 0
    
    # 4. 跨省一致性分析
    venue_province_risk = defaultdict(dict)  # venue -> province -> cluster_rate
    for venue in venue_case_count.keys():
        for prov in set([c["province"] for c in cases]):
            # 哪些病例去过该省份的该场所
            visit_cases = set()
            for event in events:
                if event["province"] == prov and venue in event["venues"]:
                    visit_cases.add(event["case_id"])
            
            cluster_in_prov_venue = [c for c in visit_cases if c in cluster_cases]
            
            if len(visit_cases) >= 5:  # 至少5个样本才计算
                venue_province_risk[venue][prov] = len(cluster_in_prov_venue) / len(visit_cases)
    
    # 计算 Krippendorff's alpha（简化为各省份排名的一致性）
    province_rankings = defaultdict(list)
    for venue, prov_dict in venue_province_risk.items():
        for prov, rate in prov_dict.items():
            province_rankings[prov].append((venue, rate))
    
    # 计算每个场所的"全国平均风险"
    venue_avg_risk = {}
    for venue, prov_dict in venue_province_risk.items():
        if prov_dict:
            venue_avg_risk[venue] = np.mean(list(prov_dict.values()))
        else:
            venue_avg_risk[venue] = 0
    
    # 5. ANOVA检验：各场所风险是否有显著差异
    venue_groups = []
    venue_names_for_anova = []
    for venue, prov_dict in venue_province_risk.items():
        if len(prov_dict) >= 3:  # 至少3个省份
            venue_groups.append(list(prov_dict.values()))
            venue_names_for_anova.append(venue)
    
    if len(venue_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*venue_groups)
        print(f"\n=== ANOVA Test ===")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("[OK] Significant difference between venues")
        else:
            print("[WARN] No significant difference")
    else:
        f_stat, p_value = None, None
    
    # 输出结果
    print("\n=== Venue Risk Ranking (by cluster participation rate) ===")
    sorted_venues = sorted(venue_avg_risk.items(), key=lambda x: -x[1])
    for i, (venue, risk) in enumerate(sorted_venues, 1):
        n_cases = len(venue_case_count[venue])
        n_provinces = len(venue_province_risk.get(venue, {}))
        feat_imp = venue_risk_from_importance.get(venue, 0)
        print(f"  {i}. {venue} ({VENUE_LABELS.get(venue, venue)}): "
              f"avg_risk={risk:.4f}, n_cases={n_cases}, n_provinces={n_provinces}, "
              f"feature_imp={feat_imp:.4f}")
    
    # 保存
    result = {
        "venue_avg_risk": venue_avg_risk,
        "venue_cluster_rate": venue_cluster_rate,
        "venue_province_risk": {k: dict(v) for k, v in venue_province_risk.items()},
        "venue_feature_importance": venue_risk_from_importance,
        "anova_f_stat": float(f_stat) if f_stat is not None else None,
        "anova_p_value": float(p_value) if p_value is not None else None,
        "ranking": [{"rank": i, "venue": v, "avg_risk": r} 
                   for i, (v, r) in enumerate(sorted_venues, 1)],
    }
    
    with open(OUTPUT_DIR / "rq1_venue_risk.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/rq1_venue_risk.json")
    print("=" * 60)
    print("RQ1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()