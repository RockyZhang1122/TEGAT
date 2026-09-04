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
    DATA_DIR, RAW_TEXT_DIR, RESULTS_DIR, REPO_ROOT,
    data_path, raw_text_path, result_path, paper_path, figure_path,
)

# Legacy aliases preserved for body text inside this module.
OUTPUT_DIR = RESULTS_DIR
BASE = REPO_ROOT

# -*- coding: utf-8 -*-
"""
事件抽取Pipeline：从流调文本中抽取结构化事件
基于规则与模式匹配（轻量级，可解释）
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# ==================== 配置 ====================
OUTPUT_DIR.mkdir(exist_ok=True)

LOCKDOWN_DATE = datetime(2020, 1, 23)

# ==================== Schema 定义 ====================

# 事件类型
EVENT_TYPES = {
    1: "出行",     # Travel
    2: "就餐",     # Dining
    3: "购物",     # Shopping
    4: "就诊",     # Hospitalization
    5: "居家",     # Home-Stay
    6: "聚会",     # Gathering
    7: "工作",     # Work
    8: "交通",     # Transportation
    9: "其他",     # Other
}

# 场所类型（基于统计的真实分布）
VENUE_TYPES = {
    "医院": ["医院", "诊所", "卫生院", "门诊", "发热门诊", "隔离", "定点", "住院", "ICU"],
    "餐厅": ["餐厅", "饭店", "餐馆", "酒店", "年夜饭", "聚餐", "包间", "火锅店", "酒楼", "食堂", "咖啡厅"],
    "超市": ["超市", "商场", "购物中心", "永辉", "沃尔玛", "华润", "盒马", "便利店", "商店"],
    "交通工具": ["高铁", "火车", "航班", "飞机", "自驾", "出租车", "滴滴", "地铁", "城轨",
               "大巴", "客车", "私家车", "顺风车", "动车", "高铁站", "机场"],
    "家庭": ["家中", "家里", "居家", "住所", "住处", "回家", "住所地"],
    "工作场所": ["公司", "单位", "上班", "工作", "工厂", "写字楼", "办公室", "车间"],
    "酒店": ["酒店", "宾馆", "民宿", "旅馆", "招待所"],
    "公共场所": ["广场", "公园", "景区", "景点", "超市", "农贸市场", "菜市场", "步行街"],
    "学校": ["学校", "幼儿园", "小学", "中学", "大学", "培训机构", "培训班"],
    "药店": ["药店", "药房", "买药"],
}

# 交通方式
TRANSPORT_TYPES = {
    "高铁": "train_hsr",
    "火车": "train",
    "动车": "train_hsr",
    "航班": "flight",
    "飞机": "flight",
    "城轨": "light_rail",
    "地铁": "subway",
    "公交": "bus",
    "大巴": "bus",
    "客车": "bus",
    "自驾": "car",
    "私家车": "car",
    "出租车": "taxi",
    "滴滴": "taxi",
    "顺风车": "car",
}

# 症状
SYMPTOMS = ["发热", "咳嗽", "乏力", "咽痛", "流涕", "肺炎", "腹泻", "气短", "胸闷"]

# ==================== 实体识别器 ====================

def extract_dates_from_text(text: str) -> List[Tuple[int, datetime]]:
    """提取日期及其位置"""
    matches = []
    # 匹配格式：1月22日、1月22日-26日、1月22日至26日
    pattern = r"(\d{1,2})月(\d{1,2})日(?:[至\-—~](\d{1,2})月?(\d{1,2})?日?)?"
    for m in re.finditer(pattern, text):
        try:
            month = int(m.group(1))
            day = int(m.group(2))
            end_month = int(m.group(3)) if m.group(3) else month
            end_day = int(m.group(4)) if m.group(4) else day
            
            start_date = datetime(2020, month, day)
            end_date = datetime(2020, end_month, end_day) if (m.group(3) or m.group(4)) else start_date
            
            matches.append((m.start(), start_date))
            matches.append((m.end(), end_date))
        except (ValueError, TypeError):
            pass
    
    # 完整日期格式
    full_pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日"
    for m in re.finditer(full_pattern, text):
        try:
            date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            matches.append((m.start(), date))
        except ValueError:
            pass
    
    return sorted(matches, key=lambda x: x[0])

def extract_venues_from_text(text: str) -> List[Tuple[int, str, str]]:
    """提取场所及其类型"""
    venues = []
    for venue_type, keywords in VENUE_TYPES.items():
        for keyword in keywords:
            for m in re.finditer(re.escape(keyword), text):
                venues.append((m.start(), m.end(), venue_type))
    
    venues.sort()
    return venues

def extract_transport(text: str) -> List[Tuple[int, str]]:
    """提取交通方式"""
    results = []
    for transport, normalized in TRANSPORT_TYPES.items():
        for m in re.finditer(re.escape(transport), text):
            results.append((m.start(), normalized))
    return sorted(results)

def extract_symptoms(text: str) -> List[Tuple[int, str]]:
    """提取症状"""
    results = []
    for symptom in SYMPTOMS:
        for m in re.finditer(re.escape(symptom), text):
            results.append((m.start(), symptom))
    return sorted(results)

# ==================== 事件抽取 ====================

def classify_event_type(text_segment: str) -> int:
    """根据文本片段分类事件类型"""
    # 优先级：就诊 > 交通 > 就餐 > 购物 > 聚会 > 工作 > 居家 > 出行 > 其他
    
    if any(kw in text_segment for kw in ["医院", "就诊", "发热门诊", "隔离治疗", "住院", "门诊"]):
        return 4  # 就诊
    if any(kw in text_segment for kw in ["高铁", "火车", "航班", "飞机", "自驾", "出租车", "滴滴", 
                                          "地铁", "城轨", "大巴", "客车"]):
        return 8  # 交通
    if any(kw in text_segment for kw in ["超市", "商场", "购物中心", "永辉", "沃尔玛", "盒马"]):
        return 3  # 购物
    if any(kw in text_segment for kw in ["聚餐", "年夜饭", "餐厅", "饭店", "餐馆", "酒店", "包间"]):
        return 2  # 就餐
    if any(kw in text_segment for kw in ["聚会", "朋友", "亲戚", "来访", "做客"]):
        return 6  # 聚会
    if any(kw in text_segment for kw in ["公司", "上班", "工作", "工厂", "办公室"]):
        return 7  # 工作
    if any(kw in text_segment for kw in ["家中", "家里", "居家", "住所", "住处"]):
        return 5  # 居家
    if any(kw in text_segment for kw in ["出发", "到达", "前往", "回到", "离沪", "离汉"]):
        return 1  # 出行
    
    return 9  # 其他

def extract_events_from_case(case_text: str, case_id: str, province: str) -> List[Dict]:
    """从单条病例文本中抽取事件"""
    events = []
    
    # 按句子分割（中文常用标点）
    sentence_pattern = r"[^。；;]+[。；;]"
    sentences = re.findall(sentence_pattern, case_text)
    
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 3:
            continue
        
        # 抽取日期
        date_matches = extract_dates_from_text(sent)
        # 抽取场所
        venue_matches = extract_venues_from_text(sent)
        # 抽取交通方式
        transport_matches = extract_transport(sent)
        # 抽取症状
        symptom_matches = extract_symptoms(sent)
        
        # 至少要有日期或场所才视为事件
        if not date_matches and not venue_matches and not transport_matches:
            continue
        
        # 分类事件
        event_type = classify_event_type(sent)
        
        # 提取事件时间
        event_date = date_matches[0][1] if date_matches else None
        
        # 提取场所
        venues = list(set([v[2] for v in venue_matches]))
        
        # 提取交通方式
        transports = list(set([t[1] for t in transport_matches]))
        
        # 提取症状
        symptoms = list(set([s[1] for s in symptom_matches]))
        
        events.append({
            "case_id": case_id,
            "province": province,
            "event_type": event_type,
            "event_type_name": EVENT_TYPES[event_type],
            "date": event_date.strftime("%Y-%m-%d") if event_date else None,
            "venues": venues,
            "transports": transports,
            "symptoms": symptoms,
            "sentence": sent[:200],
            "is_lockdown": event_date >= LOCKDOWN_DATE if event_date else None,
        })
    
    return events

def extract_case_metadata(case_text: str, case_id: str, province: str) -> Dict:
    """提取病例元数据"""
    # 年龄
    age = None
    age_match = re.search(r"(\d+)\s*岁", case_text[:200])
    if age_match:
        age = int(age_match.group(1))
    
    # 性别
    gender = None
    if "男" in case_text[:50]:
        gender = "M"
    elif "女" in case_text[:50]:
        gender = "F"
    
    # 武汉关联
    wuhan_related = int(any(kw in case_text for kw in ["武汉", "汉口", "武昌", "湖北"]))
    
    # 无症状
    is_asymptomatic = int(("无症状" in case_text) or ("一直无异常" in case_text))
    
    # 第一个日期和最后一个日期
    all_dates = extract_dates_from_text(case_text)
    first_date = all_dates[0][1] if all_dates else None
    last_date = all_dates[-1][1] if all_dates else None
    
    return {
        "case_id": case_id,
        "province": province,
        "age": age,
        "gender": gender,
        "wuhan_related": wuhan_related,
        "is_asymptomatic": is_asymptomatic,
        "first_date": first_date.strftime("%Y-%m-%d") if first_date else None,
        "last_date": last_date.strftime("%Y-%m-%d") if last_date else None,
    }

# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("Event Extraction Pipeline")
    print("=" * 60)
    
    all_events = []
    all_cases_meta = []
    
    txt_files = sorted(RAW_TEXT_DIR.glob("*.txt"))
    print(f"\nProcessing {len(txt_files)} province files...")

    for txt_file in txt_files:
        province = txt_file.stem.replace("确诊病例活动轨迹", "").replace("确诊病例行程轨迹", "")
        content = txt_file.read_text(encoding="utf-8")
        
        # 分割病例
        case_pattern = r"第(\d+)号确诊病例"
        case_matches = list(re.finditer(case_pattern, content))
        
        case_count = 0
        event_count = 0
        
        for i, m in enumerate(case_matches):
            start = m.start()
            end = case_matches[i+1].start() if i+1 < len(case_matches) else len(content)
            case_text = content[start:end]
            case_id = f"{province[:2]}_{m.group(1)}"
            
            # 提取元数据
            meta = extract_case_metadata(case_text, case_id, province)
            all_cases_meta.append(meta)
            case_count += 1
            
            # 提取事件
            events = extract_events_from_case(case_text, case_id, province)
            all_events.extend(events)
            event_count += len(events)
        
        print(f"  {province}: {case_count} cases, {event_count} events")
    
    print(f"\nTotal: {len(all_cases_meta)} cases, {len(all_events)} events")
    
    # ==================== 统计与质量检查 ====================
    
    # 事件类型分布
    event_type_counter = Counter([e["event_type_name"] for e in all_events])
    print("\nEvent Type Distribution:")
    for et, count in event_type_counter.most_common():
        print(f"  {et}: {count}")
    
    # 事件/病例比
    avg_events_per_case = len(all_events) / len(all_cases_meta)
    print(f"\nAvg events per case: {avg_events_per_case:.2f}")
    
    # 日期覆盖率
    dated_events = [e for e in all_events if e["date"] is not None]
    print(f"Dated events: {len(dated_events)} / {len(all_events)} ({len(dated_events)/len(all_events)*100:.1f}%)")
    
    # 干预前后分布
    pre = sum(1 for e in all_events if e["is_lockdown"] is False)
    post = sum(1 for e in all_events if e["is_lockdown"] is True)
    print(f"\nPre-lockdown events: {pre}")
    print(f"Post-lockdown events: {post}")
    
    # 无症状病例数
    asym_cases = [c for c in all_cases_meta if c["is_asymptomatic"]]
    print(f"\nAsymptomatic cases: {len(asym_cases)}")
    
    # ==================== 保存结果 ====================
    
    with open(OUTPUT_DIR / "events_extracted.json", "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / "cases_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_cases_meta, f, ensure_ascii=False, indent=2)
    
    # 详细统计
    stats = {
        "total_cases": len(all_cases_meta),
        "total_events": len(all_events),
        "avg_events_per_case": avg_events_per_case,
        "event_type_distribution": dict(event_type_counter),
        "dated_events": len(dated_events),
        "dated_ratio": len(dated_events)/len(all_events)*100,
        "pre_lockdown_events": pre,
        "post_lockdown_events": post,
        "asymptomatic_count": len(asym_cases),
    }
    
    with open(OUTPUT_DIR / "extraction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
