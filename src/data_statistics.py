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
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
CSKEN Dataset - 数据统计脚本
统计各省份病例数、时间分布、症状类型、场所类型等
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# ==================== 配置 ====================
OUTPUT_DIR.mkdir(exist_ok=True)

# 武汉封城时间节点
LOCKDOWN_DATE = datetime(2020, 1, 23)

# ==================== 辅助函数 ====================

def extract_province(filename):
    """从文件名提取省份名"""
    name = filename.replace("的确诊病例活动轨迹.txt", "").replace(".txt", "")
    return name

def parse_case_id(text):
    """提取病例编号"""
    patterns = [
        r"第(\d+)号确诊病例",
        r"第(\d+)号",
        r"病例(\d+)",
        r"确诊(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:100])
        if match:
            return match.group(1)
    return None

def extract_dates(text):
    """从文本中提取所有日期"""
    # 匹配格式：1月22日、1月22日-26日、1月22日、2020年1月22日
    patterns = [
        r"(\d{1,2})月(\d{1,2})日",
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
    ]
    dates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                if len(match.groups()) == 2:
                    month, day = int(match.group(1)), int(match.group(2))
                    year = 2020
                else:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                date = datetime(year, month, day)
                dates.append(date)
            except:
                pass
    return dates

def extract_asymptomatic(text):
    """检测是否为无症状感染者"""
    text_lower = text.lower()
    keywords = ["无症状", "一直无异常症状", "一直无异常", "一直没有症状", 
                "无发热", "无咳嗽", "无典型症状", "未出现发热", "未出现咳嗽",
                "咽拭子阳性", "核酸阳性", "没有症状"]
    # 检查是否提到"无症状"
    if "无症状" in text or "一直无异常" in text:
        return True
    return False

def extract_age_gender(text):
    """提取年龄和性别"""
    age, gender = None, None
    
    # 性别
    if "男" in text[:50]:
        gender = "男"
    elif "女" in text[:50]:
        gender = "女"
    
    # 年龄
    age_patterns = [
        r"(\d+)岁",
        r"年龄(\d+)",
        r"约(\d+)岁",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text[:200])
        if match:
            age = int(match.group(1))
            break
    
    return age, gender

def extract_venues(text):
    """提取场所类型"""
    venue_keywords = {
        "医院": ["医院", "诊所", "卫生院", "门诊", "发热门诊", "隔离", "定点"],
        "餐厅": ["餐厅", "饭店", "餐馆", "酒店", "年夜饭", "聚餐", "包间", "火锅店", "酒楼"],
        "超市": ["超市", "商场", "购物中心", "永辉", "沃尔玛", "华润", "盒马"],
        "交通工具": ["高铁", "火车", "航班", "飞机", "自驾", "出租车", "滴滴", "地铁", 
                    "城轨", "大巴", "客车", "私家车", "顺风车"],
        "家庭": ["家中", "家里", "居家", "住所", "住处", "回家"],
        "工作场所": ["公司", "单位", "上班", "工作", "工厂", "写字楼", "办公室"],
        "酒店": ["酒店", "宾馆", "民宿", "旅馆", "招待所"],
        "公共场所": ["广场", "公园", "景区", "景点", "超市", "农贸市场", "菜市场"],
        "学校": ["学校", "幼儿园", "小学", "中学", "大学", "培训机构"],
        "药店": ["药店", "药房", "买药"],
    }
    
    found_venues = []
    for venue_type, keywords in venue_keywords.items():
        for keyword in keywords:
            if keyword in text:
                found_venues.append(venue_type)
                break
    
    return list(set(found_venues))

def extract_symptoms(text):
    """提取症状"""
    symptom_keywords = {
        "发热": ["发热", "发烧", "体温"],
        "咳嗽": ["咳嗽"],
        "乏力": ["乏力", "疲劳", "疲倦", "无力"],
        "咽痛": ["咽痛", "喉咙痛", "嗓子疼"],
        "流涕": ["流涕", "鼻塞", "流鼻涕"],
        "肺炎": ["肺炎", "肺部"],
        "腹泻": ["腹泻", "拉肚子"],
    }
    
    found_symptoms = []
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if keyword in text:
                found_symptoms.append(symptom)
                break
    
    return list(set(found_symptoms))

def extract_wuhan_connection(text):
    """检测是否与武汉有关"""
    wuhan_keywords = ["武汉", "汉口", "武昌", "湖北"]
    return any(kw in text for kw in wuhan_keywords)

def extract_case_block(text):
    """将文本分割为单个病例块"""
    # 按"第X号确诊病例"分割
    pattern = r"第(\d+)号确诊病例"
    matches = list(re.finditer(pattern, text))
    
    cases = []
    for i, match in enumerate(matches):
        start = match.start()
        # 找到下一个病例的开始位置
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        
        case_text = text[start:end]
        case_id = match.group(1)
        dates = extract_dates(case_text)
        venues = extract_venues(case_text)
        symptoms = extract_symptoms(case_text)
        is_asymptomatic = extract_asymptomatic(case_text)
        wuhan_conn = extract_wuhan_connection(case_text)
        age, gender = extract_age_gender(case_text)
        
        cases.append({
            "case_id": case_id,
            "text": case_text[:500],  # 只保存前500字
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "first_date": min(dates).strftime("%Y-%m-%d") if dates else None,
            "last_date": max(dates).strftime("%Y-%m-%d") if dates else None,
            "venues": venues,
            "symptoms": symptoms,
            "is_asymptomatic": is_asymptomatic,
            "wuhan_connection": wuhan_conn,
            "age": age,
            "gender": gender,
        })
    
    return cases

# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("CSKEN 数据集统计开始")
    print("=" * 60)
    
    all_cases = []
    province_stats = []
    
    # 获取所有txt文件
    txt_files = list(RAW_TEXT_DIR.glob("*.txt"))
    print(f"\n发现 {len(txt_files)} 个省份文件")

    for txt_file in sorted(txt_files):
        province = extract_province(txt_file.name)
        content = txt_file.read_text(encoding="utf-8")
        
        cases = extract_case_block(content)
        all_cases.extend(cases)
        
        # 统计省份信息
        dates_all = []
        for case in cases:
            dates_all.extend(case["dates"])
        
        # 时间分布
        date_counts = Counter(dates_all)
        
        # 无症状统计
        asymptomatic_count = sum(1 for c in cases if c["is_asymptomatic"])
        
        # 武汉关联
        wuhan_count = sum(1 for c in cases if c["wuhan_connection"])
        
        # 场所类型统计
        all_venues = []
        for c in cases:
            all_venues.extend(c["venues"])
        venue_counts = Counter(all_venues)
        
        # 症状统计
        all_symptoms = []
        for c in cases:
            all_symptoms.extend(c["symptoms"])
        symptom_counts = Counter(all_symptoms)
        
        province_stats.append({
            "province": province,
            "case_count": len(cases),
            "date_count": len(dates_all),
            "asymptomatic_count": asymptomatic_count,
            "wuhan_count": wuhan_count,
            "first_case_date": min([d for d in dates_all]) if dates_all else None,
            "last_case_date": max([d for d in dates_all]) if dates_all else None,
            "top_venues": dict(venue_counts.most_common(5)),
            "top_symptoms": dict(symptom_counts.most_common(5)),
        })
        
        print(f"  {province}: {len(cases)} cases")
    
    # ==================== 全局统计 ====================
    print("\n" + "=" * 60)
    print("Global Statistics")
    print("=" * 60)
    
    total_cases = len(all_cases)
    print(f"\nTotal cases: {total_cases}")
    
    # Asymptomatic
    asym_cases = [c for c in all_cases if c["is_asymptomatic"]]
    print(f"Asymptomatic: {len(asym_cases)} ({len(asym_cases)/total_cases*100:.1f}%)")
    
    # Wuhan connection
    wuhan_cases = [c for c in all_cases if c["wuhan_connection"]]
    print(f"Wuhan-related: {len(wuhan_cases)} ({len(wuhan_cases)/total_cases*100:.1f}%)")
    
    # Time span
    all_dates = []
    for case in all_cases:
        all_dates.extend([datetime.strptime(d, "%Y-%m-%d") for d in case["dates"]])
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        print(f"\nTime span: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"Days span: {(max_date - min_date).days}")
        
        # Wuhan lockdown distribution
        pre_lockdown = len([d for d in all_dates if d < LOCKDOWN_DATE])
        post_lockdown = len([d for d in all_dates if d >= LOCKDOWN_DATE])
        print(f"\nPre-lockdown (<2020-01-23): {pre_lockdown} records ({pre_lockdown/len(all_dates)*100:.1f}%)")
        print(f"Post-lockdown (>=2020-01-23): {post_lockdown} records ({post_lockdown/len(all_dates)*100:.1f}%)")
        
        # Check coverage
        has_pre = pre_lockdown > 0
        has_post = post_lockdown > 0
        
        # Check if covers both pre and post lockdown
        if has_pre and has_post:
            first_date = min(all_dates)
            last_date = max(all_dates)
            days_before_lockdown = (LOCKDOWN_DATE - first_date).days
            days_after_lockdown = (last_date - LOCKDOWN_DATE).days
            
            print(f"\nDays before lockdown: {days_before_lockdown}")
            print(f"Days after lockdown: {days_after_lockdown}")
            
            if days_before_lockdown >= 7 and days_after_lockdown >= 14:
                print("[OK] Data covers pre/post lockdown, can do intervention analysis")
                lockdown_coverage = "SUFFICIENT"
            else:
                print("[WARN] Data coverage incomplete, intervention analysis limited")
                lockdown_coverage = "LIMITED"
        elif has_pre and not has_post:
            print("[ERROR] Only pre-lockdown data, cannot do intervention analysis")
            lockdown_coverage = "PRE_ONLY"
        elif not has_pre and has_post:
            print("[WARN] Only post-lockdown data, pre-lockdown data is limited")
            lockdown_coverage = "POST_ONLY"
        else:
            print("[ERROR] Date coverage unclear")
            lockdown_coverage = "UNKNOWN"
    else:
        print("[WARN] Cannot extract date information")
        lockdown_coverage = "UNKNOWN"
    
    # Venue statistics
    all_venues = []
    for case in all_cases:
        all_venues.extend(case["venues"])
    venue_counter = Counter(all_venues)
    print("\nVenue distribution:")
    for venue, count in venue_counter.most_common():
        print(f"  {venue}: {count}")
    
    # Symptom statistics
    all_symptoms = []
    for case in all_cases:
        all_symptoms.extend(case["symptoms"])
    symptom_counter = Counter(all_symptoms)
    print("\nSymptom distribution:")
    for symptom, count in symptom_counter.most_common():
        print(f"  {symptom}: {count}")
    
    # ==================== 保存结果 ====================
    
    # 1. 保存完整JSON
    stats = {
        "total_cases": total_cases,
        "province_count": len(txt_files),
        "asymptomatic_count": len(asym_cases),
        "asymptomatic_ratio": len(asym_cases)/total_cases*100 if total_cases > 0 else 0,
        "wuhan_connection_count": len(wuhan_cases),
        "wuhan_connection_ratio": len(wuhan_cases)/total_cases*100 if total_cases > 0 else 0,
        "date_range": {
            "min": min_date.strftime('%Y-%m-%d') if all_dates else None,
            "max": max_date.strftime('%Y-%m-%d') if all_dates else None,
            "days": (max_date - min_date).days if all_dates else 0,
        },
        "lockdown_coverage": lockdown_coverage,
        "pre_lockdown_records": len([d for d in all_dates if d < LOCKDOWN_DATE]) if all_dates else 0,
        "post_lockdown_records": len([d for d in all_dates if d >= LOCKDOWN_DATE]) if all_dates else 0,
        "venue_distribution": dict(venue_counter),
        "symptom_distribution": dict(symptom_counter),
    }
    
    with open(OUTPUT_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 2. 保存省份统计
    province_df = []
    for ps in province_stats:
        province_df.append({
            "province": ps["province"],
            "case_count": ps["case_count"],
            "asymptomatic": ps["asymptomatic_count"],
            "wuhan_related": ps["wuhan_count"],
            "first_date": ps["first_case_date"],
            "last_date": ps["last_case_date"],
            "top_venues": ps["top_venues"],
        })
    
    with open(OUTPUT_DIR / "province_stats.json", "w", encoding="utf-8") as f:
        json.dump(province_df, f, ensure_ascii=False, indent=2)
    
    # 3. 保存病例详情
    with open(OUTPUT_DIR / "cases_detail.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    # 4. Save daily statistics (for time series analysis)
    daily_counts = Counter()
    for d in all_dates:
        daily_counts[d.strftime('%Y-%m-%d')] += 1
    
    daily_data = []
    current = min_date
    while current <= max_date:
        date_str = current.strftime('%Y-%m-%d')
        count = daily_counts.get(date_str, 0)
        is_lockdown = "pre" if current < LOCKDOWN_DATE else "post"
        daily_data.append({
            "date": date_str,
            "count": count,
            "period": is_lockdown,
        })
        current += timedelta(days=1)
    
    with open(OUTPUT_DIR / "daily_counts.json", "w", encoding="utf-8") as f:
        json.dump(daily_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResult files saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 60)
    print("Statistics Complete!")
    print("=" * 60)
    
    return stats, lockdown_coverage

if __name__ == "__main__":
    stats, lockdown_coverage = main()
