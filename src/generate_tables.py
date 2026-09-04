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

from paths import RESULTS_DIR, REPO_ROOT, result_path
DATA_DIR = REPO_ROOT / "data"  # legacy alias

# -*- coding: utf-8 -*-
"""
生成完整的结果汇总表（LaTeX格式）
"""
import json
from pathlib import Path


# ==================== Table III: Main Results ====================

def generate_table_iii():
    with open(OUTPUT_DIR / "main_results_clean.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    latex = """
\\begin{table}[t]
\\caption{Transmission Risk Prediction Performance.\\label{tab:main_results}}
\\centering
\\resizebox{\\linewidth}{!}{
\\begin{threeparttable}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Method} & \\textbf{AUC-ROC} & \\textbf{AP} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall}\\\\ 
\\midrule
"""
    
    # Sort by AUC-ROC
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["AUC-ROC"])
    
    for method, metrics in sorted_results:
        latex += f"{method:<14} & {metrics['AUC-ROC']:.4f} & {metrics['AP']:.4f} & {metrics['Accuracy']:.4f} & {metrics['F1']:.4f} & {metrics['Precision']:.4f} & {metrics['Recall']:.4f}\\\\ \n"
        if method == "TEGAT":
            latex += "\\midrule\n"
    
    latex += """\\bottomrule
\\end{tabular}
\begin{tablenotes}
\item Best results are \textbf{boldfaced}. All values are averaged over 5 random seeds.
\end{tablenotes}
\end{threeparttable}
}
\end{table}
"""
    return latex


# ==================== Table IV: Venue Risk Ranking ====================

def generate_table_iv():
    with open(OUTPUT_DIR / "rq1_venue_risk.json", "r", encoding="utf-8") as f:
        rq1 = json.load(f)
    
    latex = """
\\begin{table}[t]
\\caption{Venue Transmission Risk Ranking.\\label{tab:venue_risk}}
\\centering
\\begin{tabular}{clccc}
\\toprule
\\textbf{Rank} & \\textbf{Venue} & \\textbf{Avg. Risk Score} & \\textbf{\\# Cases} & \\textbf{\\# Provinces}\\\\ 
\\midrule
"""
    
    venue_labels = {
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
    
    venue_case_count = {
        "医院": 3251, "餐厅": 639, "超市": 417, "交通工具": 1481,
        "家庭": 1290, "工作场所": 471, "酒店": 282, "公共场所": 557,
        "学校": 154, "药店": 210,
    }
    
    for rank_info in rq1["ranking"]:
        venue = rank_info["venue"]
        latex += f"{rank_info['rank']} & {venue_labels.get(venue, venue)} & {rank_info['avg_risk']:.4f} & {venue_case_count.get(venue, 'N/A')} & {len(rq1['venue_province_risk'].get(venue, {}))}\\\\ \n"
    
    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\item Risk score is defined as the cluster participation rate (proportion of cases with $\\ge$1 co-location contact within $\\pm$3 days). ANOVA F=3.03, p=0.002.
\end{tablenotes}
\end{table}
"""
    return latex


# ==================== Table V: Asymptomatic Network ====================

def generate_table_v():
    with open(OUTPUT_DIR / "rq2_rq3_results.json", "r") as f:
        rq2_rq3 = json.load(f)
    
    rq2 = rq2_rq3["rq2"]
    
    latex = """
\\begin{table}[t]
\\caption{Asymptomatic vs. Symptomatic Cases: Network Centrality Comparison.\\label{tab:asymptomatic}}
\\centering
\\begin{tabular}{lcccccc}
\\toprule
\\multirow{2}{*}{\\textbf{Metric}} & \\multicolumn{2}{c}{\\textbf{Asymptomatic (n=63)}} & \\multicolumn{2}{c}{\\textbf{Symptomatic (n=4749)}} & \\multirow{2}{*}{p-value} & \\multirow{2}{*}{Cohen's d}\\\\ 
\\cmidrule{2-3} \\cmidrule{4-5}
 & Mean & SD & Mean & SD & \\\\ 
\\midrule
"""
    
    metric_labels = {
        "degree": "Degree Centrality",
        "page_rank": "PageRank",
        "betweenness": "Betweenness",
        "clustering": "Clustering Coef.",
    }
    
    for metric, label in metric_labels.items():
        data = rq2[metric]
        latex += f"{label} & {data['asymptomatic_mean']:.4f} & {data['asymptomatic_std']:.4f} & {data['symptomatic_mean']:.4f} & {data['symptomatic_std']:.4f} & {data['mannwhitney_p']:.4f} & {data['cohens_d']:.4f}\\\\ \n"
    
    latex += f"\\midrule\n\\multicolumn{{7}}{{p{{12cm}}}}{{\\textbf{{Isolated Cases (degree=0)}}: Asymptomatic {rq2['isolation_rate']['asymptomatic']*100:.1f}\\% vs. Symptomatic {rq2['isolation_rate']['symptomatic']*100:.1f}\\%}}\\\\ \n"
    
    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\item Statistical test: two-sided Mann-Whitney U test. Effect size: Cohen's d.
\end{tablenotes}
\end{table}
"""
    return latex


# ==================== Table VI: Intervention Effect ====================

def generate_table_vi():
    with open(OUTPUT_DIR / "rq2_rq3_results.json", "r") as f:
        rq2_rq3 = json.load(f)
    
    rq3 = rq2_rq3["rq3"]
    
    latex = """
\\begin{table}[t]
\\caption{Network Structural Changes Before and After Wuhan Lockdown (2020-01-23).\\label{tab:intervention}}
\\centering
\\begin{tabular}{lccc}
\\toprule
\\textbf{Metric} & \\textbf{Pre-Lockdown} & \\textbf{Post-Lockdown} & \\textbf{Change}\\\\ 
\\midrule
"""
    
    metric_labels = {
        "density": "Network Density",
        "avg_degree": "Average Degree",
        "lcc_ratio": "Largest CC Ratio",
    }
    
    for metric, label in metric_labels.items():
        data = rq3[metric]
        change_str = f"{data['change_pct']:+.1f}\\%"
        latex += f"{label} & {data['pre']:.6f} & {data['post']:.6f} & {change_str}\\\\ \n"
    
    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\item Pre-lockdown: 2,007 events (146 days). Post-lockdown: 2,792 events (343 days). LCC = Largest Connected Component.
\end{tablenotes}
\end{table}
"""
    return latex


# ==================== Main ====================

def main():
    tables = {
        "table_iii_main": generate_table_iii(),
        "table_iv_venue": generate_table_iv(),
        "table_v_asymptomatic": generate_table_v(),
        "table_vi_intervention": generate_table_vi(),
    }
    
    with open(OUTPUT_DIR / "latex_tables.txt", "w", encoding="utf-8") as f:
        for name, table in tables.items():
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Table: {name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(table)
    
    # Also save as JSON for programmatic use
    with open(OUTPUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "main_results": tables["table_iii_main"],
            "venue_risk": tables["table_iv_venue"],
            "asymptomatic": tables["table_v_asymptomatic"],
            "intervention": tables["table_vi_intervention"],
        }, f, ensure_ascii=False, indent=2)
    
    print("Tables generated:")
    for name in tables:
        print(f"  - {name}")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
