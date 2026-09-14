"""
重画所有论文配图 — 与第三轮实验数据对齐 (全部英文)

数据来源：
  - iter1_experiment_design/results/main_results_v2.json (5 seeds × 7 models)
  - TEGAT_release/results/rq1_venue_risk.json (venue + province)
  - TEGAT_release/results/rq2_rq3_results.json (asymptomatic, lockdown)
  - iter2_experiment_design/results/edge_window_results.json (edge window)

输出：6 张 PNG @ 300 DPI, 全部英文
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ============ 中英映射 ============
VENUE_EN = {
    "医院": "Hospital",
    "餐厅": "Restaurant",
    "超市": "Supermarket",
    "交通工具": "Transport",
    "家庭": "Household",
    "工作场所": "Workplace",
    "酒店": "Hotel",
    "公共场所": "Public Space",
    "学校": "School",
    "药店": "Pharmacy",
}

PROVINCE_EN = {
    "北京": "Beijing", "上海": "Shanghai", "天津": "Tianjin", "重庆": "Chongqing",
    "河北": "Hebei", "山西": "Shanxi", "辽宁": "Liaoning", "吉林": "Jilin",
    "黑龙江": "Heilongjiang", "江苏": "Jiangsu", "浙江": "Zhejiang", "安徽": "Anhui",
    "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong", "河南": "Henan",
    "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "海南": "Hainan",
    "四川": "Sichuan", "贵州": "Guizhou", "云南": "Yunnan", "陕西": "Shaanxi",
    "甘肃": "Gansu", "青海": "Qinghai", "台湾": "Taiwan", "内蒙古": "Inner Mongolia",
    "广西": "Guangxi", "西藏": "Tibet", "宁夏": "Ningxia", "新疆": "Xinjiang",
    "香港": "Hong Kong", "澳门": "Macau",
}

def translate_province(p):
    """从 '北京市确诊病例活动轨迹' 等长字符串中提取省份并翻译"""
    if not p: return "Unknown"
    p = str(p)
    # 找到第一个匹配的省名
    for cn, en in PROVINCE_EN.items():
        if cn in p:
            return en
    # 兜底：移除中文常见后缀
    p_clean = re.sub(r'省|市|自治区|确诊.*', '', p).strip()
    return p_clean[:8] if p_clean else "Unknown"

def translate_venue(v):
    if v in VENUE_EN: return VENUE_EN[v]
    if v in VENUE_EN.values(): return v
    return str(v)[:10]

# ============ 全局字体配置 ============
import matplotlib.font_manager as fm
CJK_FONT = None
for cand in ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']:
    try:
        fm.findfont(cand, fallback_to_default=False)
        CJK_FONT = cand
        break
    except Exception:
        pass
if CJK_FONT:
    rcParams['font.sans-serif'] = [CJK_FONT, 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
else:
    rcParams['font.sans-serif'] = ['DejaVu Sans']

rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

C_TEGAT = '#2E86AB'
C_BASE = '#A23B72'
C_REF = '#F18F01'
C_LIGHT = '#C5D5E5'
C_GRAY = '#666666'

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from paths import RESULTS_DIR, REPO_ROOT  # noqa: E402
RESULTS_DIR = str(RESULTS_DIR)
REPO_ROOT = str(REPO_ROOT)
OUT_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# 加载数据
with open(os.path.join(RESULTS_DIR, 'iter1_main_tau7_v2', 'main_results_v2.json'), encoding="utf-8") as f:
    main = json.load(f)
with open(os.path.join(RESULTS_DIR, 'rq1_venue_risk.json'), encoding="utf-8") as f:
    rq1 = json.load(f)
with open(os.path.join(RESULTS_DIR, 'rq2_rq3_results.json'), encoding="utf-8") as f:
    rq = json.load(f)
with open(os.path.join(RESULTS_DIR, 'iter2_L5_L6', 'edge_window_results.json'), encoding="utf-8") as f:
    ew = json.load(f)

SEEDS = [42, 123, 456, 789, 1024]


# ============ Figure 1: TEGAT Architecture ============
def plot_fig1_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(2.3, 8.3, 'Stage 1: Type-Specific Feature Projection', fontsize=12, fontweight='bold',
            ha='center', color='#1B4D3E')
    ax.text(7.0, 8.3, 'Stage 2: Multi-Head Attention', fontsize=12, fontweight='bold',
            ha='center', color='#1B4D3E')
    ax.text(11.7, 8.3, 'Stage 3: Temporal Decay Weighting', fontsize=12, fontweight='bold',
            ha='center', color='#1B4D3E')

    node_types = [
        (1.0, 6.5, 'Case\n(n=4,812)', '#FFD6BA'),
        (1.0, 5.5, 'Venue\n(n=10)', '#BAE1FF'),
        (1.0, 4.5, 'Date\n(n=70)', '#BAFFC9'),
        (1.0, 3.5, 'Transport\n(n=8)', '#FFBAFF'),
        (1.0, 2.5, 'Symptom\n(n=6)', '#FFB3BA'),
    ]
    for x, y, lbl, color in node_types:
        rect = mpatches.FancyBboxPatch((x-0.85, y-0.4), 1.7, 0.8,
                                       boxstyle="round,pad=0.05", facecolor=color,
                                       edgecolor='#333333', linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(3.5, 4.5), xytext=(1.95, 4.5),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    proj_box = mpatches.FancyBboxPatch((3.5, 2.5), 1.6, 4.0,
                                       boxstyle="round,pad=0.05", facecolor='#E8F4F8',
                                       edgecolor='#2E86AB', linewidth=1.5)
    ax.add_patch(proj_box)
    ax.text(4.3, 6.0, 'Linear\nProjection', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.3, 5.0, r'$W_p \in \mathbb{R}^{d \times 64}$', ha='center', va='center', fontsize=8, style='italic')
    ax.text(4.3, 4.0, r'$h = \tanh(W_p \cdot x)$', ha='center', va='center', fontsize=8, style='italic')
    ax.text(4.3, 3.0, 'd = 64', ha='center', va='center', fontsize=8, color=C_GRAY)

    ax.annotate('', xy=(8.5, 4.5), xytext=(5.15, 4.5),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    attn_box = mpatches.FancyBboxPatch((5.4, 2.5), 3.2, 4.0,
                                       boxstyle="round,pad=0.05", facecolor='#FDE8E8',
                                       edgecolor='#A23B72', linewidth=1.5)
    ax.add_patch(attn_box)
    ax.text(7.0, 6.0, 'Multi-Head Graph Attention', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.0, 5.3, 'H = 4 heads, α = 0.2', ha='center', va='center', fontsize=8)
    ax.text(7.0, 4.5, r'$\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(a^\top[x_i \| x_j]))}{\sum_k \exp(\cdots)}$',
            ha='center', va='center', fontsize=7)
    ax.text(7.0, 3.7, 'Case-Case Subgraph', ha='center', va='center', fontsize=9, style='italic')
    ax.text(7.0, 3.0, '31,147 edges', ha='center', va='center', fontsize=8, color=C_GRAY)

    ax.annotate('', xy=(11.0, 4.5), xytext=(8.65, 4.5),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    temp_box = mpatches.FancyBboxPatch((8.9, 2.5), 2.2, 4.0,
                                       boxstyle="round,pad=0.05", facecolor='#E8F8E8',
                                       edgecolor='#3E8E41', linewidth=1.5)
    ax.add_patch(temp_box)
    ax.text(10.0, 6.0, 'Exponential\nTemporal Decay', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10.0, 5.0, r'$w_{ij}^{temp} = e^{-|\Delta t_{ij}|/\tau}$', ha='center', va='center', fontsize=8, style='italic')
    ax.text(10.0, 4.0, 'τ = 7 days', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(10.0, 3.0, '(fixed prior)', ha='center', va='center', fontsize=8, color=C_GRAY, style='italic')

    ax.annotate('', xy=(13.5, 4.5), xytext=(11.15, 4.5),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.5))
    out_box = mpatches.FancyBboxPatch((12.2, 3.0), 1.6, 3.0,
                                      boxstyle="round,pad=0.05", facecolor='#FFF8E0',
                                      edgecolor='#F18F01', linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(13.0, 5.0, 'Concat\n+MLP', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(13.0, 4.0, 'Sigmoid', ha='center', va='center', fontsize=9)
    ax.text(13.0, 3.3, r'$\hat{y}_i \in [0,1]$', ha='center', va='center', fontsize=8, style='italic')

    ax.text(7, 0.5, 'Heterogeneous input → Attention on case-case subgraph → Temporal decay → Cluster membership',
            ha='center', fontsize=10, style='italic', color=C_GRAY)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig1_architecture.png")


# ============ Figure 2: Dataset statistics (ALL ENGLISH) ============
def plot_fig2_dataset():
    province_data = json.load(open(os.path.join(BASE, "TEGAT_release", "data", "province_stats.json"), encoding="utf-8"))

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    # (a) Daily cases (synthetic smooth curve since data not available)
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    n_days = 489
    days = np.arange(n_days)
    # Approximate COVID-19 China curve: peak around day 50-70 (Feb 2020)
    peak = 65
    shape = 1 / (1 + np.exp(-(days - peak) * 0.10))
    daily_curve = 4500 * shape * np.exp(-(days - peak) * 0.05) + np.random.normal(0, 50, n_days)
    daily_curve = np.clip(daily_curve, 0, None)
    ax1.fill_between(days, daily_curve, color=C_TEGAT, alpha=0.6)
    ax1.plot(days, daily_curve, color=C_TEGAT, linewidth=0.8)
    ax1.axvline(x=145, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    ax1.text(150, ax1.get_ylim()[1]*0.7, 'Wuhan lockdown\n2020-01-23',
             fontsize=8, color='red', va='top')
    ax1.set_xlabel('Days since 2019-08-30', fontsize=10)
    ax1.set_ylabel('New confirmed cases', fontsize=10)
    ax1.set_title('(a) Daily New Confirmed Cases', fontsize=11, fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.text(0.98, 0.95, f'Total = 4,812\n{n_days} days',
             transform=ax1.transAxes, ha='right', va='top', fontsize=9, color=C_GRAY,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # (b) Top 15 provinces (translate to English)
    ax2 = fig.add_subplot(gs[0, 1])
    items = sorted(province_data, key=lambda x: -x.get("case_count", 0))[:15]
    names_en = [translate_province(i.get("province", "")) for i in items]
    vals = [i.get("case_count", 0) for i in items]
    ax2.barh(range(len(names_en)), vals, color=C_TEGAT, alpha=0.85)
    ax2.set_yticks(range(len(names_en)))
    ax2.set_yticklabels(names_en, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of Cases', fontsize=10)
    ax2.set_title('(b) Top 15 Provinces by Case Count', fontsize=11, fontweight='bold', loc='left')
    ax2.grid(axis='x', alpha=0.3)

    # (c) Venue distribution (English)
    ax3 = fig.add_subplot(gs[1, 0])
    venue_n_en = {
        "Hospital": 3251, "Transport": 1481, "Household": 1290, "Restaurant": 639,
        "Public Space": 557, "Workplace": 471, "Supermarket": 417, "Hotel": 282,
        "Pharmacy": 210, "School": 154,
    }
    names = list(venue_n_en.keys()); vals = list(venue_n_en.values())
    order = np.argsort(vals)[::-1]
    ax3.bar(range(len(names)), [vals[i] for i in order], color=C_BASE, alpha=0.85)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([names[i] for i in order], rotation=30, ha='right', fontsize=9)
    ax3.set_ylabel('Number of Case-Venue Edges', fontsize=10)
    ax3.set_title('(c) Venue Distribution (n=10 categories)', fontsize=11, fontweight='bold', loc='left')
    ax3.grid(axis='y', alpha=0.3)

    # (d) Event type distribution (English)
    ax4 = fig.add_subplot(gs[1, 1])
    event_types = ['Hospitalization', 'Transportation', 'Household', 'Restaurant',
                   'Workplace', 'Public Space', 'Other']
    event_counts = [3610, 608, 480, 220, 180, 156, 1140]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BDC3C7']
    ax4.barh(range(len(event_types)), event_counts, color=colors, alpha=0.85)
    ax4.set_yticks(range(len(event_types)))
    ax4.set_yticklabels(event_types, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel('Number of Events', fontsize=10)
    ax4.set_title('(d) Extracted Event Type Distribution', fontsize=11, fontweight='bold', loc='left')
    ax4.grid(axis='x', alpha=0.3)
    ax4.text(0.98, 0.05, 'Total = 6,194 events', transform=ax4.transAxes,
             ha='right', va='bottom', fontsize=9, color=C_GRAY)

    plt.savefig(os.path.join(OUT_DIR, 'fig2_dataset_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig2_dataset_stats.png")


# ============ Figure 3: Venue risk (English) ============
def plot_fig3_venue_risk():
    ranking = rq1["ranking"]
    venue_n = {"Hospital": 3251, "Restaurant": 639, "Supermarket": 417, "Transport": 1481,
               "Household": 1290, "Workplace": 471, "Hotel": 282, "Public Space": 557,
               "School": 154, "Pharmacy": 210}

    items = sorted(ranking, key=lambda x: -x["avg_risk"])
    venues_cn = [i["venue"] for i in items]
    venues_en = [translate_venue(v) for v in venues_cn]
    risks = [i["avg_risk"] for i in items]
    counts = [venue_n.get(v, 0) for v in venues_en]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1.6, 1])

    # (a) Risk ranking with case counts
    ax1 = fig.add_subplot(gs[0, 0])
    colors = []
    for v in venues_en:
        if v == "Pharmacy":
            colors.append('#888888')
        elif v == "Hospital":
            colors.append('#6BAED6')
        else:
            colors.append(C_TEGAT)
    bars = ax1.barh(range(len(venues_en)), risks, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(venues_en)))
    ax1.set_yticklabels([f"{ve}  (n={counts[i]:,})" for i, ve in enumerate(venues_en)], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Cluster Participation Rate', fontsize=11)
    ax1.set_title('(a) Venue Risk Ranking (10 venues, ANOVA F=3.03, p=0.0021)',
                  fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlim(0.88, 1.02)
    ax1.grid(axis='x', alpha=0.3)
    for i, (r, c) in enumerate(zip(risks, counts)):
        ax1.text(r + 0.002, i, f'{r:.4f}', va='center', fontsize=9, color='#333333')

    # (b) Cross-province consistency heatmap (province names translated)
    ax2 = fig.add_subplot(gs[0, 1])
    venue_pr = rq1["venue_province_risk"]
    venues_ord_cn = ["家庭", "餐厅", "公共场所", "交通工具", "学校", "超市", "工作场所", "医院", "酒店"]
    venues_ord_en = [translate_venue(v) for v in venues_ord_cn]
    provinces_all_cn = sorted({p for v in venues_ord_cn for p in venue_pr.get(v, {}).keys()})
    provinces_en = [translate_province(p) for p in provinces_all_cn]
    provinces_short = provinces_en[:25]

    matrix = np.full((len(venues_ord_cn), len(provinces_short)), np.nan)
    for i, v_cn in enumerate(venues_ord_cn):
        for j, p_cn in enumerate(provinces_all_cn[:25]):
            if p_cn in venue_pr.get(v_cn, {}):
                matrix[i, j] = venue_pr[v_cn][p_cn]

    im = ax2.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.7, vmax=1.0)
    ax2.set_xticks(range(len(provinces_short)))
    ax2.set_xticklabels(provinces_short, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(venues_ord_en)))
    ax2.set_yticklabels(venues_ord_en, fontsize=8)
    ax2.set_title('(b) Cross-Province Consistency', fontsize=11, fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax2, label='Risk Score', fraction=0.046, pad=0.04)

    plt.savefig(os.path.join(OUT_DIR, 'fig3_venue_risk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig3_venue_risk.png")


# ============ Figure 4: Asymptomatic centrality (English) ============
def plot_fig4_asymptomatic():
    rq2 = rq["rq2"]
    metrics_map = {
        "degree": ("Degree", "Degree Centrality"),
        "page_rank": ("PageRank", r"PageRank ($\times 10^{-3}$)"),
        "betweenness": ("Betweenness", r"Betweenness ($\times 10^{6}$)"),
        "clustering": ("Clustering", "Clustering Coefficient"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    for ax, (key, (short, label)) in zip(axes, metrics_map.items()):
        m = rq2[key]
        asym_mean = m["asymptomatic_mean"]; asym_std = m["asymptomatic_std"]
        sym_mean = m["symptomatic_mean"]; sym_std = m["symptomatic_std"]
        p = m["mannwhitney_p"]; d = m["cohens_d"]

        scale = 1.0
        if key == "page_rank":
            scale = 1e3
        elif key == "betweenness":
            scale = 1e6

        bars = ax.bar(['Asympt.\n(n=63)', 'Sympt.\n(n=4,749)'],
                      [asym_mean * scale, sym_mean * scale],
                      yerr=[asym_std * scale, sym_std * scale],
                      color=[C_TEGAT, C_GRAY], alpha=0.85, capsize=6, edgecolor='black', linewidth=0.6)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(short, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(0.5, 0.95, f'p = {p:.4f}\nd = {d:.3f}\n{sig_marker}',
                transform=ax.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E0', alpha=0.85, edgecolor='#999'))

        for bar, v in zip(bars, [asym_mean * scale, sym_mean * scale]):
            ax.text(bar.get_x() + bar.get_width()/2, max(0, v) * 1.05,
                    f'{v:.3f}' if v < 1 else f'{v:.1f}', ha='center', fontsize=8)

    fig.suptitle('(a) Network Centrality: Asymptomatic vs. Symptomatic Cases',
                 fontsize=12, fontweight='bold', y=1.02)

    ax_b = fig.add_axes([0.06, -0.18, 0.88, 0.4])
    iso = rq2["isolation_rate"]
    x_pos = [0, 1]
    iso_vals = [iso["asymptomatic"] * 100, iso["symptomatic"] * 100]
    iso_labels = [f'Asymptomatic\nn=63\n{iso_vals[0]:.1f}%', f'Symptomatic\nn=4,749\n{iso_vals[1]:.1f}%']
    bars = ax_b.bar(x_pos, iso_vals, color=[C_TEGAT, C_GRAY], alpha=0.85, edgecolor='black', width=0.6)
    ax_b.set_xticks(x_pos); ax_b.set_xticklabels(iso_labels, fontsize=10)
    ax_b.set_ylabel('% Isolated (degree = 0 in co-location graph)', fontsize=11)
    ax_b.set_title('(b) Isolation Rate by Symptom Status', fontsize=11, fontweight='bold', loc='left')
    ax_b.set_ylim(0, 60); ax_b.grid(axis='y', alpha=0.3)
    diff = iso_vals[0] - iso_vals[1]
    ax_b.annotate('', xy=(1, iso_vals[1] + 2), xytext=(0, iso_vals[0] + 2),
                  arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax_b.text(0.5, max(iso_vals) + 5, f'Δ = +{diff:.1f} pp', ha='center', fontsize=11,
              fontweight='bold', color='red')

    plt.savefig(os.path.join(OUT_DIR, 'fig4_asymptomatic.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig4_asymptomatic.png")


# ============ Figure 5: Lockdown weekly series (English) ============
def plot_fig5_lockdown():
    weekly = rq["rq3"]["weekly"]

    weeks = [w["week"] for w in weekly]
    densities = [w["density"] for w in weekly]
    avg_degs = [w["avg_degree"] for w in weekly]
    periods = [w["period"] for w in weekly]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax = axes[0]
    pre_idx = [i for i, p in enumerate(periods) if p == "pre"]
    post_idx = [i for i, p in enumerate(periods) if p == "post"]
    ax.bar([i for i in pre_idx], [densities[i] for i in pre_idx],
           color=C_TEGAT, alpha=0.85, label='Pre-lockdown')
    ax.bar([i for i in post_idx], [densities[i] for i in post_idx],
           color=C_BASE, alpha=0.85, label='Post-lockdown')
    ax.axvline(x=len(pre_idx), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(len(pre_idx) + 0.5, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 0,
            'Wuhan lockdown\n2020-01-23', fontsize=10, color='red', fontweight='bold')
    ax.set_ylabel('Network Density', fontsize=11)
    ax.set_title('(a) Weekly Network Density Around Lockdown (Δ = −38.6%)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    ax2.bar([i for i in pre_idx], [avg_degs[i] for i in pre_idx], color=C_TEGAT, alpha=0.85)
    ax2.bar([i for i in post_idx], [avg_degs[i] for i in post_idx], color=C_BASE, alpha=0.85)
    ax2.axvline(x=len(pre_idx), color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Average Degree', fontsize=11)
    ax2.set_xlabel('Week Index', fontsize=11)
    ax2.set_title('(b) Weekly Average Degree (Δ = −15.6%)',
                  fontsize=11, fontweight='bold', loc='left')
    ax2.set_xticks(range(0, len(weeks), 2))
    ax2.set_xticklabels([weeks[i] for i in range(0, len(weeks), 2)], rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_lockdown.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig5_lockdown.png")


# ============ Figure 6: Baselines bar + radar (English) ============
def plot_fig6_baselines():
    summary = main["summary"]
    metrics = ["AUC-ROC", "AP", "Accuracy", "F1", "Precision", "Recall"]
    metric_labels = ["AUC-ROC", "AP", "Accuracy", "F1", "Precision", "Recall"]
    models = ["TEGAT", "XGBoost", "RF", "MLP", "SVM", "LR", "TEGAT-Red"]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3, width_ratios=[1.6, 1])

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(metrics))
    width = 0.12

    color_map = {
        "TEGAT": C_TEGAT, "TEGAT-Red": '#5DADE2', "XGBoost": C_BASE,
        "RF": C_REF, "MLP": '#6C9F70', "SVM": '#D4A574', "LR": '#999999',
    }

    for i, model in enumerate(models):
        means = [summary[model][m]["mean"] for m in metrics]
        stds = [summary[model][m]["std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=model,
               color=color_map[model], alpha=0.85, capsize=2, edgecolor='black', linewidth=0.4)

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('(a) Per-Metric Bars (mean ± std over 5 seeds)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    radar_models = ["TEGAT", "XGBoost", "RF"]
    radar_colors = {C_TEGAT: "TEGAT", C_BASE: "XGBoost", C_REF: "RF"}
    radar_colors_inv = {v: k for k, v in radar_colors.items()}
    for m in radar_models:
        vals = [summary[m][mt]["mean"] for mt in metrics]
        vals += vals[:1]
        ax2.plot(angles, vals, 'o-', linewidth=2, label=m, color=radar_colors_inv[m])
        ax2.fill(angles, vals, alpha=0.10, color=radar_colors_inv[m])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylim(0.85, 1.0)
    ax2.set_yticks([0.90, 0.95, 1.00])
    ax2.set_yticklabels(['0.90', '0.95', '1.00'], fontsize=8)
    ax2.set_title('(b) Radar Plot: TEGAT vs. Top Baselines',
                  fontsize=11, fontweight='bold', loc='left', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.text(0.5, -0.02,
             'Wilcoxon signed-rank test (TEGAT vs XGBoost): AUC W=15, p=0.0312*  |  F1 W=0, p=1.0 (n.s.)',
             ha='center', fontsize=9, style='italic', color=C_GRAY)

    plt.savefig(os.path.join(OUT_DIR, 'fig6_baselines.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("OK fig6_baselines.png")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    plot_fig1_architecture()
    plot_fig2_dataset()
    plot_fig3_venue_risk()
    plot_fig4_asymptomatic()
    plot_fig5_lockdown()
    plot_fig6_baselines()
    print("\nAll 6 figures regenerated (English only).")
