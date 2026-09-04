# Claim-Evidence 矩阵

## 论文标题

> Constructing and Analyzing COVID-19 Spatiotemporal Event Networks from Multi-Provincial Epidemiological Investigation Texts

## 中央声明（Central Claims）

### C1（方法学贡献）

**Claim**：We propose Temporal Event Graph Attention Network (TEGAT), a novel framework that constructs spatiotemporal event graphs from epidemiological investigation texts and predicts COVID-19 transmission risk.

**支撑实验**：

| Evidence | 数据 | 指标 | 评估 |
|---|---|---|---|
| E1.1 事件抽取有效性 | 人工标注子集 | Precision/Recall/F1 | >0.85 |
| E1.2 TEGAT风险预测 | Train/Val/Test | AUC-ROC | >0.75 |
| E1.3 优于现有方法 | 同上 | vs. 6 baselines | 显著提升 |

**反证条件**：
- 如果TEGAT与GAT/RGCN相比无显著差异 → C1弱化
- 如果事件抽取F1<0.7 → Pipeline无效

---

### C2（数据集贡献）

**Claim**：We construct CSKEN, the largest multi-provincial COVID-19 spatiotemporal event network dataset covering 4,812 cases across 30 provinces.

**支撑证据**：

| Evidence | 描述 |
|---|---|
| E2.1 数据规模 | 4,812 cases, 30 provinces |
| E2.2 时间跨度 | 2020-01 ~ 2020-04 |
| E2.3 事件密度 | Average 5-10 events/case |
| E2.4 公开可用 | HuggingFace/GitHub release |

**反证条件**：
- 如果实际规模<1000 → 数据集贡献减弱
- 如果不能公开 → 需修改贡献声明

---

### C3（科学发现 - 场所风险）

**Claim**：Venue type significantly affects COVID-19 transmission risk, with households and restaurants/transportation identified as high-risk venues.

**支撑实验**：

| Evidence | 数据 | 指标 | 评估 |
|---|---|---|---|
| E3.1 风险评分差异 | All venues | ANOVA + Tukey | p<0.05 |
| E3.2 跨省一致性 | Per-province | Krippendorff's α | >0.7 |
| E3.3 与朴素统计对比 | All venues | Effect size | Cohen's d>0.5 |

**反证条件**：
- 如果各场所风险无显著差异 → C3不成立
- 如果跨省不一致 → C3弱化

---

### C4（科学发现 - 无症状网络角色）

**Claim**：Asymptomatic carriers exhibit distinct structural roles in the COVID-19 event network, with higher PageRank and different centrality patterns compared to symptomatic cases.

**支撑实验**：

| Evidence | 数据 | 指标 | 评估 |
|---|---|---|---|
| E4.1 中心性差异 | Symptomatic vs Asymptomatic | Wilcoxon | p<0.05 |
| E4.2 网络占位模式 | 同上 | Visual + metric | Qualitative |
| E4.3 传播角色 | 同上 | Spread index | TBD |

**反证条件**：
- 如果无症状与有症状网络指标无差异 → C4不成立
- 如果样本量不足（n<30） → 统计功效低

---

### C5（科学发现 - 干预效果）

**Claim**：Non-pharmaceutical interventions (NPIs), particularly the Wuhan lockdown, significantly altered the COVID-19 transmission network structure, reducing network density and average path length.

**支撑实验**：

| Evidence | 数据 | 指标 | 评估 | 实际结果 |
|---|---|---|---|---|
| E5.1 时序网络指标 | Weekly active weeks | Density/Degree/LCC | 显著变化 | 密度 -38.6%, 度 -15.6%, LCC -26.5% |
| E5.2 ITS 分段回归 | 活跃周 n=10 | β2 lockdown-level | 显著负效应 | β2=-0.0962 (density), -2.168 (avg_deg) |
| E5.3 格兰杰因果 | 活跃周 lag 1-3 | F-test | lag=3 显著 | F=3.33, p=0.025 (avg_deg) |
| E5.4 封城前趋势 | Pre-lockdown weeks | Linear regression | 无显著趋势 | slope p=0.615 (density) |
| E5.5 与SEIR对比 | SEIR vs Network | Trend | Consistent | 文献对比 |

**准实验设计**（已实施）：
- 中断时间序列（ITS）分段回归
- 格兰杰因果检验
- 封城前线性趋势稳健性检验
- ITS + 春节协变量（控制季节效应）

**反证条件**：
- 如果网络指标无显著时序变化 → C5不成立
- 如果封城前已有显著下降趋势 → 需更保守解读（实际：封城前无显著趋势，结果可信）
- 如果 ITS 和 Granger 都不显著 → C5 仅作为观察性证据（实际：Granger lag=3 显著）

**局限性**（已在论文说明）：
- 各省管制时间数据未获取，无法做严格 DiD
- 活跃周样本量小（n=10），ITS 显著性受样本量限制
- 不能完全排除并发因素（自愿行为变化、检测能力提升、报告周期）
- 已通过 ITS+Granger+Pre-trend 三重稳健性检验部分缓解

---

## 风险与缓解

| 风险 | 概率 | 缓解策略 |
|---|---|---|
| C1/C3冲突（TEGAT好但场所风险不显著） | 低 | 即使差异不显著，TEGAT本身仍有方法学价值 |
| C4样本量不足 | 中 | 与领域专家合作标注 / 弱化结论 |
| C5受混杂因素影响 | 高 | 用多个时间点 + 多个指标交叉验证 |
| Baseline超越TEGAT | 中 | 加强特征工程 + 调整超参数 |
| 数据隐私问题 | 低 | 申请伦理审批 + 去标识化 |