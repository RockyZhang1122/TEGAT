# 实验检查清单（Checklist）

## 论文投稿前必备项

### 数据准备

- [x] 数据集原始文件清点（30省txt文件）
- [x] 数据集统计：4,812例/30省/489天 ✅
- [x] 数据集划分脚本（70/10/20）
- [x] 武汉封城覆盖判断：✅ SUFFICIENT（封城前146天+后343天）
- [ ] 数据隐私合规检查（CSKEN公开需审查）

### 事件抽取

- [x] 事件抽取Pipeline实现 ✅
- [x] 抽取结果：6,194事件（77.5%有日期）
- [x] 事件类型分布统计
- [ ] 事件抽取质量人工评估

### 事件图构建

- [x] 异构图构建（5种节点类型） ✅
- [x] 节点特征工程（case/venue/date/transport/symptom） ✅
- [x] 图统计：4,812节点，31,147条case-case边

### 准实验分析（RQ3 因果推断增强）

- [x] ITS 分段回归（活跃周，n_active=10）✅
- [x] 格兰杰因果检验（lag=3 显著，p=0.025）✅
- [x] 封城前线性趋势稳健性检验（p=0.615 无显著趋势）✅
- [x] ITS + 春节协变量模型 ✅
- [x] 因果分析表格写入论文附录（Table 10）✅
- [x] Discussion 增加 Quasi-Experimental Validation 子节 ✅
- [x] Limitations 重写（承认无法做严格 DiD，省份管制时间数据未获取）✅

### 消融实验

- [x] Ablation-1 组件消融（6变体）✅
- [x] Ablation-2 图结构消融（时间窗口）✅
- [x] Ablation-3 特征组消融（5变体）✅
- [x] Robust-4 τ敏感性（τ=3/5/7/14天）✅
- [x] 消融结果写入论文附录（3个表格）✅
- [x] 消融分析段落写入主文 Results ✅

### 主实验

- [x] Exp1 事件抽取性能（规则+模式匹配）
- [x] Exp2 TEGAT风险预测性能 ✅ AUC=0.9858
- [x] Exp3 场所风险排序 ✅ ANOVA p=0.002
- [x] Exp4 无症状网络结构分析 ✅ p<0.05显著
- [x] Exp5 干预前后网络结构变化 ✅ 密度-38.6%

### 结果表

- [x] Table I 数据集统计 ✅
- [x] Table III TEGAT主实验结果 ✅
- [x] Table IV 场所风险排序 ✅
- [x] Table V 无症状网络对比 ✅
- [x] Table VI 干预效果对比 ✅
- [x] LaTeX表格生成 ✅

### 图表

- [ ] Fig 1 框架图（TEGAT架构）
- [ ] Fig 2 数据分布图（省份/时间）
- [ ] Fig 3 场所风险热图
- [ ] Fig 4 无症状网络对比
- [ ] Fig 5 干预前后网络演化

### 文档

- [x] 实验设计文档 ✅ v2.0
- [x] Claim-Evidence矩阵 ✅
- [x] 检查清单 ✅
- [ ] 代码仓库（GitHub）
- [ ] 数据集说明文档（README.md）

### 论文撰写

- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Method
- [ ] Experiments
- [ ] Discussion
- [ ] Conclusion

### 投稿准备

- [ ] 期刊格式检查（JBI）
- [ ] 引用格式统一
- [ ] 图表质量检查
- [ ] Cover Letter
- [ ] Author Contributions
- [ ] Conflict of Interest
- [ ] Data Availability Statement（CSKEN公开）

---

## 当前进度

| 阶段 | 状态 | 完成时间 |
|---|---|---|
| 题目确定 | ✅ 已完成 | - |
| 文献调研 | ✅ 已完成 | - |
| 实验设计 | ✅ 已完成 | 2026-09-02 |
| 数据预处理 | ✅ 已完成 | 2026-09-02 |
| 模型实现 | ✅ 已完成 | 2026-09-02 |
| 主实验 | ✅ 已完成 | 2026-09-02 |
| 科学发现分析 | ✅ 已完成 | 2026-09-02 |
| **论文撰写** | ⏳ **待开始** | - |