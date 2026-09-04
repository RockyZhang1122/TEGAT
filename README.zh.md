# TEGAT: 时序事件图注意力网络

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](requirements.txt)

> **论文**：*TEGAT: A Temporal Event Graph Attention Network for Transmission Risk Analysis via Heterogeneous Event Networks from Multi-Provincial Epidemiological Investigation Texts.*
> 张恒、刘剑锋（并列通讯作者）、杨瑞
> 投稿于《生物医学信息学杂志》(Journal of Biomedical Informatics)。

## 项目简介

本仓库给出 **TEGAT** 模型的参考实现。该模型通过对中国大陆 30 个省级卫健委
公开发布的“**省确诊病例活动轨迹**”文本进行结构化抽取与异构事件图构建，
预测 COVID-19 确诊病例之间的**传播风险**。

流程涵盖：
- 4,812 例确诊 / 30 个省级行政区 / 489 天（2019-08-30 至 2020-12-31）
- 5 类节点构成的异构图（病例 / 场所 / 日期 / 交通 / 症状）
- 多头时序图注意力网络（TEGAT），对照 LR / SVM / MLP / RF / XGBoost 五种基线
- 五随机种子（42 / 123 / 456 / 789 / 1024）报告均值与标准差

## 目录

```
TEGAT/
├── reproduce.py           # 一键复现入口
├── requirements.txt
├── LICENSE                # MIT
├── CITATION.cff
├── README.md              # 英文 README
├── README.zh.md           # 本文件
├── src/                   # 数据 / 图 / 模型 / 评估全部源码
├── data/                  # CSKEN 结构化数据集
├── data_original/         # 30 个省级原始流调轨迹文本
├── results/               # 已生成的 JSON / Markdown 数值表
├── scripts/               # 验证与辅助脚本
├── docs/                  # 实验设计、数据来源、声明清单
├── paper/                 # LaTeX 原文、参考文献、最终 PDF
└── paper/figures/         # 6 张出版级图（PNG + PDF）
```

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/RockyZhang1122/TEGAT.git
cd TEGAT
pip install -r requirements.txt
```

### 2. 复现所有结果

```bash
python reproduce.py
```

该命令将依次执行：
1. 跳过事件抽取（仓内已含与原始抽取结果一致的 `events_extracted.json`）
2. 跳过构图（仓内已含 `graph_data.npz`）
3. 数据集统计
4. 主实验（TEGAT + 五种基线 × 五随机种子）
5. 组件消融
6. RQ1 场所风险分析、RQ2 隐性传播者分析、RQ3 干预分析
7. 重新生成论文中可直接使用的 LaTeX 表

实验结果输出到 `results/`，包括 JSON / Markdown 摘要与 LaTeX 表片段。

如需快速冒烟测试（约 5 分钟，单种子）：

```bash
python reproduce.py --fast
```

跳过指定步骤：

```bash
python reproduce.py --skip extract graph rq3
```

## 论文关键数值（Section 5.1）

| 方法             | AUC-ROC | F1    | Recall | P@20  |
| ---------------- | ------- | ----- | ------ | ----- |
| LR               | 0.9719  | 0.9619 | 0.9670 | ...   |
| SVM              | 0.9788  | 0.9650 | 0.9730 | ...   |
| MLP              | 0.9837  | 0.9694 | 0.9760 | ...   |
| RF               | 0.9858  | 0.9708 | 0.9970 | ...   |
| XGBoost          | 0.9861  | 0.9784 | 0.9850 | ...   |
| **TEGAT (本文)** | **0.9858** | **0.9800** | **0.9910** | ... |

完整表格（含 P@20 与五种子标准差）在 `results/main_results.json` 与
论文 `latex_tables.txt` 中。

## 数据说明

仓内附带的 CSKEN 数据集为**结构化衍生品**——由各省级卫健委公开的流调轨迹通告
整理而成。原始流调文本（`**省确诊病例活动轨迹.txt`）未随本仓发布；如需全文数据，
可通过 `docs/data_sources.md` 中给出的 30 个省级卫健委链接自行采集，
或等待接收论文后从 Zenodo 下载完整打包。详见 `data/README.md`。

## 引用

```bibtex
@article{zhang2024tegat,
  title={TEGAT: A Temporal Event Graph Attention Network for Transmission Risk
         Analysis via Heterogeneous Event Networks from Multi-Provincial
         Epidemiological Investigation Texts},
  author={Zhang, Heng and Liu, Jianfeng and Yang, Rui},
  journal={Journal of Biomedical Informatics},
  year={2024},
  note={Under review}
}
```

## 许可

本仓库源代码采用 **MIT 协议**（见 `LICENSE`）。
衍生数据集基于公开政府通告制作，仅供学术研究使用；详见 `data/README.md`。

## 可复现性声明

- 全部随机种子固定，论文报告 5 个种子的均值 ± 标准差。
- 代码不依赖 PyTorch，NumPy + scikit-learn 路径对相同 BLAS 实现完全确定。
- 仓内 `results/*.json` 于 2024-09-04 在 `numpy=1.24 / scikit-learn=1.3 /
  xgboost=1.7 / networkx=3.0` 环境中产出，复现时建议使用相同版本。

## 致谢

感谢 30 个省级卫健委发布的公开流调通告，以及前期 DBDC 2020 结构化轨迹
数据集的整理工作。
