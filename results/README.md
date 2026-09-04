# Results

This directory contains the *precomputed* numerical outputs of the TEGAT
experiment pipeline so that downstream readers can verify the numbers reported
in the paper without having to re-run the full training loop.

| File | Source script | Description |
| ---- | ------------- | ----------- |
| `main_results.json`              | `src/tegat_clean.py`         | Main comparison table (Table 3). |
| `main_results_clean.json`        | `src/tegat_model.py`         | Raw output from the primary model. |
| `ablation_results.json`          | `src/run_ablation.py`        | Component-ablation numbers (Table 4). |
| `feature_importance.json`        | `src/tegat_clean.py`         | RF + XGBoost feature importance. |
| `rq1_venue_risk.json`            | `src/rq1_venue_risk.py`      | RQ1 (venue risk) numerical outputs. |
| `rq2_rq3_results.json`           | `src/rq2_rq3_analysis.py`    | RQ2/RQ3 numerical outputs. |
| `rq3_causal_results.json`        | `src/rq3_causal_analysis.py` | Lockdown quasi-experimental analysis. |
| `results_summary.json`           | `src/...`                    | Cross-experiment rollup. |
| `results_summary.md`             | `src/...`                    | Human-readable executive summary. |
| `ablation_summary.md`            | `src/...`                    | Human-readable ablation summary. |
| `latex_tables.txt`               | `src/generate_tables.py`     | Generated LaTeX `\begin{table}` blocks for the paper. |
| `ablation_latex_tables.txt`      | `src/generate_ablation_tables.py` | LaTeX blocks for the ablation tables. |

All numbers are reproducible by running `python reproduce.py` from the project root.

## How to interpret a JSON file

```python
import json
with open("results/main_results.json", "r", encoding="utf-8") as f:
    table = json.load(f)

for method, metrics in table.items():
    print(f"{method:18s}  AUC={metrics['AUC-ROC']:.4f}  F1={metrics['F1']:.4f}")
```

Each entry is a flat dictionary with the standard keys:

```
AUC-ROC, AP, Accuracy, F1, Precision, Recall, P@20
```
