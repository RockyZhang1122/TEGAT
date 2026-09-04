"""Generate LaTeX ablation tables from ``results/ablation_results.json``.

Run from the repository root::

    python src/generate_ablation_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS_DIR  # noqa: E402

with open(RESULTS_DIR / "ablation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ab1 = data["ablation_components"]
ab2 = data["ablation_time_window"]
ab3 = data["ablation_feature_groups"]
tau = data["tau_sensitivity"]

METRICS = ["AUC-ROC", "AP", "Accuracy", "F1", "Precision", "Recall"]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


# ==================== Table: Component Ablation ====================
print("=== Table: Component Ablation (Ablation-1) ===")
base_full = ab1["TEGAT-Full"]
rows = []
for name, r in sorted(ab1.items(), key=lambda x: -(x[1]["AUC-ROC"])):
    deltas = {m: r[m] - base_full[m] for m in METRICS}
    rows.append((name, r, deltas))

# Print as LaTeX
latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\caption{Component ablation of TEGAT. Each row removes one component from the full model.}")
latex.append(r"\label{tab:ablation-components}")
latex.append(r"\small")
latex.append(r"\begin{tabular}{l|cccccc}")
latex.append(r"\toprule")
latex.append(r"Variant & AUC-ROC & AP & Acc & F1 & Prec & Rec \\")
latex.append(r"\midrule")
for name, r, deltas in rows:
    base = name == "TEGAT-Full"
    suffix = ""
    cells = [name]
    for m in METRICS:
        if base:
            cells.append(f"{fmt(r[m])}")
        else:
            d = deltas[m]
            sign = "+" if d >= 0 else ""
            cells.append(f"{fmt(r[m])} ({sign}{d:.4f})")
    latex.append(" & ".join(cells) + r" \\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
print("\n".join(latex))
print()

# ==================== Table: Time-window Ablation ====================
print("=== Table: Time-window Ablation (Ablation-2) ===")
latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\caption{Time-window ($\tau$) ablation. The co-location window is varied from 1 to 7 days.}")
latex.append(r"\label{tab:ablation-time}")
latex.append(r"\small")
latex.append(r"\begin{tabular}{c|cccccc}")
latex.append(r"\toprule")
latex.append(r"$\tau$ (days) & AUC-ROC & AP & Acc & F1 & Prec & Rec \\")
latex.append(r"\midrule")
for name, r in sorted(ab2.items(), key=lambda x: int(x[0].replace("tau=", ""))):
    cells = [name]
    for m in METRICS:
        cells.append(fmt(r[m]))
    latex.append(" & ".join(cells) + r" \\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
print("\n".join(latex))
print()

# ==================== Table: Feature-group Ablation ====================
print("=== Table: Feature-group Ablation (Ablation-3) ===")
base_full = ab3["TEGAT-Full"]
rows = []
for name, r in sorted(ab3.items(), key=lambda x: -(x[1]["AUC-ROC"])):
    deltas = {m: r[m] - base_full[m] for m in METRICS}
    rows.append((name, r, deltas))

latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(r"\caption{Feature-group ablation. Each row removes one feature group.}")
latex.append(r"\label{tab:ablation-features}")
latex.append(r"\small")
latex.append(r"\begin{tabular}{l|cccccc}")
latex.append(r"\toprule")
latex.append(r"Variant & AUC-ROC & AP & Acc & F1 & Prec & Rec \\")
latex.append(r"\midrule")
for name, r, deltas in rows:
    base = name == "TEGAT-Full"
    cells = [name]
    for m in METRICS:
        if base:
            cells.append(fmt(r[m]))
        else:
            d = deltas[m]
            sign = "+" if d >= 0 else ""
            cells.append(f"{fmt(r[m])} ({sign}{d:.4f})")
    latex.append(" & ".join(cells) + r" \\")
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
print("\n".join(latex))

# Also write to file
out_path = RESULTS_DIR / "ablation_latex_tables.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(latex))
print(f"\nSaved to {out_path}")
