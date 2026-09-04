# Changelog

All notable changes to this repository are recorded here. Dates use ISO 8601.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project does **not** adhere to Semantic Versioning in the strict sense
(every paper revision increments the minor version).

## [1.0.0] - 2024-09-04

### Added
- Initial open-source release accompanying the manuscript *"TEGAT: A Temporal
  Event Graph Attention Network for Transmission Risk Analysis via Heterogeneous
  Event Networks from Multi-Provincial Epidemiological Investigation Texts"*
  submitted to *Journal of Biomedical Informatics*.
- Bundled CSKEN dataset (4,812 cases / 6,194 events / 31,147 edges).
- Reference implementation of TEGAT plus six baselines (LR, SVM, MLP, RF,
  XGBoost, GraphBasedMLP).
- One-command `reproduce.py` driver covering events → graph → model →
  ablation → RQ1/RQ2/RQ3 → LaTeX tables.
- English + Chinese README and bilingual dataset card.
- Smoke-test suite (`tests/test_smoke.py`).

### Fixed
- Refactored all hard-coded absolute paths out of the source files; the
  repository is now fully relocatable.
