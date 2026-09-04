"""Repository-wide path configuration.

All scripts import REPO_ROOT, DATA_DIR, RESULTS_DIR, PAPER_DIR from this module so
that the package can be cloned into any directory without further edits.

Layout::

    TEGAT_release/
    ├── data/            <- processed / structured dataset (cases, events, graph, aggregates)
    ├── data_original/   <- raw Chinese EPI trajectory text (30 provincial files)
    ├── results/         <- JSON / Markdown / LaTeX tables
    ├── paper/           <- LaTeX source, references, figures
    ├── docs/            <- experiment design & data source notes
    ├── scripts/         <- one-off verification / decoding utilities
    ├── src/             <- importable source package (this file lives here)
    └── tests/           <- smoke tests
"""
from pathlib import Path
import sys

# Repository root: this file lives in src/, so the root is the parent of its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Standard subdirectories
DATA_DIR = REPO_ROOT / 'data'
RAW_TEXT_DIR = REPO_ROOT / 'data_original'
RESULTS_DIR = REPO_ROOT / 'results'
PAPER_DIR = REPO_ROOT / 'paper'
SRC_DIR = REPO_ROOT / 'src'
DOCS_DIR = REPO_ROOT / 'docs'
SCRIPTS_DIR = REPO_ROOT / 'scripts'

# Ensure result/data subdirectories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def data_path(name: str) -> Path:
    """Resolve a file under data/."""
    return DATA_DIR / name


def raw_text_path(name: str) -> Path:
    """Resolve a file under data_original/ (raw Chinese EPI trajectory text)."""
    return RAW_TEXT_DIR / name


def result_path(name: str) -> Path:
    """Resolve a file under results/."""
    return RESULTS_DIR / name


def paper_path(name: str) -> Path:
    """Resolve a file under paper/."""
    return PAPER_DIR / name


def figure_path(name: str) -> Path:
    """Resolve a figure under paper/figures/."""
    return PAPER_DIR / 'figures' / name


if __name__ == '__main__':
    # Quick sanity check.
    print(f'REPO_ROOT     = {REPO_ROOT}')
    print(f'DATA_DIR      = {DATA_DIR}  ({len(list(DATA_DIR.iterdir()))} entries)')
    print(f'RAW_TEXT_DIR  = {RAW_TEXT_DIR}  ({len(list(RAW_TEXT_DIR.glob("*.txt"))) if RAW_TEXT_DIR.is_dir() else 0} .txt files)')
    print(f'RESULTS_DIR   = {RESULTS_DIR}  ({len(list(RESULTS_DIR.iterdir()))} entries)')
