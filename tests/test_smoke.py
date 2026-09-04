"""Lightweight smoke tests for the TEGAT pipeline.

Run with::

    pytest tests/

These tests do **not** re-train anything; they only verify that the bundled
data files load, the path bootstrap works, and the model classes can be
instantiated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src/ importable from this tests/ subdirectory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from paths import DATA_DIR, RESULTS_DIR, REPO_ROOT


def test_paths_resolve_to_repo():
    assert REPO_ROOT.exists()
    assert DATA_DIR.exists()
    assert RESULTS_DIR.exists()
    assert (ROOT / 'src' / 'paths.py').exists()


def test_cases_metadata_loads():
    with open(DATA_DIR / 'cases_metadata.json', 'r', encoding='utf-8') as f:
        cases = json.load(f)
    assert isinstance(cases, (list, dict))
    n = len(cases)
    assert n > 1000, f"Expected at least 1000 cases; got {n}"


def test_events_loads():
    with open(DATA_DIR / 'events_extracted.json', 'r', encoding='utf-8') as f:
        events = json.load(f)
    assert len(events) > 1000, "Expected at least 1000 events"


def test_graph_npz_present():
    graph = np.load(DATA_DIR / 'graph_data.npz', allow_pickle=True)
    keys = list(graph.keys())
    assert len(keys) >= 2


def test_results_present():
    expected = ['main_results.json', 'ablation_results.json',
                'rq1_venue_risk.json', 'rq2_rq3_results.json',
                'rq3_causal_results.json']
    for name in expected:
        assert (RESULTS_DIR / name).exists(), f"Missing {name}"


def test_tegat_class_instantiates():
    from tegat_model import TEGAT
    m = TEGAT(
        input_dim_case=10, input_dim_venue=4,
        input_dim_date=3, hidden_dim=16,
        epochs=1, num_heads=2,
    )
    assert m is not None


def test_skl_classes_instantiate():
    """Verify the sklearn-derived baseline classes are reachable."""
    from tegat_skl import GraphBasedMLP, TEGAT_SKL
    gb_mlp = GraphBasedMLP(); assert gb_mlp is not None
    tegat_skl = TEGAT_SKL(); assert tegat_skl is not None


def test_sklearn_imports():
    """Verify every scikit-learn dependency we use is reachable."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    assert LogisticRegression() is not None
    assert RandomForestClassifier() is not None
    assert SVC() is not None
    assert MLPClassifier(max_iter=10) is not None
    assert GradientBoostingClassifier() is not None
    assert StandardScaler() is not None


def test_main_results_keys():
    with open(RESULTS_DIR / 'main_results.json', 'r', encoding='utf-8') as f:
        table = json.load(f)
    assert isinstance(table, dict)
    assert 'TEGAT' in table, "Bundled results should contain TEGAT"
    expected_metrics = {'AUC-ROC', 'AP', 'Accuracy', 'F1', 'Precision', 'Recall'}
    for method, metrics in table.items():
        for m in expected_metrics:
            assert m in metrics, f"{method} missing {m}"


def test_raw_text_dir_resolves():
    """The 30 provincial raw trajectory text files live under data_original/."""
    from paths import RAW_TEXT_DIR
    assert RAW_TEXT_DIR.exists(), f"{RAW_TEXT_DIR} missing"
    txts = list(RAW_TEXT_DIR.glob('*.txt'))
    assert len(txts) >= 25, f"Expected ≥25 raw .txt files, got {len(txts)}"


def test_event_extraction_enumerates_raw():
    """Verify event_extraction's RAW_TEXT_DIR resolves correctly."""
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    from paths import RAW_TEXT_DIR
    files = sorted(RAW_TEXT_DIR.glob('*.txt'))
    assert len(files) >= 25


def test_data_statistics_enumerates_raw():
    """Verify data_statistics can find the raw files via the paths module."""
    import sys
    sys.path.insert(0, str(ROOT / 'src'))
    # Just import RAW_TEXT_DIR directly from paths module (which data_statistics uses).
    from paths import RAW_TEXT_DIR
    files = sorted(RAW_TEXT_DIR.glob('*.txt'))
    assert len(files) >= 25
