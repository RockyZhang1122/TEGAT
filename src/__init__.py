# src/__init__.py
"""TEGAT pipeline source code.

This package makes the data processing, graph construction, modelling, and
analysis modules importable as a single Python package::

    from src.tegat_clean import TEGAT
    from src.paths import DATA_DIR, RESULTS_DIR

All modules rely on the centralised :mod:`paths` module to resolve files,
so the package is fully relocatable (no absolute paths anywhere).
"""
__version__ = "1.0.0"
