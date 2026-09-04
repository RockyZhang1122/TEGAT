"""One-command reproduction of the TEGAT main + ablation + RQ analyses.

Usage::

    python reproduce.py                # full pipeline
    python reproduce.py --fast         # 1-seed version for CI / smoke check
    python reproduce.py --skip rq3     # skip the quasi-experimental RQ3 step

The script depends on the bundled `data/` (structured dataset) and
`data_original/` (30 provincial raw trajectory text files) subdirectories of
this repository. Both are included in the GitHub release; no external download
is required.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src/` importable when this file is run from anywhere.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

# Import lazily so users without sklearn can still get a helpful error.
from paths import REPO_ROOT  # noqa: E402


def _banner(title: str) -> None:
    print('\n' + '=' * 72)
    print(f'  {title}')
    print('=' * 72)


def step_event_extraction() -> None:
    """Skipped when events_extracted.json already exists in data/."""
    target = REPO_ROOT / 'data' / 'events_extracted.json'
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f'[skip] {target.name} already present.')
        return
    print('Running rule-based event extraction (uses data_original/) ...')
    import event_extraction  # noqa: F401
    event_extraction.main()


def step_graph_build() -> None:
    target = REPO_ROOT / 'data' / 'graph_data.npz'
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f'[skip] {target.name} already present.')
        return
    print('Building heterogeneous graph ...')
    import graph_builder  # noqa: F401
    graph_builder.main()


def step_data_stats() -> None:
    print('Computing dataset statistics ...')
    import data_statistics  # noqa: F401
    data_statistics.main()


def step_main_experiment() -> None:
    _banner('Step 4 / 7  -  Main experiment (TEGAT vs baselines)')
    import tegat_clean  # noqa: F401
    tegat_clean.main()


def step_ablation() -> None:
    _banner('Step 5 / 7  -  Component ablation')
    import run_ablation  # noqa: F401
    run_ablation.main()


def step_rq1() -> None:
    _banner('Step 6a / 7  -  RQ1 venue risk')
    import rq1_venue_risk  # noqa: F401
    rq1_venue_risk.main()


def step_rq2() -> None:
    print('  RQ2 & RQ3 analysis ...')
    import rq2_rq3_analysis  # noqa: F401
    rq2_rq3_analysis.main()


def step_rq3_causal() -> None:
    _banner('Step 6c / 7  -  RQ3 causal (lockdown)')
    import rq3_causal_analysis  # noqa: F401
    rq3_causal_analysis.main()


def step_tables() -> None:
    _banner('Step 7 / 7  -  LaTeX table generation')
    import generate_tables           # noqa: F401
    generate_tables.main()
    import generate_ablation_tables  # noqa: F401
    generate_ablation_tables.main()


def main() -> None:
    parser = argparse.ArgumentParser(description='TEGAT reproduction driver.')
    parser.add_argument('--fast', action='store_true',
                        help='Use a single random seed (for CI / quick checks).')
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['extract', 'graph', 'stats',
                                 'main', 'ablation', 'rq1', 'rq2', 'rq3',
                                 'tables'],
                        help='Skip one or more steps.')
    args = parser.parse_args()

    skip = set(args.skip)
    t0 = time.time()

    steps = [
        ('extract',  step_event_extraction),
        ('graph',    step_graph_build),
        ('stats',    step_data_stats),
        ('main',     step_main_experiment),
        ('ablation', step_ablation),
        ('rq1',      step_rq1),
        ('rq2',      step_rq2),
        ('rq3',      step_rq3_causal),
        ('tables',   step_tables),
    ]

    for name, fn in steps:
        if name in skip:
            print(f'[skip] step: {name}')
            continue
        fn()

    elapsed = time.time() - t0
    print('\n' + '#' * 72)
    print(f'# Reproduction complete in {elapsed/60:.1f} minutes.')
    print('# Results are under results/.  LaTeX tables are ready for the paper.')
    print('#' * 72)


if __name__ == '__main__':
    main()
