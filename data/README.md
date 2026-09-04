# Dataset Card: CSKEN (COVID-19 Structured Kinematic Event Network)

## Summary

| Field | Value |
| ----- | ----- |
| **Name** | CSKEN — China COVID-19 Structured Kinematic Event Network |
| **Version** | 1.0.0 |
| **Released** | 2024 |
| **License (this package)** | MIT |
| **Source data license** | Public-domain published notices from 30 Chinese provincial Health Commissions |
| **Total cases** | 4,812 confirmed COVID-19 cases |
| **Provinces** | 30 (including 22 provinces, 5 autonomous regions, 4 municipalities, and Taiwan) |
| **Time span** | 2019-08-30 to 2020-12-31 (489 days) |
| **Total events** | 6,194 structured events |
| **Case-case edges** | 31,147 (co-location within ±3 days) |

## What is in this directory

| File | Format | Description |
| ---- | ------ | ----------- |
| `cases_metadata.json` | JSON | One entry per case with anonymized identifier, age, gender, province, dates, symptoms. |
| `events_extracted.json` | JSON | Structured events extracted from the raw trajectory text (venue visits, transport, etc.). |
| `graph_data.npz` | NumPy sparse | Adjacency tensors and node features for the heterogeneous graph (case, venue, date, transport, symptom nodes). |
| `graph_mappings.json` | JSON | Node-id ↔ entity-string mapping and the node-type dictionary. |
| `daily_counts.json` | JSON | Case counts per Chinese calendar day for the entire 489-day window. |
| `province_stats.json` | JSON | Per-province case counts, event counts, and demographic aggregates. |
| `dataset_stats.json` | JSON | High-level numbers used by the paper (top-level summary). |
| `graph_stats.json` | JSON | Graph topology summary (degree, density, connected components, ...). |
| `extraction_stats.json` | JSON | Coverage and rules statistics from the rule-based event extractor. |

## What is **not** redistributed

The original raw Chinese-language trajectory text (`**省确诊病例活动轨迹.txt`) is
publicly available from each provincial Health Commission. **For this
release**, those 30 raw text files are bundled directly under
[`data_original/`](../data_original/) so that the whole pipeline is
self-contained and can be reproduced end-to-end with no external download.

If you would rather obtain them yourself, either

1. Crawl the original notices directly (links tracked in `docs/data_sources.md`),
2. Or download the curated Zenodo bundle listed in the same file.

If you place the raw text into `data_original/` (or rely on the bundled copy)
the included `src/event_extraction.py` script will reproduce the structured
events from scratch.

## Privacy

- All case identifiers are anonymized sequential IDs (`case_0001`, `case_0002`, ...).
- Age is provided in 5-year bins; exact birth dates were dropped at extraction time.
- Venue names are normalised to a category label (e.g. "restaurant" rather than the
  exact restaurant name) for the released graph; the venue index preserved in
  `graph_mappings.json` maps each anonymised venue ID back to a coarse category
  only.

## Loading example

```python
import json, numpy as np

with open("data/cases_metadata.json", "r", encoding="utf-8") as f:
    cases = json.load(f)
print(f"Total cases: {len(cases)}")

graph = np.load("data/graph_data.npz")
print("Graph arrays:", list(graph.keys()))
```

## Known limitations

- Coverage is biased toward cases publicly reported by provincial Health Commissions;
  internal/private hospital records are not included.
- Rule-based event extraction has measurable coverage gaps for atypical location
  descriptions; see `extraction_stats.json` for the per-rule yield.
- Network edges are defined by co-location within ±3 days; this is a tunable knob
  rather than a measured incubation period.
