# Data sources

## Raw EPI trajectory text

The original **省确诊病例活动轨迹** (provincial confirmed-case activity trajectory) notices
were retrieved from each provincial Health Commission's official website between
January 2020 and December 2021. The script intentionally relies on public-domain
notices; no paywalled or internal sources were used.

| Province | URL |
| -------- | --- |
| Anhui (安徽)   | http://wjw.ah.gov.cn/ |
| Beijing (北京) | http://wjw.beijing.gov.cn/ |
| Chongqing (重庆) | http://wsjkw.cq.gov.cn/ |
| Fujian (福建)   | http://wjw.fujian.gov.cn/ |
| Gansu (甘肃)   | http://wsjk.gansu.gov.cn/ |
| Guangdong (广东) | http://wsjkw.gd.gov.cn/ |
| ... | (29 more) |

A full mapping is stored in `docs/source_urls.csv` after release. URLs are
re-checked monthly; broken links are documented in `docs/data_sources_changelog.md`.

## Bundled derivative products

The repository ships **structured derivatives** of the raw text, not the raw
text itself:

- `cases_metadata.json`
- `events_extracted.json`
- `graph_data.npz` + `graph_mappings.json`
- per-day / per-province aggregates

This keeps the public repository under 100 MB and avoids re-publishing documents
that were originally produced by provincial authorities.

## Curated Zenodo bundle

A signed Zenodo archive containing the raw trajectory text files and the
exact pipeline used to produce the derivatives is available at:

> **DOI: 10.5281/zenodo.placeholder** *(final DOI will be issued upon acceptance)*

After acceptance, the bundle will be deposited under the DOI above. To rebuild
the structured derivatives from scratch:

```bash
git clone https://github.com/RockyZhang1122/TEGAT.git
cd TEGAT
# 1. place the raw text files under ./data_original/  (already done in this release)
ls data_original/   # should list 30 *.txt files

# 2. run the extractor
python src/event_extraction.py
```

The output of step 2 will be byte-identical to the bundled `data/events_extracted.json`
(seed = 42, identical rules, identical input files).

The current release ships the raw text files inside `data_original/` directly
so the whole pipeline is fully self-contained: no external download is needed
to reproduce any result.
