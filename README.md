<img src="assets/icon.svg" width="64" height="64" alt="">

# Advanced Chooser Option Pricing Model

[![Week 2 preprocessing pipeline](https://github.com/aiden-zl/advanced-chooser-option-pricing-model/actions/workflows/preprocessing.yml/badge.svg)](https://github.com/aiden-zl/advanced-chooser-option-pricing-model/actions/workflows/preprocessing.yml)

An 8-week project that rebuilds the Black-Scholes-Merton chooser option model from
Huang, Wang & Wan (2021), *Exploration of JPMorgan Chooser Option Pricing*, using
real-world data, then extends it with machine-learning volatility and pricing models.
The underlying asset is JPMorgan Chase (`JPM`), studied over 2018–2024.

## Weekly deliverables

Each week ships source code (in this repo), figures and PDF reports under `deliverables/weekN/`.

| Week | Module | Deliverables |
|------|--------|--------------|
| 1 | Data source design & initial collection | [Data specification (PDF)](deliverables/week1/data_specification.pdf) · [Lab report (PDF)](deliverables/week1/lab_report.pdf) · [figures](deliverables/week1/figures) · raw dataset in [`data/raw/`](data/raw) |
| 2 | Data preprocessing & feature engineering | [Feature documentation (PDF)](deliverables/week2/feature_documentation.pdf) · [Lab report (PDF)](deliverables/week2/lab_report.pdf) · [figures](deliverables/week2/figures) · cleaned dataset in [`data/processed/`](data/processed) (1,761 days × 22 features, CSV + Parquet) · [scheduled pipeline](.github/workflows/preprocessing.yml) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Week 2: rebuild the feature dataset from the committed raw files (offline, no keys)
python -m src.pipeline.run      # 10 validation checks, writes data/processed/
python -m pytest -q             # 10 offline tests, incl. the look-ahead test
python -m src.week2_figures     # Week 2 figures

# Week 1: re-download raw data (needs API keys; overwrites the committed raw files)
cp .env.example .env            # add free FRED and Alpha Vantage API keys
python -m src.pull_all          # test APIs, pull 2018-2024 raw data, run 23 validation checks
python -m src.pull_week2        # Week 2 raw additions (only fetches files that are missing)

# Rebuild a PDF (LaTeX; brew install tectonic)
tectonic -X compile deliverables/week2/lab_report.tex
```

Test a single connection: `python -m src.data_sources.yahoo`, `... .fred`, `... .alpha_vantage` or `... .gdelt`.

## Layout

```
src/data_sources/   API connection modules (Yahoo Finance, FRED, Alpha Vantage, GDELT) + config.py
src/pull_all.py     Week 1 pull + 23 validation checks, writes data/raw/manifest.json
src/pull_week2.py   Week 2 raw additions (warm-up, first releases, GDELT, weekly closes) + 16 checks
src/pipeline/       Week 2 cleaning (clean.py), features (features.py), runner + checks (run.py)
tests/              offline tests (no network, no keys)
data/raw/           raw source files, saved as received and never edited
data/processed/     cleaned point-in-time feature dataset, dictionary and cleaning log
deliverables/       weekly PDFs, their LaTeX sources and figures
```

API keys live only in `.env`, which is gitignored. Never commit it. The GitHub Actions
pipeline runs on the committed raw files and needs no keys.
