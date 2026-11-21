<img src="assets/icon.svg" width="64" height="64" alt="">

# Advanced Chooser Option Pricing Model

An 8-week project that rebuilds the Black-Scholes-Merton chooser option model from
Huang, Wang & Wan (2021), *Exploration of JPMorgan Chooser Option Pricing*, using
real-world data, then extends it with machine-learning volatility and pricing models.
The underlying asset is JPMorgan Chase (`JPM`), studied over 2018–2024.

## Weekly deliverables

Each week ships source code (in this repo), figures and a PDF lab report under `deliverables/weekN/`.

| Week | Module | Deliverables |
|------|--------|--------------|
| 1 | Data source design & initial collection | [Data specification (PDF)](deliverables/week1/data_specification.pdf) · [Lab report (PDF)](deliverables/week1/lab_report.pdf) · [figures](deliverables/week1/figures) · raw dataset in [`data/raw/`](data/raw) |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add free FRED and Alpha Vantage API keys
python -m src.pull_all          # test APIs, pull 2018-2024 raw data, run 20 validation checks
python -m src.week1_figures     # Week 1 figures and summary stats
python -m src.build_pdf deliverables/week1/lab_report.html   # rebuild a PDF (needs Google Chrome)
```

Test a single connection: `python -m src.data_sources.yahoo`, `... .fred` or `... .alpha_vantage`.

## Layout

```
src/data_sources/   API connection modules (Yahoo Finance, FRED, Alpha Vantage) + config.py
src/pull_all.py     one-command pull + validation, writes data/raw/manifest.json
data/raw/           raw source files, saved as received (no cleaning)
deliverables/       weekly PDFs, HTML sources and figures
```

API keys live only in `.env`, which is gitignored. Never commit it.
