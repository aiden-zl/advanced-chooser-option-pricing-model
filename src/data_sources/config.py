"""Shared configuration for Week 1 data collection.

Everything that defines *what* we collect lives here, so the data
specification document and the code stay in sync.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"  # gitignored: raw API responses

load_dotenv(ROOT / ".env")

# --- Study window ----------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"  # inclusive

# --- Yahoo Finance (no key) -------------------------------------------------
YAHOO_TICKERS = {
    "JPM": "JPMorgan Chase & Co. common stock (underlying asset)",
    "^VIX": "Cboe Volatility Index (market implied volatility)",
}

# --- FRED (key required) ----------------------------------------------------
# series_id: (description, native frequency)
FRED_SERIES = {
    "DTB3": ("3-Month Treasury Bill secondary market rate, %", "daily"),
    "DGS3MO": ("3-Month Treasury constant maturity yield, %", "daily"),
    "DGS6MO": ("6-Month Treasury constant maturity yield, % (matches T1 = 0.5y)", "daily"),
    "DGS1": ("1-Year Treasury constant maturity yield, % (matches T2 = 1y)", "daily"),
    "DGS2": ("2-Year Treasury constant maturity yield, %", "daily"),
    "DGS10": ("10-Year Treasury constant maturity yield, %", "daily"),
    "DFF": ("Effective Federal Funds Rate, %", "daily"),
    "VIXCLS": ("Cboe VIX close (cross-check for Yahoo ^VIX)", "daily"),
    "CPIAUCSL": ("CPI, all urban consumers, index 1982-84=100, SA", "monthly"),
    "UNRATE": ("Unemployment rate, %, SA", "monthly"),
}

# --- Alpha Vantage (key required, free tier: 25 calls/day, ~1 call/sec) ----
# Full daily history (outputsize=full) and NEWS_SENTIMENT are premium-only
# on the free key (verified 2026-11-20), so we use the free endpoints below.
ALPHA_VANTAGE_CALLS = {
    "jpm_daily_compact": ("TIME_SERIES_DAILY", {"symbol": "JPM", "outputsize": "compact"}),
    "jpm_monthly_adjusted": ("TIME_SERIES_MONTHLY_ADJUSTED", {"symbol": "JPM"}),
    "jpm_dividends": ("DIVIDENDS", {"symbol": "JPM"}),
    "treasury_3month": ("TREASURY_YIELD", {"interval": "daily", "maturity": "3month"}),
    "treasury_10year": ("TREASURY_YIELD", {"interval": "daily", "maturity": "10year"}),
}

# --- Reference values from Huang, Wang & Wan (2021), Table 2 ---------------
PAPER_ANCHOR_DATE = "2021-08-23"
PAPER_S0 = 156.70
PAPER_PARAMS = {
    "S0": 156.70,
    "risk_free_rate": 0.0015,
    "sigma": 0.282,
    "strike": 150.0,
    "dividend_yield": 0.0233,
    "T1": 0.5,
    "T2": 1.0,
}


def require_key(name: str) -> str:
    """Return an API key from the environment or fail with a clear message."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and add your key."
        )
    return value
