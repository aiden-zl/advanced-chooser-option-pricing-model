"""Yahoo Finance connection: daily OHLCV, dividends and splits.

Run directly to test the connection:  python -m src.data_sources.yahoo
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .config import END_DATE, PAPER_ANCHOR_DATE, PAPER_S0, RAW_DIR, START_DATE, YAHOO_TICKERS


def _exclusive_end(end: str) -> str:
    # yfinance treats `end` as exclusive, so add one day to keep END_DATE.
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_daily(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download unadjusted daily bars plus Adj Close, dividends and splits."""
    df = yf.download(
        ticker,
        start=start,
        end=_exclusive_end(end),
        auto_adjust=False,  # keep raw Close (paper's S0) AND Adj Close
        actions=True,  # include Dividends / Stock Splits columns
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def file_name(ticker: str) -> str:
    return f"yahoo_{ticker.replace('^', '').lower()}_daily.csv"


def pull(out_dir=RAW_DIR) -> dict[str, int]:
    """Download every configured ticker and save one CSV each. Returns row counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for ticker in YAHOO_TICKERS:
        df = fetch_daily(ticker)
        df.to_csv(out_dir / file_name(ticker))
        counts[file_name(ticker)] = len(df)
    return counts


def check_connection() -> None:
    """Small live test: fetch a short window and compare with the paper's S0."""
    df = fetch_daily("JPM", start="2021-08-20", end="2021-08-24")
    close = float(df.loc[PAPER_ANCHOR_DATE, "close"])
    assert abs(close - PAPER_S0) < 0.05, f"JPM close {close:.2f} != paper S0 {PAPER_S0}"
    print(f"[yahoo] OK  JPM close on {PAPER_ANCHOR_DATE} = {close:.2f} (paper: {PAPER_S0})")


if __name__ == "__main__":
    check_connection()
