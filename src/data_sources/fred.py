"""FRED connection: Treasury rates, Fed funds, VIX cross-check, macro series.

Run directly to test the connection:  python -m src.data_sources.fred
"""

from __future__ import annotations

import pandas as pd
from fredapi import Fred

from .config import END_DATE, FRED_SERIES, PAPER_ANCHOR_DATE, RAW_DIR, START_DATE, require_key


def client() -> Fred:
    return Fred(api_key=require_key("FRED_API_KEY"))


def fetch_series(series_id: str, start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    s = client().get_series(series_id, observation_start=start, observation_end=end)
    if s.empty:
        raise RuntimeError(f"FRED returned no data for {series_id}")
    s.index.name = "date"
    s.name = series_id
    return s


def pull(out_dir=RAW_DIR) -> dict[str, int]:
    """Save daily series to one wide CSV and monthly series to another.

    Missing values (market holidays) are left as blanks on purpose:
    this is the *raw* dataset, cleaning happens in Week 2.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_freq: dict[str, list[pd.Series]] = {"daily": [], "monthly": []}
    for sid, (_, freq) in FRED_SERIES.items():
        by_freq[freq].append(fetch_series(sid))
    counts = {}
    for freq, series in by_freq.items():
        df = pd.concat(series, axis=1).sort_index()
        name = f"fred_{freq}.csv"
        df.to_csv(out_dir / name)
        counts[name] = len(df)
    return counts


def check_connection() -> None:
    s = fetch_series("DGS1", start="2021-08-16", end="2021-08-27")
    print(f"[fred]  OK  DGS1 on {PAPER_ANCHOR_DATE} = {s.loc[PAPER_ANCHOR_DATE]:.2f}%  ({len(s)} obs)")


if __name__ == "__main__":
    check_connection()
