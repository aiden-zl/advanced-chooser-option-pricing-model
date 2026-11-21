"""Alpha Vantage connection (free tier).

Free-tier limits (verified 2026-11-20 with a free key):
  * ~25 requests/day and at most ~1 request/second.
  * TIME_SERIES_DAILY with outputsize=full is PREMIUM -> only the last 100
    days ("compact") are free, so Yahoo is our primary daily price source.
  * NEWS_SENTIMENT is PREMIUM.
  * Errors come back as HTTP 200 with an "Information"/"Note"/"Error Message"
    field, so we check for the expected payload, not the status code.

Responses are cached in data/cache/ so re-runs don't burn the daily quota.

Run directly to test the connection:  python -m src.data_sources.alpha_vantage
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from .config import ALPHA_VANTAGE_CALLS, CACHE_DIR, END_DATE, RAW_DIR, START_DATE, require_key

URL = "https://www.alphavantage.co/query"
_MIN_INTERVAL = 1.5  # seconds between live calls
_last_call = 0.0


class AlphaVantageError(RuntimeError):
    pass


def query(function: str, params: dict, cache_name: str | None = None, use_cache: bool = True) -> dict:
    """Call one endpoint, with disk caching and free-tier rate limiting."""
    global _last_call
    cache = CACHE_DIR / f"av_{cache_name or function}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    wait = _MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(
        URL,
        params={"function": function, "apikey": require_key("ALPHAVANTAGE_API_KEY"), **params},
        timeout=60,
    )
    _last_call = time.time()
    resp.raise_for_status()
    data = resp.json()

    msg = data.get("Information") or data.get("Note") or data.get("Error Message")
    payload_keys = [k for k in data if k != "Meta Data"]
    if msg or not payload_keys:
        raise AlphaVantageError(f"{function}: {msg or 'empty response'}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def _time_series_to_df(data: dict) -> pd.DataFrame:
    key = next(k for k in data if "Time Series" in k)
    df = pd.DataFrame.from_dict(data[key], orient="index").astype(float)
    df.columns = [c.split(". ", 1)[1].replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def _records_to_df(data: dict, date_col: str) -> pd.DataFrame:
    df = pd.DataFrame(data["data"])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in ("value", "amount") else df[c]
    df.index.name = "date"
    return df


def to_frame(name: str, data: dict) -> pd.DataFrame:
    function = ALPHA_VANTAGE_CALLS[name][0]
    if function.startswith("TIME_SERIES"):
        df = _time_series_to_df(data)
    elif function == "DIVIDENDS":
        df = _records_to_df(data, "ex_dividend_date")
    else:  # TREASURY_YIELD etc.
        df = _records_to_df(data, "date")
    if name != "jpm_daily_compact":  # compact = most recent 100 days, keep as-is
        df = df.loc[START_DATE:END_DATE]
    return df


def pull(out_dir=RAW_DIR, use_cache: bool = True) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (function, params) in ALPHA_VANTAGE_CALLS.items():
        df = to_frame(name, query(function, params, cache_name=name, use_cache=use_cache))
        fname = f"av_{name}.csv"
        df.to_csv(out_dir / fname)
        counts[fname] = len(df)
    return counts


def check_connection() -> None:
    """Live call (bypasses cache) to the smallest free endpoint."""
    data = query("GLOBAL_QUOTE", {"symbol": "JPM"}, cache_name="check_global_quote", use_cache=False)
    q = data["Global Quote"]
    print(f"[alpha] OK  JPM latest trading day {q['07. latest trading day']}, price {float(q['05. price']):.2f}")


if __name__ == "__main__":
    check_connection()
