"""GDELT DOC 2.0 API: daily news tone and article volume for JPMorgan.

Feasibility check promised in the Week 1 data specification (section 5), verified
2026-11-25: the DOC API returns daily `timelinetone` and `timelinevolraw` series
from 2017-01-01 onward ("Invalid query start date" before that). No key is needed,
but the API allows roughly one request every 5 seconds and rate-limits bursts, so
responses are cached in data/cache/ and calls are spaced 30 s apart.

Tone is GDELT's average document tone of all matching articles that day
(positive minus negative word share, in percent; typical daily values -5 to +5).

Run directly to test the connection:  python -m src.data_sources.gdelt
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from .config import CACHE_DIR, END_DATE, RAW_DIR

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '"JPMorgan"'
GDELT_START = "2017-01-01"  # earliest date the DOC API accepts
_PACE = 30  # seconds between live calls


def _get(mode: str, start: str, end: str, use_cache: bool = True, cache_name: str | None = None) -> dict:
    cache = CACHE_DIR / f"gdelt_jpm_{cache_name or mode}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    params = {
        "query": QUERY, "mode": mode, "format": "json",
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d000000"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d235959"),
    }
    for _ in range(6):
        time.sleep(_PACE)
        text = requests.get(URL, params=params, timeout=120).text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue  # rate-limit notice or error text; retry after pacing
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"GDELT {mode}: no valid response after retries")


def _to_series(data: dict, name: str) -> pd.Series:
    rows = data["timeline"][0]["data"]
    s = pd.Series({pd.Timestamp(r["date"][:8]): float(r["value"]) for r in rows}, name=name)
    s.index.name = "date"
    return s.sort_index()


def pull(out_dir=RAW_DIR, use_cache: bool = True) -> dict[str, int]:
    tone = _to_series(_get("timelinetone", GDELT_START, END_DATE, use_cache), "tone")
    vol = _to_series(_get("timelinevolraw", GDELT_START, END_DATE, use_cache), "articles")
    df = pd.concat([tone, vol], axis=1).loc[GDELT_START:END_DATE]
    name = "gdelt_jpm_tone_daily.csv"
    df.to_csv(out_dir / name)
    return {name: len(df)}


def check_connection() -> None:
    """Live call for the last 7 days (bypasses cache)."""
    end = pd.Timestamp.today().normalize()
    data = _get("timelinetone", str((end - pd.Timedelta(days=7)).date()), str(end.date()),
                use_cache=False, cache_name="check_last7d")
    s = _to_series(data, "tone")
    print(f"[gdelt] OK  {len(s)} daily tone values, latest {s.index[-1].date()} = {s.iloc[-1]:+.2f}")


if __name__ == "__main__":
    check_connection()
