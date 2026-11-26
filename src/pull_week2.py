"""Week 2 raw-data additions. Writes NEW files only; Week 1 raw files are never touched.

1. Warm-up history (2015-07-01 to 2017-12-31) for JPM, VIX and the FRED daily series,
   so that 252-day volatility, year-on-year dividend growth and CPI growth have no
   blanks from the first study day (2018-01-02).
2. First-release (point-in-time) values and release dates for CPI, unemployment and
   consumer sentiment from ALFRED, so monthly data enter the dataset only on the day
   they were actually published.
3. Consumer-sentiment history 1978-2014 (current vintage) as the baseline for scaling
   the sentiment score to [0, 1] with an expanding window.
4. GDELT daily JPMorgan news tone and article volume, 2017-2024.

Usage:  python -m src.pull_week2             # fetch only files that don't exist yet
        python -m src.pull_week2 --refresh   # re-download everything (overwrites raw files)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .data_sources import alpha_vantage, fred, gdelt, yahoo
from .data_sources.config import END_DATE, FRED_SERIES, RAW_DIR, START_DATE, YAHOO_TICKERS

WARMUP_START = "2015-07-01"
WARMUP_END = "2017-12-31"
WARMUP_DIR = RAW_DIR / "warmup"
MONTHLY_FIRST_RELEASE = ["CPIAUCSL", "UNRATE", "UMCSENT"]
UMCSENT_HISTORY = ("1978-01-01", "2014-12-01")  # monthly survey from 1978


def pull_warmup() -> dict[str, int]:
    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for ticker in YAHOO_TICKERS:
        df = yahoo.fetch_daily(ticker, start=WARMUP_START, end=WARMUP_END)
        name = yahoo.file_name(ticker).replace("_daily.csv", "_daily_warmup.csv")
        df.to_csv(WARMUP_DIR / name)
        counts[f"warmup/{name}"] = len(df)
    daily = [sid for sid, (_, freq) in FRED_SERIES.items() if freq == "daily"]
    df = pd.concat([fred.fetch_series(s, WARMUP_START, WARMUP_END) for s in daily], axis=1, sort=True)
    df.to_csv(WARMUP_DIR / "fred_daily_warmup.csv")
    counts["warmup/fred_daily_warmup.csv"] = len(df)
    return counts


def pull_first_releases() -> dict[str, int]:
    """Long table: series, reference month, date first published, first-published value."""
    rows = []
    client = fred.client()
    for sid in MONTHLY_FIRST_RELEASE:
        a = client.get_series_all_releases(sid, realtime_start="2015-01-01")
        a["date"] = pd.to_datetime(a["date"])
        a["realtime_start"] = pd.to_datetime(a["realtime_start"])
        a = a[(a["date"] >= "2015-01-01") & (a["date"] <= END_DATE)]
        first = a.sort_values("realtime_start").groupby("date", as_index=False).first()
        first["series"] = sid
        rows.append(first[["series", "date", "realtime_start", "value"]])
    out = pd.concat(rows).rename(columns={"date": "ref_month", "realtime_start": "available_from"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out.to_csv(RAW_DIR / "fred_monthly_first_release.csv", index=False)

    hist = fred.fetch_series("UMCSENT", *UMCSENT_HISTORY)
    hist.to_frame().to_csv(RAW_DIR / "fred_umcsent_history.csv")
    return {"fred_monthly_first_release.csv": len(out), "fred_umcsent_history.csv": len(hist)}


def pull_av_weekly() -> dict[str, int]:
    """Alpha Vantage weekly JPM closes (free, full history): independent check of the
    Yahoo daily closes, used to confirm that extreme returns are not data errors."""
    data = alpha_vantage.query("TIME_SERIES_WEEKLY", {"symbol": "JPM"}, cache_name="jpm_weekly")
    df = pd.DataFrame.from_dict(data["Weekly Time Series"], orient="index").astype(float)
    df.columns = [c.split(". ", 1)[1] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index().loc[START_DATE:END_DATE]
    df.to_csv(RAW_DIR / "av_jpm_weekly.csv")
    return {"av_jpm_weekly.csv": len(df)}


def validate() -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for t in ("jpm", "vix"):
        w = pd.read_csv(WARMUP_DIR / f"yahoo_{t}_daily_warmup.csv", index_col="date", parse_dates=True)
        main = pd.read_csv(RAW_DIR / f"yahoo_{t}_daily.csv", index_col="date", parse_dates=True)
        gap_days = (main.index[0] - w.index[-1]).days
        jump = abs(main["close"].iloc[0] / w["close"].iloc[-1] - 1)
        add(f"{t.upper()} warm-up joins Week 1 file without gap or overlap",
            w.index[-1] < main.index[0] and gap_days <= 5,
            f"warm-up ends {w.index[-1].date()}, Week 1 starts {main.index[0].date()} ({gap_days} days)")
        if t == "jpm":
            add("JPM close continuous across the join (<10% move)", jump < 0.10, f"{jump:.2%}")
        add(f"{t.upper()} warm-up has ~630 trading days", 600 <= len(w) <= 650, f"{len(w)} rows")

    wv = pd.read_csv(WARMUP_DIR / "yahoo_vix_daily_warmup.csv", index_col="date", parse_dates=True)["close"]
    wf = pd.read_csv(WARMUP_DIR / "fred_daily_warmup.csv", index_col="date", parse_dates=True)["VIXCLS"].dropna()
    both = wv.index.intersection(wf.index)
    agree = float(((wv.loc[both] - wf.loc[both]).abs() <= 0.01).mean())
    add("Warm-up Yahoo VIX agrees with FRED VIXCLS on >=99.5% of days", agree >= 0.995,
        f"{agree:.1%} of {len(both)} days")

    fr = pd.read_csv(RAW_DIR / "fred_monthly_first_release.csv", parse_dates=["ref_month", "available_from"])
    for sid in MONTHLY_FIRST_RELEASE:
        s = fr[fr.series == sid]
        lag = (s.available_from - s.ref_month).dt.days
        add(f"{sid} first releases cover every month 2015-2024 (120)", len(s) == 120 and s.value.notna().all(),
            f"{len(s)} months; publication lag {lag.min()}-{lag.max()} days after month start")
        add(f"{sid} never published before its reference month", (lag > 0).all(), f"min lag {lag.min()} days")

    wk = pd.read_csv(RAW_DIR / "av_jpm_weekly.csv", index_col="date", parse_dates=True)["close"]
    yj = pd.read_csv(RAW_DIR / "yahoo_jpm_daily.csv", index_col="date", parse_dates=True)["close"]
    both = wk.index.intersection(yj.index)
    d = (wk.loc[both] - yj.loc[both]).abs()
    add("Alpha Vantage weekly closes match Yahoo daily closes (all weeks)",
        len(both) == len(wk) and d.max() < 0.01, f"{len(both)}/{len(wk)} weeks; max diff ${d.max():.5f}")

    g = pd.read_csv(RAW_DIR / "gdelt_jpm_tone_daily.csv", index_col="date", parse_dates=True)
    full = pd.date_range("2017-01-01", END_DATE, freq="D")
    missing = full.difference(g.index)
    add("GDELT covers >=99.8% of calendar days 2017-2024 (source outages listed)",
        len(g) / len(full) >= 0.998 and g.index.isin(full).all(),
        f"{len(g)} of {len(full)} days; missing: {', '.join(str(d.date()) for d in missing) or 'none'}")
    add("GDELT has at least 1 article every day", (g["articles"] >= 1).all(), f"min {int(g['articles'].min())} articles/day")
    add("GDELT tone within plausible range (-20, +20)", g["tone"].between(-20, 20).all(),
        f"{g['tone'].min():.2f} to {g['tone'].max():.2f}")
    return {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="overwrite existing raw files")
    args = ap.parse_args()
    # Raw files are immutable once committed: Yahoo silently revises adjusted closes,
    # so re-downloading would rewrite history. Only missing files are fetched by default.
    steps = [
        (pull_warmup, [WARMUP_DIR / "yahoo_jpm_daily_warmup.csv", WARMUP_DIR / "yahoo_vix_daily_warmup.csv",
                       WARMUP_DIR / "fred_daily_warmup.csv"]),
        (pull_first_releases, [RAW_DIR / "fred_monthly_first_release.csv", RAW_DIR / "fred_umcsent_history.csv"]),
        (gdelt.pull, [RAW_DIR / "gdelt_jpm_tone_daily.csv"]),
        (pull_av_weekly, [RAW_DIR / "av_jpm_weekly.csv"]),
    ]
    counts = {}
    print("== Week 2 raw additions ==")
    for fn, outputs in steps:
        if args.refresh or not all(p.exists() for p in outputs):
            counts.update(fn())
        else:
            for p in outputs:
                counts[str(p.relative_to(RAW_DIR))] = sum(1 for _ in open(p)) - 1
            print(f"  (kept existing) {', '.join(p.name for p in outputs)}")
    for k, v in counts.items():
        print(f"  {k:<42} {v:>6} rows")
    res = validate()
    for c in res["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"  {res['passed']}/{res['total']} checks passed")
    manifest = {"pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warmup_window": [WARMUP_START, WARMUP_END], "study_window": [START_DATE, END_DATE],
                "files": counts, "validation": res}
    (RAW_DIR / "manifest_week2.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
