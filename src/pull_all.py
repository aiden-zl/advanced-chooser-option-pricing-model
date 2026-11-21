"""Week 1: test every API connection, pull the 2018-2024 raw dataset, validate it.

Usage (from the repo root):
    python -m src.pull_all              # use cached Alpha Vantage responses if present
    python -m src.pull_all --refresh    # force fresh Alpha Vantage calls (uses quota)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .data_sources import alpha_vantage, fred, yahoo
from .data_sources.config import END_DATE, PAPER_ANCHOR_DATE, PAPER_S0, RAW_DIR, START_DATE


def validate() -> dict:
    """Sanity checks on the saved raw files. Returns a dict of results."""
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    jpm = pd.read_csv(RAW_DIR / "yahoo_jpm_daily.csv", index_col="date", parse_dates=True)
    vix = pd.read_csv(RAW_DIR / "yahoo_vix_daily.csv", index_col="date", parse_dates=True)
    fd = pd.read_csv(RAW_DIR / "fred_daily.csv", index_col="date", parse_dates=True)
    fm = pd.read_csv(RAW_DIR / "fred_monthly.csv", index_col="date", parse_dates=True)
    avd = pd.read_csv(RAW_DIR / "av_jpm_dividends.csv", index_col="date", parse_dates=True)

    # Coverage: ~252 trading days/yr * 7 yrs ~ 1,760
    for label, df in (("JPM", jpm), ("VIX", vix)):
        add(f"{label} row count ~1,760 trading days", 1700 <= len(df) <= 1800, f"{len(df)} rows")
        add(
            f"{label} spans full window",
            df.index.min() <= pd.Timestamp("2018-01-03") and df.index.max() >= pd.Timestamp("2024-12-30"),
            f"{df.index.min().date()} to {df.index.max().date()}",
        )
    add("JPM and VIX share trading calendar", jpm.index.equals(vix.index), "identical date index" if jpm.index.equals(vix.index) else "dates differ")

    close = float(jpm.loc[PAPER_ANCHOR_DATE, "close"])
    add("JPM close matches paper S0 on 2021-08-23", abs(close - PAPER_S0) < 0.05, f"{close:.2f} vs {PAPER_S0}")

    add("JPM prices positive, no missing closes", (jpm["close"] > 0).all() and jpm["close"].notna().all(), f"min close {jpm['close'].min():.2f}")
    add("VIX within plausible range (5-100)", vix["close"].between(5, 100).all(), f"{vix['close'].min():.2f} to {vix['close'].max():.2f}")

    n_div = int((jpm["dividends"] > 0).sum())
    add("JPM quarterly dividends present (~28)", 26 <= n_div <= 29, f"{n_div} dividend days")
    y_div = jpm.loc[jpm["dividends"] > 0, "dividends"]
    same_dates = y_div.index.equals(avd.index)
    amt_diff = float((y_div - avd["amount"]).abs().max()) if same_dates else float("nan")
    add(
        "Alpha Vantage dividends match Yahoo (ex-dates and amounts)",
        same_dates and amt_diff < 0.005,
        f"{len(avd)} AV vs {len(y_div)} Yahoo; dates identical={same_dates}; max amount diff ${amt_diff:.3f}",
    )

    # Alpha Vantage month-end close vs Yahoo unadjusted close on the same dates
    avm = pd.read_csv(RAW_DIR / "av_jpm_monthly_adjusted.csv", index_col="date", parse_dates=True)
    common_m = avm.index.intersection(jpm.index)
    mdiff = (avm.loc[common_m, "close"] - jpm.loc[common_m, "close"]).abs()
    add(
        "Alpha Vantage month-end close matches Yahoo close",
        len(common_m) == len(avm) and float(mdiff.max()) < 0.02,
        f"{len(common_m)}/{len(avm)} month-ends found in Yahoo; max diff ${mdiff.max():.3f}",
    )

    # Alpha Vantage Treasury yields vs FRED constant-maturity series
    for av_name, sid in (("av_treasury_3month.csv", "DGS3MO"), ("av_treasury_10year.csv", "DGS10")):
        av = pd.read_csv(RAW_DIR / av_name, index_col="date", parse_dates=True)["value"].dropna()
        fr = fd[sid].dropna()
        both = av.index.intersection(fr.index)
        d = (av.loc[both] - fr.loc[both]).abs()
        share = float((d <= 0.005).mean())
        add(
            f"Alpha Vantage {sid} matches FRED on >=99% of days",
            share >= 0.99,
            f"{(d <= 0.005).sum()}/{len(both)} days within 0.005 pt; max diff {d.max():.2f}",
        )

    common = vix.index.intersection(fd["VIXCLS"].dropna().index)
    diff = (vix.loc[common, "close"] - fd.loc[common, "VIXCLS"]).abs()
    agree = float((diff <= 0.01).mean())
    add(
        "Yahoo ^VIX agrees with FRED VIXCLS on >=99.5% of days",
        agree >= 0.995,
        f"{(diff <= 0.01).sum()}/{len(common)} days within 0.01; max diff {diff.max():.2f}",
    )

    # DFF is published 7 days a week, so the wide table has weekend rows.
    # Judge each series on weekdays only; the remaining blanks are bond-market holidays.
    weekdays = fd[fd.index.day_of_week < 5]
    for sid in fd.columns:
        miss = int(weekdays[sid].isna().sum())
        add(
            f"FRED {sid} weekday blanks < 5% (market holidays)",
            miss / len(weekdays) < 0.05,
            f"{miss} of {len(weekdays)} weekdays",
        )
    add("FRED monthly series cover 84 months", len(fm) == 84, f"{len(fm)} months")

    return {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore Alpha Vantage cache")
    args = ap.parse_args()

    print("== 1. Connection tests ==")
    for mod in (yahoo, fred, alpha_vantage):
        mod.check_connection()

    print("\n== 2. Pulling raw data", START_DATE, "to", END_DATE, "==")
    counts = {}
    counts.update(yahoo.pull())
    counts.update(fred.pull())
    counts.update(alpha_vantage.pull(use_cache=not args.refresh))
    for name, n in counts.items():
        print(f"  {name:<34} {n:>6} rows")

    print("\n== 3. Validation ==")
    result = validate()
    for c in result["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"  {result['passed']}/{result['total']} checks passed")

    manifest = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": [START_DATE, END_DATE],
        "files": counts,
        "validation": result,
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
