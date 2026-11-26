"""Week 2 feature engineering. Every feature on trading day t uses only data that
was public by the close of t (see clean.py for the availability rules).

Features are computed on the full calendar (2015-07 warm-up + study window) and
then sliced to 2018-01-02 .. 2024-12-31, so no feature is blank on day one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_sources.config import END_DATE, START_DATE
from .clean import Raw, asof_strictly_before

ANN = 252

# name -> (group, formula / meaning, source, unit)
FEATURES = {
    "ret_1d":          ("Returns", "ln(S_t / S_{t-1})", "Yahoo JPM close", "log return"),
    "rv_21":           ("Volatility", "std of last 21 daily log returns x sqrt(252)", "Yahoo JPM close", "annualized, decimal"),
    "rv_63":           ("Volatility", "std of last 63 daily log returns x sqrt(252)", "Yahoo JPM close", "annualized, decimal"),
    "rv_252":          ("Volatility", "std of last 252 daily log returns x sqrt(252)", "Yahoo JPM close", "annualized, decimal"),
    "park_21":         ("Volatility", "Parkinson range estimator over 21 days: sqrt(252/(4 ln 2) x mean[ln(H/L)^2])", "Yahoo JPM high/low", "annualized, decimal"),
    "volume_ratio_21": ("Liquidity", "volume_t / mean volume of last 21 days", "Yahoo JPM volume", "ratio"),
    "div_ttm":         ("Dividends", "sum of the last 4 quarterly dividends (ex-date <= t)", "Yahoo JPM dividends", "USD per share"),
    "div_yield":       ("Dividends", "div_ttm / S_t  (the model's q)", "Yahoo JPM", "decimal"),
    "div_growth_yoy":  ("Dividends", "div_ttm / (sum of the 4 dividends before those) - 1", "Yahoo JPM dividends", "decimal"),
    "vix":             ("Market volatility", "VIX close / 100", "Yahoo ^VIX", "decimal"),
    "corr_jpm_vix_63": ("Market volatility", "63-day correlation of JPM log returns with VIX log changes", "Yahoo JPM, ^VIX", "[-1, 1]"),
    "r_3m":            ("Rates", "3-month T-bill (DTB3) / 100, previous day", "FRED", "decimal"),
    "r_6m":            ("Rates", "6-month Treasury (DGS6MO) / 100, previous day", "FRED", "decimal"),
    "r_1y":            ("Rates", "1-year Treasury (DGS1) / 100, previous day (the model's r)", "FRED", "decimal"),
    "slope_10y_3m":    ("Rates", "(DGS10 - DGS3MO) / 100, previous day", "FRED", "decimal"),
    "rate_mom_21":     ("Rates", "r_1y_t - r_1y_{t-21}: change in the 1-year rate over ~1 month", "FRED", "decimal"),
    "rate_mom_63":     ("Rates", "r_1y_t - r_1y_{t-63}: change in the 1-year rate over ~3 months", "FRED", "decimal"),
    "fed_funds":       ("Rates", "effective fed funds (DFF) / 100, previous day", "FRED", "decimal"),
    "cpi_yoy":         ("Macro", "latest published CPI / CPI 12 months earlier - 1 (first releases)", "FRED/ALFRED CPIAUCSL", "decimal"),
    "unrate":          ("Macro", "latest published unemployment rate / 100 (first release)", "FRED/ALFRED UNRATE", "decimal"),
    "sent_news":       ("Sentiment", "percentile of 5-day mean GDELT JPMorgan news tone vs all earlier days (expanding), days before t", "GDELT DOC API", "[0, 1]"),
    "sent_consumer":   ("Sentiment", "percentile of latest published consumer sentiment vs all earlier releases since 1978 (expanding)", "FRED/ALFRED UMCSENT", "[0, 1]"),
}
STATE_COLUMNS = ["close"]  # spot price S_t, carried for pricing; not counted as a feature


def _expanding_percentile(s: pd.Series) -> pd.Series:
    """Share of observations up to and including each date that are <= the current one."""
    v = s.to_numpy()
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isnan(v[i]):
            past = v[: i + 1]
            past = past[~np.isnan(past)]
            out[i] = (past <= v[i]).mean()
    return pd.Series(out, index=s.index)


def _dividend_features(jpm: pd.DataFrame) -> pd.DataFrame:
    pays = jpm.loc[jpm.dividends > 0, "dividends"]
    cal = jpm.index
    idx = pays.index.searchsorted(cal, side="right")  # number of payments with ex-date <= t
    amounts = pays.to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(amounts)])
    ttm = np.where(idx >= 4, csum[idx] - csum[np.clip(idx - 4, 0, None)], np.nan)
    prev = np.where(idx >= 8, csum[np.clip(idx - 4, 0, None)] - csum[np.clip(idx - 8, 0, None)], np.nan)
    # span of the 4 payments in use: must be < 13 months (a normal quarterly schedule)
    dates = pays.index.to_numpy()
    span_days = np.where(idx >= 4, (dates[np.clip(idx - 1, 0, None)] - dates[np.clip(idx - 4, 0, None)]) / np.timedelta64(1, "D"), np.nan)
    df = pd.DataFrame({"div_ttm": ttm, "div_prev_ttm": prev, "div_span_days": span_days}, index=cal)
    df["div_yield"] = df.div_ttm / jpm.close
    df["div_growth_yoy"] = df.div_ttm / df.div_prev_ttm - 1
    return df


def _monthly_asof(monthly: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Latest first-published value available on each trading day (published on or before t;
    CPI and jobs data are released at 8:30 ET, sentiment at 10:00 ET, i.e. before the close)."""
    out = {}
    for sid, g in monthly.groupby("series"):
        # Several months can be first published on the same day (e.g. FRED posted the
        # March and April 2018 sentiment readings together). Sort by publication date,
        # then reference month, so the latest month published by t always wins.
        g = g.sort_values(["available_from", "ref_month"], kind="stable")
        pos = g.available_from.searchsorted(calendar, side="right") - 1
        ok = pos >= 0
        p = np.clip(pos, 0, None)
        out[f"{sid}_value"] = np.where(ok, g.value.to_numpy()[p], np.nan)
        out[f"{sid}_ref"] = np.where(ok, g.ref_month.to_numpy()[p], np.datetime64("NaT", "ns"))
    return pd.DataFrame(out, index=calendar)


def build_features(raw: Raw) -> pd.DataFrame:
    jpm, vix = raw.jpm, raw.vix
    cal = jpm.index
    f = pd.DataFrame(index=cal)
    f["close"] = jpm.close

    # Returns and volatility
    r = np.log(jpm.close).diff()
    f["ret_1d"] = r
    for n in (21, 63, 252):
        f[f"rv_{n}"] = r.rolling(n).std(ddof=1) * np.sqrt(ANN)
    hl2 = np.log(jpm.high / jpm.low) ** 2
    f["park_21"] = np.sqrt(hl2.rolling(21).mean() * ANN / (4 * np.log(2)))
    f["volume_ratio_21"] = jpm.volume / jpm.volume.rolling(21).mean()

    # Dividends
    d = _dividend_features(jpm)
    f[["div_ttm", "div_yield", "div_growth_yoy"]] = d[["div_ttm", "div_yield", "div_growth_yoy"]]

    # Market volatility
    f["vix"] = vix.close.reindex(cal) / 100
    dv = np.log(vix.close.reindex(cal)).diff()
    f["corr_jpm_vix_63"] = r.rolling(63).corr(dv)

    # Rates: previous-day values (strictly before t)
    rt = asof_strictly_before(raw.rates, cal) / 100
    f["r_3m"], f["r_6m"], f["r_1y"] = rt.DTB3, rt.DGS6MO, rt.DGS1
    f["slope_10y_3m"] = rt.DGS10 - rt.DGS3MO
    f["rate_mom_21"] = f.r_1y - f.r_1y.shift(21)
    f["rate_mom_63"] = f.r_1y - f.r_1y.shift(63)
    f["fed_funds"] = rt.DFF

    # Macro (first releases, from publication date)
    m = _monthly_asof(raw.monthly, cal)
    cpi = raw.monthly[raw.monthly.series == "CPIAUCSL"].set_index("ref_month").value
    ref = pd.to_datetime(m.CPIAUCSL_ref)
    base = cpi.reindex(ref - pd.DateOffset(years=1)).to_numpy()
    f["cpi_yoy"] = m.CPIAUCSL_value / base - 1
    f["unrate"] = m.UNRATE_value / 100

    # Sentiment: news (GDELT) and consumer (UMCSENT), both scaled to [0, 1] with expanding windows
    tone5 = raw.gdelt.tone.rolling(5, min_periods=3).mean()
    pct_news = _expanding_percentile(tone5)
    f["sent_news"] = asof_strictly_before(pct_news.rename("p"), cal)["p"]

    um = raw.monthly[raw.monthly.series == "UMCSENT"].sort_values(["available_from", "ref_month"], kind="stable")
    hist = raw.umcsent_hist.to_numpy()
    avail, vals = um.available_from.to_numpy(), um.value.to_numpy()
    # rank each reading against the 1978-2014 history plus every release published on or before its date
    pct = [(np.concatenate([hist, vals[avail <= a]]) <= v).mean() for a, v in zip(avail, vals)]
    um_pct = pd.Series(pct, index=avail)
    pos = um_pct.index.searchsorted(cal, side="right") - 1
    f["sent_consumer"] = np.where(pos >= 0, um_pct.to_numpy()[np.clip(pos, 0, None)], np.nan)

    f.attrs["div_span_days_max"] = float(d.div_span_days.loc[START_DATE:END_DATE].max())
    return f.loc[START_DATE:END_DATE, STATE_COLUMNS + list(FEATURES)]
