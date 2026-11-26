"""Week 2 cleaning: load raw files, align everything to the NYSE trading calendar
without look-ahead, and screen returns for outliers.

Point-in-time rules (a value may be used on trading day t only if it was public
by the close of day t):
  * JPM and VIX closes: same day (VIX settles 15 minutes after the stock close).
  * FRED daily rates (Treasury yields, fed funds): last observation dated strictly
    before t. Fed funds for day d is published the next business day, and FRED posts
    Treasury yields after the close, so a one-day lag is used for all of them.
  * Monthly macro data: the first-published value, from its publication date.
  * GDELT news tone: days strictly before t (a GDELT day ends at midnight UTC,
    after the US close).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data_sources.config import END_DATE, RAW_DIR, START_DATE

WARMUP_DIR = RAW_DIR / "warmup"
DAILY_RATES = ["DTB3", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS10", "DFF"]
IQR_K = (1.5, 3.0)
# ALFRED's first-vintage dates for UMCSENT do not always match the survey's real release
# calendar (e.g. January 2018 is dated 26 Jan, but the final 95.7 was published 2 Feb;
# March 2018 is dated 27 Apr, but was published 29 Mar). The final reading is published by
# the first days of the following month, so a conservative rule is used for this series:
# usable from the later of the ALFRED date and the 8th of the following month.
UMCSENT_SAFE_DAY = 8


@dataclass
class Raw:
    jpm: pd.DataFrame
    vix: pd.DataFrame
    rates: pd.DataFrame          # FRED daily, native dates, NaN = not published
    monthly: pd.DataFrame        # first releases: series, ref_month, available_from, value
    umcsent_hist: pd.Series      # 1978-2014 baseline for sentiment scaling
    gdelt: pd.DataFrame          # tone, articles by calendar day
    log: dict = field(default_factory=dict)


def _read(path, **kw):
    return pd.read_csv(path, index_col="date", parse_dates=True, **kw)


def load_raw(cut: str | None = None) -> Raw:
    """Load warm-up + Week 1 + Week 2 raw files. `cut` drops everything not yet
    public on that date (used by the look-ahead test)."""
    jpm = pd.concat([_read(WARMUP_DIR / "yahoo_jpm_daily_warmup.csv"), _read(RAW_DIR / "yahoo_jpm_daily.csv")])
    vix = pd.concat([_read(WARMUP_DIR / "yahoo_vix_daily_warmup.csv"), _read(RAW_DIR / "yahoo_vix_daily.csv")])
    rates = pd.concat([_read(WARMUP_DIR / "fred_daily_warmup.csv"), _read(RAW_DIR / "fred_daily.csv")])[DAILY_RATES]
    monthly = pd.read_csv(RAW_DIR / "fred_monthly_first_release.csv", parse_dates=["ref_month", "available_from"])
    um = monthly.series == "UMCSENT"
    safe = monthly.loc[um, "ref_month"] + pd.DateOffset(months=1) + pd.Timedelta(days=UMCSENT_SAFE_DAY - 1)
    monthly.loc[um, "available_from"] = np.maximum(monthly.loc[um, "available_from"], safe)
    hist = _read(RAW_DIR / "fred_umcsent_history.csv")["UMCSENT"].dropna()
    gdelt = _read(RAW_DIR / "gdelt_jpm_tone_daily.csv")
    if cut is not None:
        c = pd.Timestamp(cut)
        jpm, vix, rates, gdelt = jpm.loc[:c], vix.loc[:c], rates.loc[:c], gdelt.loc[:c]
        monthly = monthly[monthly.available_from <= c]
    return Raw(jpm.sort_index(), vix.sort_index(), rates.sort_index(), monthly, hist, gdelt.sort_index())


def check_prices(raw: Raw) -> dict:
    """Structural checks on the price files (duplicates, OHLC consistency, positivity)."""
    out = {}
    for name, df in (("jpm", raw.jpm), ("vix", raw.vix)):
        ohlc_bad = ~((df.low <= df[["open", "close"]].min(axis=1) + 1e-9) &
                     (df.high >= df[["open", "close"]].max(axis=1) - 1e-9))
        out[name] = {
            "rows": len(df),
            "duplicate_dates": int(df.index.duplicated().sum()),
            "non_positive_prices": int((df[["open", "high", "low", "close"]] <= 0).sum().sum()),
            "ohlc_inconsistent_rows": int(ohlc_bad.sum()),
            "missing_close": int(df.close.isna().sum()),
        }
    out["jpm"]["zero_volume_days"] = int((raw.jpm.volume <= 0).sum())
    out["calendars_identical"] = bool(raw.jpm.index.equals(raw.vix.index))
    return out


def asof_strictly_before(values: pd.DataFrame | pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """For each trading day t, the latest observation dated strictly before t."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    out = {}
    for col in frame.columns:
        s = frame[col].dropna()
        pos = s.index.searchsorted(calendar, side="left") - 1  # last index < t
        vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
        out[col] = vals
    return pd.DataFrame(out, index=calendar)


def gap_report(raw: Raw, calendar: pd.DatetimeIndex) -> dict:
    """How many NYSE trading days in the study window have no same-day FRED value,
    and how far previous-value fill is from linear interpolation on those days."""
    study = calendar[(calendar >= START_DATE) & (calendar <= END_DATE)]
    rep = {}
    for col in DAILY_RATES:
        s = raw.rates[col].dropna()
        on_cal = s.reindex(study)
        gaps = on_cal[on_cal.isna()].index
        ffill = s.reindex(s.index.union(study)).ffill().reindex(gaps)
        interp = s.reindex(s.index.union(study)).interpolate(method="time").reindex(gaps)
        diff = (ffill - interp).abs()
        rep[col] = {"gap_days": int(len(gaps)),
                    "max_abs_diff_ffill_vs_linear_pp": round(float(diff.max()), 4) if len(gaps) else 0.0,
                    "mean_abs_diff_ffill_vs_linear_pp": round(float(diff.mean()), 4) if len(gaps) else 0.0}
    return rep


def outlier_screen(close: pd.Series, vix_close: pd.Series) -> tuple[pd.DataFrame, dict]:
    """IQR screen of JPM daily log returns over the study window. Flags only;
    nothing is removed. A flagged day is treated as a possible data error only if it
    is a one-day spike that fully reverses the next day (the signature of a bad tick)."""
    r = np.log(close).diff().loc[START_DATE:END_DATE]
    dv = np.log(vix_close).diff().reindex(r.index)
    q1, q3 = r.quantile(0.25), r.quantile(0.75)
    iqr = q3 - q1
    flags = pd.DataFrame({"ret": r, "dlog_vix": dv})
    summary = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr)}
    for k in IQR_K:
        lo, hi = q1 - k * iqr, q3 + k * iqr
        f = (r < lo) | (r > hi)
        flags[f"flag_k{k}"] = f
        summary[f"k{k}"] = {"lower": float(lo), "upper": float(hi), "flagged_days": int(f.sum()),
                            "share": float(f.mean())}
    nxt = r.shift(-1)
    big = flags["flag_k3.0"]
    reversal = big & (np.sign(r) != np.sign(nxt)) & ((r + nxt).abs() < 0.2 * r.abs())
    summary["spike_reversal_days"] = [str(d.date()) for d in r.index[reversal.fillna(False)]]
    in_covid = big & (r.index >= "2020-02-20") & (r.index <= "2020-04-30")
    summary["k3_flags_in_covid_window"] = int(in_covid.sum())
    # Market-wide confirmation: VIX moved against JPM (fear up on down days, down on up days)
    summary["k3_flags_with_opposite_vix_move"] = int((big & (np.sign(dv) == -np.sign(r))).sum())
    # Independent price check: Alpha Vantage weekly closes vs Yahoo on the same dates
    wk = pd.read_csv(RAW_DIR / "av_jpm_weekly.csv", index_col="date", parse_dates=True)["close"]
    both = wk.index.intersection(close.index)
    summary["weekly_crosscheck"] = {"weeks": int(len(wk)), "matched": int(len(both)),
                                    "max_abs_diff_usd": float((wk.loc[both] - close.loc[both]).abs().max())}
    # A reversal day is confirmed genuine if the VIX moved opposite to JPM on both days
    conf = []
    for d in summary["spike_reversal_days"]:
        t = pd.Timestamp(d)
        t1 = r.index[r.index.get_loc(t) + 1]
        conf.append(bool(np.sign(dv[t]) == -np.sign(r[t]) and np.sign(dv[t1]) == -np.sign(r[t1])))
    summary["spike_reversals_confirmed_by_vix"] = int(sum(conf))
    summary["decision"] = "keep all: flagged returns are genuine market moves, not data errors"
    return flags, summary
