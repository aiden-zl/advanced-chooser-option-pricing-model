"""Week 1: exploratory figures and summary statistics for the raw dataset.

Also re-derives the paper's Table 2 inputs (sigma, dividend yield, risk-free
rate) from the raw data at the paper's date, 2021-08-23, as a first check that
the dataset can reproduce the paper's parameters.

Usage:  python -m src.week1_figures
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_sources.config import PAPER_ANCHOR_DATE, PAPER_PARAMS, RAW_DIR, ROOT

OUT = ROOT / "deliverables" / "week1" / "figures"

# Reference categorical palette (validated, light mode), fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
})

EVENTS = [  # shaded regimes referenced in the report
    ("2020-02-20", "2020-04-30", "COVID-19 crash"),
    ("2022-03-16", "2023-07-26", "Fed hiking cycle"),
]


def _load():
    jpm = pd.read_csv(RAW_DIR / "yahoo_jpm_daily.csv", index_col="date", parse_dates=True)
    vix = pd.read_csv(RAW_DIR / "yahoo_vix_daily.csv", index_col="date", parse_dates=True)
    fd = pd.read_csv(RAW_DIR / "fred_daily.csv", index_col="date", parse_dates=True)
    return jpm, vix, fd


def _shade(ax, which=(0,), label=True):
    for i in which:
        a, b, name = EVENTS[i]
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#8a8984", alpha=0.12, lw=0)
        if label:
            ax.text(pd.Timestamp(b), 0.03, f" {name}", transform=ax.get_xaxis_transform(),
                    va="bottom", ha="left", fontsize=8, color=INK2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.85))


def _finish(fig, ax, name, source):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.01, 0.005, f"Source: {source}", fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def fig_price(jpm):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(jpm.index, jpm["close"], color=C[0], label="Close (unadjusted)")
    ax.plot(jpm.index, jpm["adj_close"], color=C[1], label="Adjusted close (dividends reinvested)")
    _shade(ax)
    t = pd.Timestamp(PAPER_ANCHOR_DATE)
    s0 = jpm.loc[t, "close"]
    ax.scatter([t], [s0], s=36, color=C[0], edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate(f"Paper S0 = ${s0:.2f}\n(23 Aug 2021)", (t, s0), xytext=(-95, 28),
                textcoords="offset points", fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_title("JPM daily price, 2018-2024")
    ax.set_ylabel("USD per share")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    _finish(fig, ax, "fig1_jpm_price.png", "Yahoo Finance (yfinance), auto_adjust=False")


def fig_vix(vix):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(vix.index, vix["close"], color=C[0])
    _shade(ax)
    pk = vix["close"].idxmax()
    ax.annotate(f"Peak {vix['close'].max():.1f} ({pk:%d %b %Y})", (pk, vix["close"].max()),
                xytext=(12, -6), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(vix["close"].median(), color=INK2, lw=0.8, ls="--")
    ax.text(vix.index[0], 44, f"dashed line: median {vix['close'].median():.1f}",
            va="bottom", ha="left", fontsize=8, color=INK2)
    ax.set_title("VIX index close, 2018-2024")
    ax.set_ylabel("Index points (implied vol, %)")
    _finish(fig, ax, "fig2_vix.png", "Yahoo Finance ^VIX (cross-checked with FRED VIXCLS)")


def fig_rates(fd):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = [("DTB3", "3-month T-bill"), ("DGS6MO", "6-month Treasury"),
              ("DGS1", "1-year Treasury"), ("DGS10", "10-year Treasury"), ("DFF", "Fed funds (effective)")]
    for (sid, lab), col in zip(series, C):
        s = fd[sid].dropna()
        ax.plot(s.index, s, color=col, label=lab, lw=1.2)
    _shade(ax, which=(0, 1))
    ax.set_title("US short- and long-term interest rates, 2018-2024")
    ax.set_ylabel("Percent per year")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88), ncol=2)
    _finish(fig, ax, "fig3_rates.png", "FRED (St. Louis Fed)")


def fig_vol(jpm, vix):
    r = np.log(jpm["close"]).diff()
    rv21 = r.rolling(21).std() * np.sqrt(252) * 100
    rv252 = r.rolling(252).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(rv21.index, rv21, color=C[0], lw=1.0, label="JPM 21-day realized vol")
    ax.plot(rv252.index, rv252, color=C[1], lw=1.6, label="JPM 252-day realized vol")
    _shade(ax)
    t = pd.Timestamp(PAPER_ANCHOR_DATE)
    ax.scatter([t], [PAPER_PARAMS["sigma"] * 100], s=36, color=INK, zorder=5)
    ax.annotate("Paper sigma 28.2%\n(1-yr historical, Aug 2021)", (t, PAPER_PARAMS["sigma"] * 100),
                xytext=(30, 55), textcoords="offset points", fontsize=8, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_title("JPM realized volatility: 1-month and 1-year rolling windows")
    ax.set_ylabel("Annualized volatility, %")
    ax.set_ylim(0, float(rv21.max()) * 1.05)
    ax.legend(loc="upper right")
    _finish(fig, ax, "fig4_volatility.png", "Yahoo Finance; realized vol = std of daily log returns x sqrt(252)")


def fig_dividends(jpm):
    d = jpm.loc[jpm["dividends"] > 0, "dividends"]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.bar(d.index, d.values, width=40, color=C[0], edgecolor="white", linewidth=1)
    for x, y in [(d.index[0], d.iloc[0]), (d.index[-1], d.iloc[-1])]:
        ax.text(x, y, f"${y:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("JPM quarterly dividend per share (ex-dividend dates)")
    ax.set_ylabel("USD per share")
    ax.grid(axis="x", visible=False)
    _finish(fig, ax, "fig5_dividends.png", "Yahoo Finance (cross-checked with Alpha Vantage DIVIDENDS)")


def _bsm(S, K, T, r, q, sigma, call=True):
    """Black-Scholes-Merton European option value with continuous dividend yield q."""
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def chooser_value(S0, K, T1, T2, r, q, sigma):
    """Rubinstein simple chooser: call to T2 + e^{-q tau} put(K e^{-(r-q) tau}) to T1."""
    tau = T2 - T1
    return _bsm(S0, K, T2, r, q, sigma) + np.exp(-q * tau) * _bsm(
        S0, K * np.exp(-(r - q) * tau), T1, r, q, sigma, call=False
    )


def fig_chooser_background():
    """Background figure: how a chooser works under BSM, using the paper's Table 2 inputs."""
    p = PAPER_PARAMS
    K, r, q, sig, T1, T2 = p["strike"], p["risk_free_rate"], p["dividend_yield"], p["sigma"], p["T1"], p["T2"]
    tau = T2 - T1
    s_star = K * np.exp(-(r - q) * tau)  # where call value = put value at T1

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) timeline
    ax0.set_xlim(-0.14, 1.16); ax0.set_ylim(-1, 1.25); ax0.axis("off")
    ax0.plot([0, 1], [0, 0], color=INK2, lw=1.4)
    for x, top, bottom in [(0, r"$t = 0$", "buy chooser,\npay premium " + r"$V_0$"),
                           (T1, r"$T_1$ = 0.5 y", "choose call or put,\nwhichever is\nworth more"),
                           (T2, r"$T_2$ = 1 y", "expiry: payoff\n" + r"$(S_{T_2}-K)^{+}$" + "\nor " + r"$(K-S_{T_2})^{+}$")]:
        ax0.plot([x], [0], "o", ms=8, color=C[0], mec="white", mew=1.5, zorder=3)
        ax0.text(x, 0.22, top, ha="center", va="bottom", fontsize=9, color=INK)
        ax0.text(x, -0.2, bottom, ha="center", va="top", fontsize=7.6, color=INK2)
    ax0.annotate("", xy=(T1, 0.85), xytext=(0, 0.85), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax0.text(T1 / 2, 0.92, "undecided:\nbenefits from up OR down", ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax0.annotate("", xy=(T2, 0.85), xytext=(T1, 0.85), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax0.text((T1 + T2) / 2, 0.92, "committed:\nordinary call or put", ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax0.set_title("(a) Contract timeline", fontsize=10)

    # (b) value at the choice date
    S = np.linspace(80, 240, 400)
    call = _bsm(S, K, tau, r, q, sig)
    put = _bsm(S, K, tau, r, q, sig, call=False)
    ax1.plot(S, call, color=C[0], lw=1.2, ls="--", label=r"Call value $C(S_{T_1}, K, \tau)$")
    ax1.plot(S, put, color=C[1], lw=1.2, ls="--", label=r"Put value $P(S_{T_1}, K, \tau)$")
    ax1.plot(S, np.maximum(call, put), color=INK, lw=2.0, label=r"Chooser $=\max(C, P)$")
    ax1.axvline(s_star, color=INK2, lw=0.8, ls=":")
    ax1.text(s_star - 4, 62, "switch point\n" + r"$S^{*} = K\,e^{-(r-q)\tau}$" + f"\n= \\${s_star:.2f}", fontsize=7.8, color=INK2, va="top", ha="right")
    ax1.text(100, 4, "choose PUT", fontsize=8, color=C[1], ha="center")
    ax1.text(205, 4, "choose CALL", fontsize=8, color=C[0], ha="center")
    ax1.set_xlabel(r"JPM price at the choice date, $S_{T_1}$ (USD)")
    ax1.set_ylabel(r"Option value at $T_1$ (USD)")
    ax1.set_ylim(0, 118)
    ax1.legend(loc="upper center", fontsize=7.8, ncol=1, bbox_to_anchor=(0.5, 1.0),
               frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax1.set_title("(b) Value at the choice date (BSM)", fontsize=10)

    v0 = chooser_value(p["S0"], K, T1, T2, r, q, sig)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_chooser_bsm.png", facecolor="white")
    plt.close(fig)
    return round(float(v0), 2)


def paper_parameter_check(jpm, fd) -> dict:
    """Re-derive Table 2 inputs from raw data at 2021-08-23."""
    t = pd.Timestamp(PAPER_ANCHOR_DATE)
    r = np.log(jpm["close"]).diff()
    win = r.loc[t - pd.DateOffset(years=1) + pd.Timedelta(days=1): t]
    sigma = float(win.std() * np.sqrt(252))
    divs = jpm.loc[t - pd.DateOffset(years=1) + pd.Timedelta(days=1): t, "dividends"].sum()
    s0 = float(jpm.loc[t, "close"])
    rates = {sid: float(fd.loc[:t, sid].dropna().iloc[-1]) for sid in ("DTB3", "DGS3MO", "DGS6MO", "DGS1")}
    out = {
        "S0": {"paper": PAPER_PARAMS["S0"], "data": round(s0, 2)},
        "sigma_1y_hist": {"paper": PAPER_PARAMS["sigma"], "data": round(sigma, 4), "n_returns": int(win.notna().sum())},
        "dividend_yield_trailing_12m": {"paper": PAPER_PARAMS["dividend_yield"], "data": round(float(divs / s0), 4), "dividends_sum": round(float(divs), 2)},
        "risk_free_rate_pct": {"paper": PAPER_PARAMS["risk_free_rate"] * 100, "data": rates},
    }
    return out


def summary_stats(jpm, vix, fd) -> dict:
    r = np.log(jpm["close"]).diff().dropna()
    by_year = r.groupby(r.index.year).std().mul(np.sqrt(252)).round(4).to_dict()
    return {
        "jpm_close": {"start": round(float(jpm["close"].iloc[0]), 2), "end": round(float(jpm["close"].iloc[-1]), 2),
                      "min": round(float(jpm["close"].min()), 2), "min_date": str(jpm["close"].idxmin().date()),
                      "max": round(float(jpm["close"].max()), 2), "max_date": str(jpm["close"].idxmax().date())},
        "jpm_daily_log_return": {"mean_annualized": round(float(r.mean() * 252), 4), "vol_annualized": round(float(r.std() * np.sqrt(252)), 4),
                                 "worst_day": round(float(r.min()), 4), "worst_day_date": str(r.idxmin().date()),
                                 "best_day": round(float(r.max()), 4), "best_day_date": str(r.idxmax().date())},
        "jpm_vol_by_year": {str(k): v for k, v in by_year.items()},
        "vix": {"median": round(float(vix["close"].median()), 2), "max": round(float(vix["close"].max()), 2),
                "max_date": str(vix["close"].idxmax().date()), "min": round(float(vix["close"].min()), 2)},
        "rates_range_pct": {sid: [round(float(fd[sid].min()), 2), round(float(fd[sid].max()), 2)] for sid in ("DTB3", "DGS1", "DGS10", "DFF")},
        "corr_jpm_ret_vs_vix_change": round(float(r.corr(vix["close"].diff().reindex(r.index))), 3),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jpm, vix, fd = _load()
    v0 = fig_chooser_background()
    fig_price(jpm); fig_vix(vix); fig_rates(fd); fig_vol(jpm, vix); fig_dividends(jpm)
    stats = {"chooser_value_paper_inputs": v0, "paper_parameter_check": paper_parameter_check(jpm, fd), "summary": summary_stats(jpm, vix, fd)}
    (OUT.parent / "week1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
