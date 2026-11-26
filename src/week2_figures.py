"""Week 2 figures: data checks on cleaning and the engineered features.

Usage:  python -m src.week2_figures   (run after python -m src.pipeline.run)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .data_sources.config import ROOT
from .pipeline.features import FEATURES
from .week1_figures import C, GRID, INK, INK2  # same palette and style as Week 1

PROC = ROOT / "data" / "processed"
OUT = ROOT / "deliverables" / "week2" / "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
})


def _years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _save(fig, name, source):
    fig.text(0.01, 0.005, f"Source: {source}", fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def fig_gaps(log):
    """Trading days in 2018-2024 with no same-day value, before cleaning."""
    gaps = {k: v["gap_days"] for k, v in log["fred_gap_days"].items()}
    rows = [("JPM close (Yahoo)", 0), ("VIX close (Yahoo)", 0)] + \
           [(f"{k} (FRED)", v) for k, v in gaps.items()] + \
           [("GDELT news tone (calendar days)", len(log["gdelt_missing_days"]))]
    labels, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=C[0], height=0.6, edgecolor="white")
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, str(v), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Days without a same-day value (before cleaning)")
    ax.set_xlim(0, max(vals) * 1.25 + 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Missing values by series, 2018-2024")
    _save(fig, "fig1_missing_values.png", "cleaning_log.json; after cleaning every series has 0 blanks")


def fig_outliers(flags, log):
    o = log["outliers"]
    r = flags["ret"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(r.index, r * 100, color=C[0], lw=0.6, label="JPM daily log return")
    for k, ls in (("k1.5", "--"), ("k3.0", "-")):
        for b in ("lower", "upper"):
            ax.axhline(o[k][b] * 100, color=INK2, lw=0.8, ls=ls)
    f3 = flags["flag_k3.0"]
    ax.scatter(r.index[f3], r[f3] * 100, s=22, color=C[1], edgecolor="white", linewidth=1, zorder=4,
               label=f"Outside 3 x IQR fence ({int(f3.sum())} days, all kept)")
    ax.text(r.index[5], o["k1.5"]["upper"] * 100 + 0.4, "1.5 x IQR fence (dashed)", fontsize=7.8, color=INK2)
    ax.text(r.index[5], o["k3.0"]["upper"] * 100 + 0.4, "3 x IQR fence (solid)", fontsize=7.8, color=INK2)
    ax.set_ylabel("Daily return, %")
    ax.set_title("Outlier screen on JPM daily returns (IQR method)")
    ax.legend(loc="lower right", fontsize=8)
    _years(ax)
    _save(fig, "fig2_outliers.png", "Yahoo Finance; fences from 2018-2024 quartiles of daily log returns")


def fig_vol(df):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for col, c, lab in (("rv_21", C[0], "21-day"), ("rv_63", C[1], "63-day"), ("rv_252", C[2], "252-day")):
        a.plot(df.index, df[col] * 100, color=c, lw=1.0 if col == "rv_21" else 1.5, label=lab)
    a.set_title("(a) Close-to-close volatility", fontsize=10)
    a.set_ylabel("Annualized volatility, %")
    a.legend(loc="upper right", fontsize=8)
    b.plot(df.index, df.rv_21 * 100, color=C[0], lw=1.0, label="Close-to-close, 21-day")
    b.plot(df.index, df.park_21 * 100, color=C[1], lw=1.0, label="Parkinson (high-low), 21-day")
    b.set_title("(b) Two 21-day estimators", fontsize=10)
    b.legend(loc="upper right", fontsize=8)
    for ax in (a, b):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, "fig3_volatility_features.png", "features_daily.parquet")


def fig_sentiment(df):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df.index, df.sent_news, color=C[0], lw=0.6, alpha=0.85, label="News sentiment (GDELT, daily)")
    ax.plot(df.index, df.sent_consumer, color=C[1], lw=1.8, label="Consumer sentiment (U. Michigan, monthly)")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0 = most negative, 1 = most positive)")
    ax.set_title("Sentiment features scaled to [0, 1]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=2)
    _years(ax)
    _save(fig, "fig4_sentiment.png", "GDELT DOC API; FRED/ALFRED UMCSENT first releases")


def fig_corr(df):
    feats = list(FEATURES)
    c = df[feats].corr().to_numpy()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(feats)), feats, fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Correlation", color=INK2)
    ax.set_title("Correlation between the 22 features, 2018-2024")
    _save(fig, "fig5_feature_correlation.png", "features_daily.parquet")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "features_daily.parquet")
    flags = pd.read_csv(PROC / "outlier_flags.csv", index_col="date", parse_dates=True)
    log = json.loads((PROC / "cleaning_log.json").read_text())
    fig_gaps(log); fig_outliers(flags, log); fig_vol(df); fig_sentiment(df); fig_corr(df)
    c = df[list(FEATURES)].corr().abs().where(~np.eye(len(FEATURES), dtype=bool))
    top = c.stack().sort_values(ascending=False).iloc[::2].head(3)
    print("Top |corr| pairs:", [(a, b, round(v, 3)) for (a, b), v in top.items()])


if __name__ == "__main__":
    main()
