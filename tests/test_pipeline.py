"""Offline tests for the Week 2 pipeline (no network, no API keys).

Run:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.data_sources.config import PAPER_ANCHOR_DATE, RAW_DIR
from src.pipeline.clean import load_raw
from src.pipeline.features import FEATURES, build_features
from src.pipeline.run import CSV, PARQUET


@pytest.fixture(scope="module")
def full():
    return build_features(load_raw())


@pytest.mark.parametrize("cut", ["2018-04-27", "2018-06-29", "2020-03-20", "2021-08-23", "2023-01-31"])
def test_no_lookahead(full, cut):
    """Rebuild with every raw value published after `cut` removed. Each feature up to
    `cut` must be identical: if a feature used future data, it would change."""
    part = build_features(load_raw(cut=cut))
    pd.testing.assert_frame_equal(part, full.loc[:cut], check_exact=False, rtol=1e-12, atol=0)


def test_feature_count_and_completeness(full):
    assert len(FEATURES) >= 10
    assert not full.isna().any().any()
    week1 = pd.read_csv(RAW_DIR / "yahoo_jpm_daily.csv", index_col="date", parse_dates=True).index
    assert full.index.equals(week1)


def test_ranges(full):
    for col in ("sent_news", "sent_consumer"):
        assert full[col].between(0, 1).all()
    assert full["corr_jpm_vix_63"].between(-1, 1).all()
    for col in ("rv_21", "rv_63", "rv_252", "park_21", "vix"):
        assert (full[col] > 0).all()


def test_week1_anchor(full):
    a = full.loc[PAPER_ANCHOR_DATE]
    assert a.close == pytest.approx(156.70, abs=0.005)
    assert a.rv_252 == pytest.approx(0.2722, abs=0.0005)
    assert a.div_yield == pytest.approx(0.0230, abs=0.0001)


def test_committed_outputs_match_rebuild(full):
    """The committed Parquet file equals a fresh build, and the CSV equals the Parquet."""
    pq = pd.read_parquet(PARQUET)
    pd.testing.assert_frame_equal(pq, full, check_freq=False, rtol=1e-12)
    csv = pd.read_csv(CSV, index_col="date", parse_dates=True)
    assert np.allclose(csv.to_numpy(), pq.to_numpy(), rtol=1e-9, atol=1e-12)
    assert list(csv.columns) == list(pq.columns)


def test_deterministic():
    a = build_features(load_raw())
    b = build_features(load_raw())
    pd.testing.assert_frame_equal(a, b, check_exact=True)
