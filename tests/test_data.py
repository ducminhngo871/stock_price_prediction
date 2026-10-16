"""Guards against the Kaggle/yfinance stitching bugs discovered during setup:
the Kaggle file is missing ~66% of tickers in the newer andrewmvd dataset (not used
here) and, if split-adjustment is handled naively, splits create a fake price jump
exactly at the point the two sources are joined.
"""

import pandas as pd
import pytest

from stock_price_prediction.data import KAGGLE_END, load_combined, load_kaggle, load_recent

TEST_TICKERS = ["AAPL", "NVDA", "JNJ"]


@pytest.fixture(scope="module")
def combined():
    return {t: load_combined([t], end="2024-07-01") for t in TEST_TICKERS}


def test_kaggle_has_no_missing_tickers():
    df = load_kaggle()
    valid_counts = df.groupby("ticker")["close"].apply(lambda x: x.notnull().sum())
    assert (valid_counts == 0).sum() == 0


def test_combined_has_no_nulls_or_duplicate_dates(combined):
    for ticker, df in combined.items():
        assert df.isnull().sum().sum() == 0, ticker
        assert df.duplicated(subset="date").sum() == 0, ticker


def test_no_discontinuity_at_kaggle_yfinance_stitch(combined):
    """The split-adjustment bug this guards against produced >4x jumps right at the stitch date."""
    for ticker, df in combined.items():
        window = df[(df["date"] >= KAGGLE_END - pd.Timedelta(days=5))
                     & (df["date"] <= KAGGLE_END + pd.Timedelta(days=5))].sort_values("date")
        pct_change = window["close"].pct_change().abs()
        assert pct_change.max() < 0.15, f"{ticker}: {pct_change.max():.1%} jump near the stitch date"


def test_no_discontinuity_at_known_splits(combined):
    """NVDA split 4:1 on 2021-07-20 and 10:1 on 2024-06-10; a missed adjustment leaves a >50% jump."""
    df = combined["NVDA"]
    for split_date in ["2021-07-20", "2024-06-10"]:
        split_date = pd.Timestamp(split_date)
        window = df[(df["date"] >= split_date - pd.Timedelta(days=5))
                     & (df["date"] <= split_date + pd.Timedelta(days=5))].sort_values("date")
        pct_change = window["close"].pct_change().abs()
        assert pct_change.max() < 0.15, f"NVDA split {split_date.date()}: {pct_change.max():.1%} jump"


def test_fractional_split_ratio_does_not_crash_volume_adjustment():
    """STI (SunTrust) converted into Truist at a 1.371 ratio -- a fractional "split" that
    pandas 3.x rejects when written into an int64 volume column unless it's cast to float
    first. This must not raise."""
    df = load_combined(["STI"], end="2024-07-01")
    assert not df.empty
    assert df["volume"].dtype.kind == "f"


def test_load_combined_raises_clearly_for_a_ticker_with_no_recent_data():
    """XLNX (Xilinx) was acquired by AMD and delisted in 2022; yfinance has no data for it
    past the Kaggle cutoff. Silently falling back to 2018-era data when "latest" was asked
    for would look like a normal, current result -- it must raise instead."""
    with pytest.raises(ValueError, match="No recent price data"):
        load_combined(["XLNX"], end=None)


def test_load_recent_is_bounded_to_the_requested_window():
    """The whole point of load_recent is a smaller, faster dataset -- it must not silently
    return more than asked for, even if a wider-range file is already cached locally."""
    df = load_recent(["AAPL"], years=5, end="2024-07-01")
    assert not df.empty
    assert df["date"].min() >= pd.Timestamp("2024-07-01") - pd.DateOffset(years=5)
    assert df["date"].max() <= pd.Timestamp("2024-07-01")
    assert df.isnull().sum().sum() == 0


def test_load_recent_matches_load_combined_over_their_shared_window():
    """load_recent skips the Kaggle file and its split-adjustment entirely; its prices for
    dates both cover should still agree with load_combined's (same underlying yfinance
    source, just without the Kaggle-stitching machinery)."""
    recent = load_recent(["AAPL"], years=2, end="2024-07-01")
    combined = load_combined(["AAPL"], end="2024-07-01")
    merged = recent.merge(combined, on="date", suffixes=("_recent", "_combined"))
    assert len(merged) == len(recent)
    assert (merged["close_recent"] - merged["close_combined"]).abs().max() < 0.01
