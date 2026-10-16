"""Lag/rolling features for next-day close-price forecasting.

Every feature at row t uses only information available up to and including
day t (lags, rolling stats of the past, today's own close and volume). The
target is close at t+1, so no future information leaks into the features.

`lag_1` is today's close (row t), not yesterday's: a previous version applied
an extra `.shift(1)` to every feature here, which excluded today's close from
the feature set entirely and made every "next-day" prediction actually a
stale two-day-ahead one (lag_1 at day t-1 predicting a target at day t+1).
Caught while building `future.py`'s recursive forecaster, which feeds each
prediction back in as "today's close" for the next step -- that only works
if lag_1 genuinely means today.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_5",
    "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
    "rolling_std_5", "rolling_std_10",
    "return_1", "volume_today",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag/rolling features and the next-day-close target for a single ticker's price series."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]

    df["lag_1"] = close
    df["lag_2"] = close.shift(1)
    df["lag_3"] = close.shift(2)
    df["lag_5"] = close.shift(4)
    df["rolling_mean_5"] = close.rolling(5).mean()
    df["rolling_mean_10"] = close.rolling(10).mean()
    df["rolling_mean_20"] = close.rolling(20).mean()
    df["rolling_std_5"] = close.rolling(5).std()
    df["rolling_std_10"] = close.rolling(10).std()
    df["return_1"] = close.pct_change()
    df["volume_today"] = df["volume"]
    df["target"] = close.shift(-1)

    return df.dropna(subset=[*FEATURE_COLUMNS, "target"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split (no shuffling) -- a random split would leak future prices into training."""
    split_at = int(len(df) * (1 - test_size))
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()
