"""Lag/rolling features for next-day close-price forecasting.

Every feature at row t uses only information available up to and including
day t (lags, rolling stats of the past). The target is close at t+1, so no
future information leaks into the features.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_5",
    "rolling_mean_5", "rolling_mean_10", "rolling_mean_20",
    "rolling_std_5", "rolling_std_10",
    "return_1", "volume_lag_1",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag/rolling features and the next-day-close target for a single ticker's price series."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]

    df["lag_1"] = close.shift(1)
    df["lag_2"] = close.shift(2)
    df["lag_3"] = close.shift(3)
    df["lag_5"] = close.shift(5)
    df["rolling_mean_5"] = close.shift(1).rolling(5).mean()
    df["rolling_mean_10"] = close.shift(1).rolling(10).mean()
    df["rolling_mean_20"] = close.shift(1).rolling(20).mean()
    df["rolling_std_5"] = close.shift(1).rolling(5).std()
    df["rolling_std_10"] = close.shift(1).rolling(10).std()
    df["return_1"] = close.shift(1).pct_change()
    df["volume_lag_1"] = df["volume"].shift(1)
    df["target"] = close.shift(-1)

    return df.dropna(subset=[*FEATURE_COLUMNS, "target"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split (no shuffling) -- a random split would leak future prices into training."""
    split_at = int(len(df) * (1 - test_size))
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()
