"""Load and combine price history from the Kaggle S&P 500 dump and yfinance.

Kaggle (camnugent/sandp500) covers 2013-02-08 to 2018-02-07, verified free of
missing tickers. yfinance extends that range up to the present. Both are
normalized to the same schema before concatenation:
    date (datetime64), ticker (str), open, high, low, close, volume (float)

yfinance's OHLC (even with auto_adjust=False) is always back-adjusted for every
split up to today -- that flag only controls dividend adjustment, not splits.
The Kaggle file was scraped around its 2018-02-07 cutoff, so it is back-adjusted
only for splits that had already happened by then (e.g. AAPL's 2014 7-for-1),
not later ones (e.g. AAPL's 2020 4-for-1). Naively concatenating the two, or
re-adjusting both for full split history, creates a price discontinuity right
at the stitch point. The fix: adjust only the Kaggle rows, and only for splits
dated after the Kaggle cutoff (`adjust_for_splits`).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_CSV = ROOT / "data" / "raw" / "kaggle_sandp500" / "all_stocks_5yr.csv"
YFINANCE_DIR = ROOT / "data" / "raw" / "yfinance"

KAGGLE_END = pd.Timestamp("2018-02-07")


def available_tickers() -> list[str]:
    """Tickers present in the Kaggle file (so every one has a full 2013-onward history)."""
    return sorted(pd.read_csv(KAGGLE_CSV, usecols=["Name"])["Name"].unique().tolist())


def load_kaggle(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load the Kaggle S&P 500 CSV (2013-02-08 to 2018-02-07)."""
    df = pd.read_csv(KAGGLE_CSV, parse_dates=["date"])
    df = df.rename(columns={"Name": "ticker"})
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values(["ticker", "date"])


def fetch_yfinance(tickers: list[str], start: str = "2018-02-08", end: str | None = None,
                    cache: bool = True) -> pd.DataFrame:
    """Download daily OHLCV for tickers from yfinance, caching each ticker as a CSV.

    A cached file is only reused if it already reaches the requested end date (within
    a long weekend's slack); otherwise it's refetched. Without this check, a cache
    written once would silently serve the same stale end date forever -- the opposite
    of what an app showing "latest data" needs.
    """
    import yfinance as yf

    target_end = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    frames = []
    YFINANCE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        cache_path = YFINANCE_DIR / f"{ticker}.csv"
        cached = pd.read_csv(cache_path, parse_dates=["date"]) if cache_path.exists() else None
        fresh_enough = cached is not None and cached["date"].max() >= target_end - pd.Timedelta(days=4)
        if cache and fresh_enough:
            frame = cached
        else:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if raw.empty:
                continue
            raw.columns = raw.columns.get_level_values(0)
            frame = raw.reset_index().rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            frame["ticker"] = ticker
            frame = frame[["date", "ticker", "open", "high", "low", "close", "volume"]]
            if cache:
                frame.to_csv(cache_path, index=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


def adjust_kaggle_for_late_splits(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Back-adjust Kaggle rows for splits dated after the Kaggle cutoff (KAGGLE_END).

    Splits before the cutoff are already baked into the Kaggle prices; yfinance
    rows need no adjustment at all (see module docstring).
    """
    import yfinance as yf

    splits = yf.Ticker(ticker).splits
    # yfinance's metadata endpoint occasionally returns None on a transient failure
    # instead of raising or returning an empty Series; treat that the same as "no splits"
    # rather than crashing the whole forecast over a flaky network call.
    if splits is None or splits.empty:
        return df
    splits = splits.copy()
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    late_splits = splits[splits.index > KAGGLE_END]
    if late_splits.empty:
        return df
    df = df.sort_values("date").copy()
    for split_date, ratio in late_splits.sort_index(ascending=False).items():
        mask = df["date"] < split_date
        df.loc[mask, ["open", "high", "low", "close"]] /= ratio
        df.loc[mask, "volume"] *= ratio
    return df


def load_combined(tickers: list[str], end: str | None = None, cache: bool = True) -> pd.DataFrame:
    """Kaggle history (2013-2018) + yfinance (2018-02-08 onward), stitched into one consistent series."""
    kaggle = pd.concat(
        [adjust_kaggle_for_late_splits(group, ticker)
         for ticker, group in load_kaggle(tickers).groupby("ticker")],
        ignore_index=True,
    )
    recent = fetch_yfinance(tickers, start="2018-02-08", end=end, cache=cache)
    combined = pd.concat([kaggle, recent], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)
