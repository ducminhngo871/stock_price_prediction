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
    """Load the Kaggle S&P 500 CSV (2013-02-08 to 2018-02-07).

    Filters by ticker before parsing dates, not after: the file holds 505 tickers, and
    date-parsing is the slow part of reading it, so there's no reason to pay that cost
    for the ~504 tickers a single-ticker call is about to throw away.
    """
    df = pd.read_csv(KAGGLE_CSV)
    df = df.rename(columns={"Name": "ticker"})
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]].sort_values(["ticker", "date"])


def fetch_yfinance(tickers: list[str], start: str = "2018-02-08", end: str | None = None,
                    cache: bool = True) -> pd.DataFrame:
    """Download daily OHLCV for tickers from yfinance, caching each ticker as a CSV.

    A cached file is only reused if it already covers both the requested start and end
    dates (within a long weekend's slack); otherwise it's refetched. Both ends matter:
    without the end check, a cache written once would silently serve the same stale
    "latest" date forever; without the start check, a narrower-start cache (e.g. from
    load_recent's trailing-years window) would silently miss older rows a full-history
    call needs, or vice versa -- both call this with the same per-ticker cache file.
    """
    import yfinance as yf

    target_end = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    target_start = pd.Timestamp(start)
    frames = []
    YFINANCE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        cache_path = YFINANCE_DIR / f"{ticker}.csv"
        cached = pd.read_csv(cache_path, parse_dates=["date"]) if cache_path.exists() else None
        fresh_enough = (cached is not None
                        and cached["date"].max() >= target_end - pd.Timedelta(days=4)
                        and cached["date"].min() <= target_start + pd.Timedelta(days=4))
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
        # A reused cache file can span well past what was asked for -- e.g. it was written by
        # an earlier "latest data" call, and this call pins a specific historical "as of" date.
        # Trim to exactly [target_start, target_end] so a historical request can never silently
        # include newer rows the cache happens to already have.
        frame = frame[(frame["date"] >= target_start) & (frame["date"] <= target_end)]
        frames.append(frame)
    if not frames:
        # Explicit dtypes matter here: an untyped empty frame defaults every column to
        # object, and concatenating that with the (float) Kaggle portion below silently
        # upcasts the whole combined "close" column to object even though this frame
        # contributes zero rows -- which then breaks ARIMA with an opaque isnan TypeError.
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"), "ticker": pd.Series(dtype=str),
            "open": pd.Series(dtype=float), "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float), "close": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
        })
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
    # Not every "split" yfinance reports is a clean integer ratio: merger/acquisition share
    # conversions (e.g. SunTrust -> Truist at 1.371 shares) show up here too, and pandas 3.x
    # raises rather than silently upcasting when a fractional value is written into volume's
    # int64 column.
    df["volume"] = df["volume"].astype(float)
    for split_date, ratio in late_splits.sort_index(ascending=False).items():
        mask = df["date"] < split_date
        df.loc[mask, ["open", "high", "low", "close"]] /= ratio
        df.loc[mask, "volume"] *= ratio
    return df


def _raise_if_stale(df: pd.DataFrame, tickers: list[str], end: str | None) -> None:
    """When "latest data" was requested (end=None), silently handing back old data for a
    ticker with nothing recent -- typically one delisted, renamed, or acquired -- would
    look like a normal, current result. A wide cutoff on purpose: a dead ticker's data
    stops entirely (years stale), a world apart from an active ticker's free-data-feed
    lag, observed up to ~2 months on some names.
    """
    if end is not None:
        return
    stale_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=90)
    for ticker in tickers:
        last = df.loc[df["ticker"] == ticker, "date"].max()
        if pd.isna(last) or last < stale_cutoff:
            when = "no data at all" if pd.isna(last) else f"last priced {last.date()}"
            raise ValueError(f"No recent price data for {ticker} ({when}). It may be delisted, "
                             "renamed, or acquired, or the data source may be lagging for this "
                             "ticker. Try another ticker, or pin an earlier 'as of' date.")


def load_combined(tickers: list[str], end: str | None = None, cache: bool = True) -> pd.DataFrame:
    """Kaggle history (2013-2018) + yfinance (2018-02-08 onward), stitched into one consistent series.

    Use this for a full backtest or report figure. For the app's interactive use, prefer
    `load_recent`: it skips the Kaggle file and the stitching logic entirely, which is
    both the slowest part of this function and unnecessary for a window short enough to
    be covered by yfinance alone.
    """
    kaggle = pd.concat(
        [adjust_kaggle_for_late_splits(group, ticker)
         for ticker, group in load_kaggle(tickers).groupby("ticker")],
        ignore_index=True,
    )
    recent = fetch_yfinance(tickers, start="2018-02-08", end=end, cache=cache)
    combined = pd.concat([kaggle, recent], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    _raise_if_stale(combined, tickers, end)
    return combined


def load_recent(tickers: list[str], years: int = 5, end: str | None = None, cache: bool = True) -> pd.DataFrame:
    """Trailing `years` of daily OHLCV from yfinance alone -- no Kaggle file, no stitching.

    yfinance's own OHLC is already fully split-adjusted (see module docstring), so a
    window this short needs none of load_combined's stitching logic. This skips: parsing
    the 505-ticker, 619k-row Kaggle CSV; a second network round trip per ticker for split
    metadata; and most of the rows a backtest model would otherwise have to fit on --
    the main reason an interactive app built on load_combined felt slow.
    """
    target_end = pd.Timestamp(end) if end else pd.Timestamp.now().normalize()
    start = target_end - pd.DateOffset(years=years)
    recent = fetch_yfinance(tickers, start=start.strftime("%Y-%m-%d"), end=end, cache=cache)
    # Trim even if a wider-range cache (e.g. from a prior load_combined call) was reused,
    # so this always does the one job load_recent exists for: bounding how much data the
    # rest of the pipeline has to process.
    recent = recent[recent["date"] >= start].sort_values(["ticker", "date"]).reset_index(drop=True)
    _raise_if_stale(recent, tickers, end)
    return recent
