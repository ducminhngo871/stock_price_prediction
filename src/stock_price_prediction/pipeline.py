"""End-to-end: load prices for one ticker, build features, split, fit every model."""

from stock_price_prediction.data import load_combined
from stock_price_prediction.features import build_features, time_split
from stock_price_prediction.model import run_all_models


def run_forecast(ticker: str, end: str | None = None, test_size: float = 0.2, cache: bool = True) -> dict:
    prices = load_combined([ticker], end=end, cache=cache)
    featured = build_features(prices)
    train_df, test_df = time_split(featured, test_size=test_size)
    run = run_all_models(train_df, test_df)
    return {"ticker": ticker, "prices": prices, "train": train_df, "test": test_df, **run}
