"""Recommended ("ideal") demo configuration, and the evidence behind it.

Unlike the thin-film project's POR (a config chosen to pass its spec), this one
is chosen to best demonstrate the project's actual finding: that liquid stock
prices are close enough to a random walk that none of these models reliably
beats "tomorrow equals today." AAPL is the most-covered, most liquid single
name in the dataset, with the longest clean history (2013-present), so it
shows that result with the least noise.
"""

IDEAL = {
    "settings": {"ticker": "AAPL", "test_size": 0.2},
    "why": "AAPL has the longest clean, continuous history in this dataset (2013-present, no missing "
           "days) and is one of the most liquid single stocks traded, so its price is about as close to "
           "an efficient, information-driven random walk as a single name gets. On this config Ridge and "
           "ARIMA land within ~1% of naive's RMSE, and every model's directional accuracy sits near 50% -- "
           "the clearest demonstration of the project's main finding.",
    "source": "results/price_forecast_AAPL",
}
