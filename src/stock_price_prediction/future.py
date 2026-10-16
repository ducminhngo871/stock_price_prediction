"""Genuine future forecasts: fit every model on ALL available history and project forward
past the last known trading day.

This is a different exercise from `model.py`'s backtest. The backtest holds out a known
test period so predictions can be scored against real outcomes. Here there is no real
outcome to hold out or score against -- by definition, nobody knows next year's price --
so there are no RMSE/directional-accuracy numbers here, only projections and, where
possible, honest uncertainty bands.

Ridge and random forest have no native multi-step mode: they predict one day at a time
from lag/rolling features of the close price. With no real future closes to build tomorrow's
features from, each model's own prediction is fed back in as if it were the real close (a
standard "recursive forecast"). Errors compound with every step this way, and random forest
in particular is already known (see model.py / README) to be unable to extrapolate past the
price range it was trained on -- recursively forecasting with it tends to flatten out fast.
ARIMA, by contrast, forecasts multiple steps natively and comes with a real confidence
interval; naive's band comes from the historical daily volatility scaled by sqrt(time), the
standard random-walk result.
"""

import numpy as np
import pandas as pd

from stock_price_prediction.features import FEATURE_COLUMNS, build_features
from stock_price_prediction.model import fit_linear_regression, fit_random_forest


def future_business_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def _latest_feature_row(history: pd.DataFrame) -> pd.DataFrame:
    """The same lag/rolling features as features.py, computed for just the most recent row.

    lag_1 is today's close (history's last row) -- the whole point of a recursive forecast is
    that each step's own prediction becomes "today" for the next step, so this must match
    build_features' definition of lag_1 exactly or the model sees an off-distribution input.
    """
    close = history["close"]
    row = {
        "lag_1": close.iloc[-1], "lag_2": close.iloc[-2], "lag_3": close.iloc[-3], "lag_5": close.iloc[-5],
        "rolling_mean_5": close.iloc[-5:].mean(), "rolling_mean_10": close.iloc[-10:].mean(),
        "rolling_mean_20": close.iloc[-20:].mean(),
        "rolling_std_5": close.iloc[-5:].std(), "rolling_std_10": close.iloc[-10:].std(),
        "return_1": close.iloc[-1] / close.iloc[-2] - 1,
        "volume_today": history["volume"].iloc[-1],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def recursive_forecast(model, history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Step forward one day at a time, feeding each prediction back in as the next day's lag_1."""
    dates = future_business_dates(history["date"].iloc[-1], horizon)
    extended = history[["date", "close", "volume"]].copy()
    preds = np.empty(horizon)
    last_volume = extended["volume"].iloc[-1]
    for i in range(horizon):
        pred = float(model.predict(_latest_feature_row(extended))[0])
        preds[i] = pred
        extended.loc[len(extended)] = {"date": dates[i], "close": pred, "volume": last_volume}
    return preds


def naive_future_forecast(history: pd.DataFrame, horizon: int) -> dict:
    """Flat at the last close (the random-walk expectation), with a 95% band from
    historical daily volatility scaled by sqrt(time) -- the standard random-walk result,
    and a direct illustration of how little a long-horizon point forecast means here."""
    last_close = float(history["close"].iloc[-1])
    daily_vol = history["close"].pct_change().std()
    point = np.full(horizon, last_close)
    band = last_close * daily_vol * np.sqrt(np.arange(1, horizon + 1))
    return {"point": point, "lo": point - 1.96 * band, "hi": point + 1.96 * band}


def arima_future_forecast(history: pd.DataFrame, horizon: int, order=(5, 1, 0)) -> dict:
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(history["close"].to_numpy(), order=order).fit()
    result = fitted.get_forecast(steps=horizon)
    ci = np.asarray(result.conf_int(alpha=0.05))
    return {"point": np.asarray(result.predicted_mean), "lo": ci[:, 0], "hi": ci[:, 1]}


def all_future_forecasts(ticker: str, end: str | None = None, horizon: int = 252) -> dict:
    """Fit every model on the full history for `ticker` and project `horizon` trading days forward."""
    from stock_price_prediction.data import load_combined

    prices = load_combined([ticker], end=end)
    featured = build_features(prices)  # drops only the final row, whose target (tomorrow) isn't known yet

    ridge = fit_linear_regression(featured)
    rf = fit_random_forest(featured)

    dates = future_business_dates(prices["date"].iloc[-1], horizon)
    return {
        "ticker": ticker,
        "prices": prices,
        "last_date": prices["date"].iloc[-1],
        "last_close": float(prices["close"].iloc[-1]),
        "dates": dates,
        "forecasts": {
            "naive": naive_future_forecast(prices, horizon),
            "linear_regression": {"point": recursive_forecast(ridge, prices, horizon)},
            "random_forest": {"point": recursive_forecast(rf, prices, horizon)},
            "arima": arima_future_forecast(prices, horizon),
        },
    }
