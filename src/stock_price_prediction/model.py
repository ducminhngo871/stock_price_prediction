"""Next-day close-price forecasting models, from a naive baseline up to ML regressors.

The naive (random-walk) baseline matters more here than in most forecasting
problems: liquid stock prices are close to a random walk, so "tomorrow equals
today" is a notoriously hard baseline to beat on raw price-level error. Every
model is judged against it, and directional accuracy (did it get up/down
right) is reported alongside RMSE, since a model can have low price-level
error while being no better than a coin flip on direction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from stock_price_prediction.features import FEATURE_COLUMNS

SEED = 42


def naive_forecast(test_df: pd.DataFrame) -> np.ndarray:
    """Predict tomorrow's close as today's close (the random-walk baseline).

    Because this prediction literally equals y_prev, its directional accuracy
    (see `evaluate`) is 0% except on the rare flat trading day (no price
    change, where "no change" happens to be the right call): it never
    deliberately predicts a direction at all. That's not a bug, it's what
    "no change" means -- report it alongside the other models' directional
    accuracy with that in mind.
    """
    return test_df["lag_1"].to_numpy()


def fit_linear_regression(train_df: pd.DataFrame, alpha: float = 1.0) -> Ridge:
    """Ridge, not plain OLS: lag_1..lag_5 and the rolling means are >0.99 correlated,
    which makes the OLS normal equations near-singular and its coefficients degenerate
    (observed: predictions collapsed to a near-constant far from the actual price range).
    Ridge's L2 term conditions the problem without having to drop any of the features.
    """
    model = Ridge(alpha=alpha)
    model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
    return model


def fit_random_forest(train_df: pd.DataFrame, n_estimators: int = 100) -> RandomForestRegressor:
    """100 trees, not 300: verified to give essentially identical RMSE/directional accuracy on
    backtests (expected -- random forest's inability to extrapolate past its training range is
    structural, not a function of tree count), while being 3x cheaper per prediction. That matters
    here specifically because recursive_forecast calls .predict() hundreds of times in a loop,
    where this cost is paid on every single step.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=6, random_state=SEED, n_jobs=-1)
    model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
    return model


def arima_rolling_forecast(train_close: pd.Series, test_close: pd.Series, order=(5, 1, 0)) -> np.ndarray:
    """One-step-ahead ARIMA forecasts, updating with each true observation as the test period unfolds.

    This is the standard rolling-origin backtest: the model never sees a test value before
    forecasting it, but afterwards is told the true value (as it would be, one day later).
    """
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(train_close.to_numpy(), order=order).fit()
    preds = []
    for actual in test_close.to_numpy():
        preds.append(fitted.forecast(1)[0])
        fitted = fitted.append([actual], refit=False)
    return np.array(preds)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> dict:
    """RMSE/MAE/MAPE on price level, plus directional accuracy relative to the prior close (y_prev)."""
    err = y_true - y_pred
    direction_true = np.sign(y_true - y_prev)
    direction_pred = np.sign(y_pred - y_prev)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
        "directional_accuracy": float(np.mean(direction_true == direction_pred)),
    }


def run_all_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Fit every model on train_df, forecast test_df, and return predictions + metrics for each."""
    y_true = test_df["target"].to_numpy()
    y_prev = test_df["lag_1"].to_numpy()

    predictions = {"naive": naive_forecast(test_df)}

    lr = fit_linear_regression(train_df)
    predictions["linear_regression"] = lr.predict(test_df[FEATURE_COLUMNS])

    rf = fit_random_forest(train_df)
    predictions["random_forest"] = rf.predict(test_df[FEATURE_COLUMNS])

    predictions["arima"] = arima_rolling_forecast(train_df["close"], test_df["close"])

    metrics = {name: evaluate(y_true, preds, y_prev) for name, preds in predictions.items()}
    return {"predictions": predictions, "metrics": metrics, "y_true": y_true, "y_prev": y_prev}
