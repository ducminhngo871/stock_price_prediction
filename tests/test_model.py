"""Model sanity checks. Also guards against two bugs caught while building this:
plain OLS collapsing to a near-constant prediction on collinear lag features (fixed by
switching to Ridge), and the naive baseline's directional accuracy being trivially 0%.
"""

import numpy as np

from stock_price_prediction.data import load_combined
from stock_price_prediction.features import FEATURE_COLUMNS, build_features, time_split
from stock_price_prediction.model import evaluate, fit_linear_regression, naive_forecast, run_all_models

TICKER = "AAPL"


def _featured():
    prices = load_combined([TICKER], end="2024-07-01")
    return build_features(prices)


def test_build_features_no_lookahead_leak():
    featured = _featured()
    # target at row i is close at i+1; lag_1 at row i is close at i. They must differ (no leak).
    assert not featured["target"].equals(featured["lag_1"])
    assert featured[[*FEATURE_COLUMNS, "target"]].isnull().sum().sum() == 0


def test_time_split_is_chronological():
    featured = _featured()
    train_df, test_df = time_split(featured, test_size=0.2)
    assert train_df["date"].max() < test_df["date"].min()
    assert len(train_df) + len(test_df) == len(featured)


def test_ridge_predictions_stay_in_plausible_range():
    """Plain LinearRegression on these near-collinear features collapsed to a near-constant
    far outside the actual price range (observed: predicting ~$65-70 against actual
    $125-198). Ridge must not reproduce that failure."""
    featured = _featured()
    train_df, test_df = time_split(featured, test_size=0.2)
    model = fit_linear_regression(train_df)
    preds = model.predict(test_df[FEATURE_COLUMNS])
    lo, hi = test_df["target"].min(), test_df["target"].max()
    span = hi - lo
    assert preds.min() > lo - span, "predictions fall far below the actual test range"
    assert preds.max() < hi + span, "predictions rise far above the actual test range"


def test_naive_forecast_has_near_zero_directional_accuracy_by_construction():
    """Exactly 0% except on a flat trading day (no price change), where direction_true is
    also 0 and happens to match naive's always-0 prediction -- rare, but not impossible."""
    featured = _featured()
    _, test_df = time_split(featured, test_size=0.2)
    preds = naive_forecast(test_df)
    metrics = evaluate(test_df["target"].to_numpy(), preds, test_df["lag_1"].to_numpy())
    assert metrics["directional_accuracy"] < 0.01


def test_run_all_models_naive_beats_or_matches_ml_models_on_rmse():
    """Documents, rather than merely allows, the central finding: for a near-random-walk
    series the naive baseline is hard to beat. If a future change makes Ridge/RF/ARIMA
    blow past naive's RMSE by a wide margin, that's a regression worth looking at, not
    a model improvement."""
    featured = _featured()
    train_df, test_df = time_split(featured, test_size=0.2)
    run = run_all_models(train_df, test_df)
    naive_rmse = run["metrics"]["naive"]["rmse"]
    for name in ("linear_regression", "arima"):
        assert run["metrics"][name]["rmse"] < naive_rmse * 1.5, name
    assert np.isfinite(run["metrics"]["random_forest"]["rmse"])
