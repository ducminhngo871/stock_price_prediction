"""Future-forecast sanity checks: the model actually extrapolates from the latest known
price, forward past the end of history, with dates that make sense."""

import numpy as np
import pandas as pd

from stock_price_prediction.future import all_future_forecasts, future_business_dates

TICKER = "AAPL"
HORIZON = 60


def _result():
    return all_future_forecasts(TICKER, end="2024-07-01", horizon=HORIZON)


def test_future_dates_start_after_history_and_skip_weekends():
    last = pd.Timestamp("2024-06-28")  # a Friday
    dates = future_business_dates(last, 5)
    assert dates[0] > last
    assert all(d.weekday() < 5 for d in dates)
    assert dates[0] == pd.Timestamp("2024-07-01")  # the following Monday


def test_all_models_produce_horizon_length_forecasts_starting_from_latest_price():
    result = _result()
    assert len(result["dates"]) == HORIZON
    assert result["dates"][0] > result["last_date"]
    for name, f in result["forecasts"].items():
        assert len(f["point"]) == HORIZON, name
        assert np.all(np.isfinite(f["point"])), name


def test_naive_future_is_flat_at_last_close_with_widening_band():
    result = _result()
    naive = result["forecasts"]["naive"]
    assert np.allclose(naive["point"], result["last_close"])
    width = naive["hi"] - naive["lo"]
    # a random-walk band must widen over time (sqrt(t) scaling), never narrow
    assert np.all(np.diff(width) > 0)


def test_arima_future_has_a_widening_confidence_interval():
    result = _result()
    arima = result["forecasts"]["arima"]
    assert np.all(arima["lo"] <= arima["point"])
    assert np.all(arima["point"] <= arima["hi"])
    width = arima["hi"] - arima["lo"]
    assert width[-1] > width[0]


def test_random_forest_future_forecast_flattens_out():
    """Documents the known failure mode: recursively feeding its own predictions back in,
    random forest quickly settles near a fixed point instead of tracking any real trend,
    because it can't extrapolate past the price range it was trained on."""
    result = _result()
    rf_point = result["forecasts"]["random_forest"]["point"]
    late_half = rf_point[HORIZON // 2:]
    assert late_half.std() < 0.5 * rf_point[:5].std() + 1e-6 or late_half.std() < 1.0
