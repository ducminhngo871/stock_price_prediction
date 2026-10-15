"""Streamlit app: loads, runs a forecast, recommended-config loading, comparison and history all work."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from stock_price_prediction.ideal import IDEAL

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def fresh() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180).run()


def test_app_loads_idle():
    at = fresh()
    assert not at.exception
    assert at.session_state["last_run"] is None


def test_run_forecast_populates_history():
    at = fresh()
    at.button(key="run_button").click().run()
    assert not at.exception
    assert at.session_state["last_run"] is not None
    # one history row per model (4 models)
    assert len(at.session_state["history"]) == 4
    assert {row["model"] for row in at.session_state["history"]} == \
        {"naive", "linear_regression", "random_forest", "arima"}


def test_recommended_button_sets_config():
    at = fresh()
    at.selectbox(key="ticker").set_value("MSFT").run()
    at.button(key="ideal_button").click().run()
    assert at.selectbox(key="ticker").value == IDEAL["settings"]["ticker"]
    assert at.slider(key="test_size_pct").value == int(IDEAL["settings"]["test_size"] * 100)


def test_compare_runs_both_configs():
    at = fresh()
    at.selectbox(key="ticker").set_value("MSFT").run()
    at.button(key="split_button").click().run()
    assert not at.exception
    runs = at.session_state["last_run"]["runs"]
    assert [r["label"] for r in runs] == ["RECIPE", "RECOMMENDED"]
    assert runs[0]["cfg"]["ticker"] == "MSFT"
    assert runs[1]["cfg"]["ticker"] == IDEAL["settings"]["ticker"]
    # 4 models x 2 configs
    assert len(at.session_state["history"]) == 8


def test_compare_is_disabled_when_config_matches_recommended():
    at = fresh()
    at.button(key="ideal_button").click().run()
    button = at.button(key="split_button")
    assert button.disabled


def test_replay_slider_controls_visible_window():
    at = fresh()
    at.button(key="run_button").click().run()
    slider = at.slider(key="replay_day")
    half = max(1, slider.value // 2)
    at.slider(key="replay_day").set_value(half).run()
    assert not at.exception
    assert at.slider(key="replay_day").value == half
