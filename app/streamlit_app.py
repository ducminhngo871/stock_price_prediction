"""Stock price forecasting showcase: pick a ticker, run every model, inspect and compare results.

Run with:  streamlit run app/streamlit_app.py   (from the project root, so .streamlit/config.toml applies)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from stock_price_prediction.data import available_tickers  # noqa: E402
from stock_price_prediction.future import all_future_forecasts  # noqa: E402
from stock_price_prediction.ideal import IDEAL  # noqa: E402
from stock_price_prediction.pipeline import run_forecast  # noqa: E402
from stock_price_prediction.plotting import MODEL_COLORS, MODEL_LABELS  # noqa: E402
from stock_price_prediction.specs import SPECS, recipe_name, run_id, spec_check  # noqa: E402

LOT = "SPX-0001"
MODEL_NAMES = ["naive", "linear_regression", "random_forest", "arima"]
HORIZON_OPTIONS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252, "2 years": 504}

st.set_page_config(page_title="Stock Price Forecasting", page_icon="\U0001F4C8", layout="wide")
st.html("""<style>
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap");
  h1, h2, h3, h4, [data-baseweb="tab"], [data-testid="stMetricValue"] { font-family: "JetBrains Mono", ui-monospace, monospace !important; }
  [data-testid="stSidebar"] { border-right: 1px solid #243140; }
  .recipe { font-family: "JetBrains Mono", monospace; font-size: 12px; color: #38d6e8; word-break: break-all; }
</style>""")

# ---------------------------------------------------------------- state
ALL_TICKERS = available_tickers()
DEFAULTS = {"ticker": "AAPL", "test_size_pct": 20, "use_latest": True, "end_date": None}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
    st.session_state[key] = st.session_state[key]
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_run", None)
st.session_state.setdefault("future_run", None)


def ideal_settings() -> dict:
    return {"ticker": IDEAL["settings"]["ticker"], "test_size_pct": int(IDEAL["settings"]["test_size"] * 100)}


def load_ideal():
    st.session_state.update(ideal_settings())


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Forecast config")
    ticker = st.selectbox("Ticker", ALL_TICKERS, key="ticker")
    st.button("⭐ Load recommended", key="ideal_button", on_click=load_ideal, width="stretch",
              help="Load the config that best demonstrates this project's main finding")

    test_size_pct = st.slider("Test period (% of history, most recent)", 10, 40, step=5, key="test_size_pct")

    with st.expander("Advanced"):
        use_latest = st.checkbox("Use latest available data", key="use_latest")
        end_date = None if use_latest else st.date_input("As of date", key="end_date_input")
        st.caption("Models: naive, Ridge regression, random forest, ARIMA(5,1,0) -- all four run together.")

    is_ideal = all(st.session_state[k] == v for k, v in ideal_settings().items())
    recipe = recipe_name(ticker, "2013", "latest" if use_latest else str(end_date), test_size_pct / 100)
    st.markdown(f'<div class="recipe">{"⭐ recommended · " if is_ideal else ""}{recipe}</div>',
                unsafe_allow_html=True)
    st.caption(f"Specs: {SPECS['beats_naive_rmse']['text']}; {SPECS['directional_accuracy']['text']}")
    run = st.button("▶ Run forecast", key="run_button", type="primary", width="stretch")
    split = st.button(
        "✓ Config matches recommended" if is_ideal else "⇄ Compare with recommended",
        key="split_button", width="stretch", disabled=is_ideal,
        help=("This config already matches the recommended one. Change ticker or test period to compare."
              if is_ideal else "Run this config and the recommended one side by side"),
    )

    st.markdown("### Future forecast")
    st.caption("Projects past the latest known price -- a different exercise from the backtest above: "
              "there's no real outcome to score this against, only a projection (see the FUTURE tab).")
    horizon_label = st.selectbox("Horizon", list(HORIZON_OPTIONS), index=3, key="horizon_label")
    forecast_future = st.button("\U0001F52E Forecast future", key="future_button", width="stretch")


# ---------------------------------------------------------------- simulation
@st.cache_data(show_spinner="Fetching prices and fitting models...", max_entries=32, ttl=3600)
def simulate(ticker: str, end: str | None, test_size: float) -> dict:
    return run_forecast(ticker, end=end, test_size=test_size)


@st.cache_data(show_spinner="Fetching prices and fitting models on full history...", max_entries=32, ttl=3600)
def simulate_future(ticker: str, end: str | None, horizon: int) -> dict:
    return all_future_forecasts(ticker, end=end, horizon=horizon)


def current_config() -> dict:
    end = None if use_latest else str(end_date)
    return {"ticker": ticker, "end": end, "test_size": test_size_pct / 100}


def ideal_config(base: dict | None = None) -> dict:
    base = current_config() if base is None else dict(base)
    s = ideal_settings()
    return {**base, "ticker": s["ticker"], "test_size": s["test_size_pct"] / 100}


def process_run(cfg: dict, label: str) -> dict:
    """Simulate, spec-check every model, log one history row per model; return the display payload."""
    result = simulate(**cfg)
    naive_metrics = result["metrics"]["naive"]
    for model_name in MODEL_NAMES:
        m = result["metrics"][model_name]
        status = spec_check(m, naive_metrics)
        st.session_state["history"].append({
            "run": run_id(LOT, len(st.session_state["history"]) + 1),
            "ticker": cfg["ticker"], "recipe": recipe_name(cfg["ticker"], "2013", cfg["end"] or "latest",
                                                            cfg["test_size"]),
            "label": label, "model": model_name, "result": status,
            "rmse": m["rmse"], "mae": m["mae"], "mape": m["mape"], "dir_acc": m["directional_accuracy"],
        })
    return {"cfg": cfg, "label": label, "result": result}


def comparison_table(runs: list[dict]) -> pd.DataFrame:
    current, reference = runs
    rows = []
    for model_name in MODEL_NAMES:
        cm, rm = current["result"]["metrics"][model_name], reference["result"]["metrics"][model_name]
        rows.append((MODEL_LABELS[model_name], f"{cm['rmse']:.3f}", f"{rm['rmse']:.3f}", f"{cm['rmse']-rm['rmse']:+.3f}"))
    return pd.DataFrame(rows, columns=["Model", "Your config (RMSE)", "Recommended (RMSE)", "Difference"])


if run or split:
    try:
        cfg_snapshot = current_config()
        runs = [process_run(cfg_snapshot, "RECIPE" if split else "RECOMMENDED" if is_ideal else "RECIPE")]
        if split:
            runs.append(process_run(ideal_config(cfg_snapshot), "RECOMMENDED"))
        st.session_state["last_run"] = {"runs": runs, "comparison": split}
    except ValueError as exc:
        st.error(str(exc))

if forecast_future:
    try:
        end = None if use_latest else str(end_date)
        st.session_state["future_run"] = simulate_future(ticker, end, HORIZON_OPTIONS[horizon_label])
    except ValueError as exc:
        st.error(str(exc))

hist = pd.DataFrame(st.session_state["history"])

# ---------------------------------------------------------------- page
st.markdown("### Stock price forecasting")
tab_run, tab_future, tab_hist, tab_about = st.tabs(["RUN", "FUTURE", "RUN HISTORY", "ABOUT"])

with tab_run:
    last = st.session_state["last_run"]
    if last is None:
        st.info("Pick a ticker in the sidebar (or press **⭐ Load recommended**), then **▶ Run forecast**. "
                "**⇄ Compare with recommended** runs your config and the recommended one side by side.")
    else:
        main_run = last["runs"][0]
        prices, train_df, test_df = main_run["result"]["prices"], main_run["result"]["train"], main_run["result"]["test"]
        split_date = test_df["date"].iloc[0]
        t = main_run["cfg"]["ticker"]

        st.markdown(f"#### {t}: price history, {prices['date'].min():%Y-%m-%d} to {prices['date'].max():%Y-%m-%d}")
        price_chart = alt.Chart(prices).mark_line(color="#38d6e8", size=1).encode(
            x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="Close ($)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("close:Q", format=".2f")],
        ).properties(height=260).interactive()
        split_rule = alt.Chart(pd.DataFrame({"date": [split_date]})).mark_rule(
            color="#ff5a4e", strokeDash=[6, 4]).encode(x="date:T")
        st.altair_chart((price_chart + split_rule).configure(background="transparent")
                        .configure_axis(gridColor="#243140", labelColor="#7b8a99", titleColor="#7b8a99"),
                        width="stretch")
        st.caption("Dashed line: where the train/test split falls. Zoom/pan and hover for exact values.")

        st.markdown("#### Test-period forecasts (scrub through the test period)")
        shown_models = st.multiselect("Models to show", MODEL_NAMES, default=MODEL_NAMES,
                                      format_func=lambda m: MODEL_LABELS[m], key="shown_models")
        day = st.slider("Day", 1, len(test_df), value=len(test_df), key="replay_day")

        window = test_df.iloc[:day][["date"]].copy()
        window["Actual"] = main_run["result"]["y_true"][:day]
        for m in shown_models:
            window[MODEL_LABELS[m]] = main_run["result"]["predictions"][m][:day]
        long = window.melt("date", var_name="series", value_name="price")
        domain = ["Actual"] + [MODEL_LABELS[m] for m in shown_models]
        range_ = ["#e8e8e8"] + [MODEL_COLORS[m] for m in shown_models]
        pred_chart = alt.Chart(long).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="Close ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title=None)),
            size=alt.condition(alt.datum.series == "Actual", alt.value(2.2), alt.value(1.2)),
            tooltip=["date:T", "series:N", alt.Tooltip("price:Q", format=".2f")],
        ).properties(height=320).configure(background="transparent").configure_axis(
            gridColor="#243140", labelColor="#7b8a99", titleColor="#7b8a99")
        st.altair_chart(pred_chart, width="stretch")

        st.markdown("#### Model metrics (full test period)")
        rows = []
        naive_m = main_run["result"]["metrics"]["naive"]
        for m in MODEL_NAMES:
            met = main_run["result"]["metrics"][m]
            rows.append({"Model": MODEL_LABELS[m], "RMSE ($)": round(met["rmse"], 3),
                        "MAE ($)": round(met["mae"], 3), "MAPE (%)": round(met["mape"], 2),
                        "Directional acc. (%)": round(met["directional_accuracy"] * 100, 1),
                        "Spec result": spec_check(met, naive_m)})
        metrics_df = pd.DataFrame(rows)
        styled = metrics_df.style.format({
            "RMSE ($)": "{:.3f}", "MAE ($)": "{:.3f}", "MAPE (%)": "{:.2f}", "Directional acc. (%)": "{:.1f}",
        }).map(lambda v: {"PASS": "color:#3ddc84", "FAIL": "color:#ff5a4e"}.get(v, ""), subset=["Spec result"])
        st.dataframe(styled, hide_index=True, width="stretch")

        if last.get("comparison"):
            st.markdown("#### Your config vs. recommended")
            st.dataframe(comparison_table(last["runs"]), hide_index=True, width="stretch")

        with st.popover("⬇ Export"):
            pred_export = test_df[["date"]].copy()
            pred_export["actual"] = main_run["result"]["y_true"]
            for m in MODEL_NAMES:
                pred_export[m] = main_run["result"]["predictions"][m]
            st.download_button("Predictions (CSV)", pred_export.to_csv(index=False),
                               f"{t}_predictions.csv", key="export_csv")
            st.download_button("Metrics (JSON)", json.dumps(rows, indent=2), f"{t}_metrics.json",
                               key="export_json")
            st.download_button("Run history (CSV)", hist.to_csv(index=False) if not hist.empty else "",
                               f"{LOT}_history.csv", key="export_hist")

with tab_future:
    fr = st.session_state["future_run"]
    if fr is None:
        st.info("Pick a ticker and horizon in the sidebar, then **\U0001F52E Forecast future**. Unlike RUN, "
                "this projects past the latest known price into dates that haven't happened yet -- there's "
                "no real outcome to score it against, only a projection.")
    else:
        ft = fr["ticker"]
        st.markdown(f"#### {ft}: {len(fr['dates'])}-trading-day forecast past {fr['last_date']:%Y-%m-%d} "
                   f"(last close ${fr['last_close']:.2f})")

        tail = fr["prices"].tail(252)[["date", "close"]].rename(columns={"close": "price"})
        tail["series"] = "Actual (history)"
        layers = [alt.Chart(tail).mark_line(color="#e8e8e8", size=1.6).encode(
            x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="Close ($)", scale=alt.Scale(zero=False)))]

        for name in MODEL_NAMES:
            f = fr["forecasts"][name]
            color = MODEL_COLORS[name]
            proj = pd.DataFrame({"date": fr["dates"], "price": f["point"]})
            layers.append(alt.Chart(proj).mark_line(color=color, strokeDash=[5, 3], size=1.6).encode(
                x="date:T", y="price:Q",
                tooltip=["date:T", alt.Tooltip("price:Q", format=".2f")]))
            if "lo" in f:
                band = pd.DataFrame({"date": fr["dates"], "lo": f["lo"], "hi": f["hi"]})
                layers.insert(0, alt.Chart(band).mark_area(color=color, opacity=0.08).encode(
                    x="date:T", y="lo:Q", y2="hi:Q"))

        chart = (alt.layer(*layers).properties(height=380).configure(background="transparent")
                .configure_axis(gridColor="#243140", labelColor="#7b8a99", titleColor="#7b8a99"))
        st.altair_chart(chart, width="stretch")
        legend = "  ".join(f"<span style='color:{MODEL_COLORS[m]}'>&#9679;</span> {MODEL_LABELS[m]}"
                           for m in MODEL_NAMES)
        st.markdown(f"<span style='color:#e8e8e8'>&#9679;</span> Actual (history)  {legend}", unsafe_allow_html=True)
        st.caption("Shaded bands (naive, ARIMA) are 95% uncertainty ranges, not alternate point forecasts. "
                  "Random forest and Ridge have no native uncertainty estimate here.")

        rows = []
        for name in MODEL_NAMES:
            f = fr["forecasts"][name]
            band = f"${f['lo'][-1]:.2f} - ${f['hi'][-1]:.2f}" if "lo" in f else "n/a"
            rows.append({"Model": MODEL_LABELS[name], f"Price in {len(fr['dates'])} trading days": round(float(f["point"][-1]), 2),
                        "95% range": band})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.warning(
            "**This is an extrapolation, not a prediction with a known error rate.** There is no actual "
            "price to compare against yet, so none of the RUN tab's metrics or specs apply here. Random "
            "forest and Ridge are forecast recursively (each day's prediction feeds the next day's "
            "features), which compounds error with every step -- random forest in particular is known to "
            "flatten out fast past its training range (see ABOUT). Naive's and ARIMA's bands widen with "
            "the horizon for a reason: a 1-year-ahead point forecast for a near-random-walk stock means "
            "very little without one.",
            icon="⚠️",
        )

        with st.popover("⬇ Export"):
            fdf = pd.DataFrame({"date": fr["dates"]})
            for name in MODEL_NAMES:
                f = fr["forecasts"][name]
                fdf[f"{name}_point"] = f["point"]
                if "lo" in f:
                    fdf[f"{name}_lo"], fdf[f"{name}_hi"] = f["lo"], f["hi"]
            st.download_button("Future forecast (CSV)", fdf.to_csv(index=False),
                               f"{ft}_future_forecast.csv", key="export_future_csv")

COLOR = alt.Scale(domain=["PASS", "FAIL"], range=["#3ddc84", "#ff5a4e"])


def trend_chart(df: pd.DataFrame, metric: str, title: str, axis_format: str = ""):
    base = alt.Chart(df).encode(x=alt.X("run:N", sort=None, title=None))
    y_axis = alt.Axis(format=axis_format) if axis_format else alt.Undefined
    layers = [base.mark_line(color="#4b5866").encode(y=alt.Y(f"{metric}:Q", title=title, axis=y_axis)),
              base.mark_circle(size=70, opacity=1).encode(
                  y=f"{metric}:Q", color=alt.Color("result:N", scale=COLOR, legend=None),
                  tooltip=["run", "ticker", "model", "label", alt.Tooltip(f"{metric}:Q", format=".3f"), "result"])]
    return (alt.layer(*layers).properties(height=220).configure(background="transparent")
            .configure_axis(gridColor="#243140", labelColor="#7b8a99", titleColor="#7b8a99", domainColor="#243140"))


with tab_hist:
    if hist.empty:
        st.info("No runs yet.")
    else:
        st.caption(f"Lot {LOT} · {hist['run'].nunique()} runs · {len(hist)} model results · "
                  f"{int((hist['result']=='PASS').sum())} pass, {int((hist['result']=='FAIL').sum())} fail")
        mod = hist[hist["model"] == st.selectbox("Model to trend", MODEL_NAMES,
                                                  format_func=lambda m: MODEL_LABELS[m], key="trend_model")]
        if not mod.empty:
            a, b = st.columns(2)
            a.altair_chart(trend_chart(mod, "rmse", "RMSE ($)"), width="stretch")
            b.altair_chart(trend_chart(mod, "dir_acc", "Directional accuracy", axis_format=".0%"), width="stretch")
        show = hist.copy()
        show["model"] = show["model"].map(MODEL_LABELS)
        for col, fmt in {"rmse": "{:.3f}", "mae": "{:.3f}", "mape": "{:.2f}", "dir_acc": "{:.3f}"}.items():
            show[col] = show[col].map(fmt.format)
        st.dataframe(show.style.map(lambda v: {"PASS": "color:#3ddc84", "FAIL": "color:#ff5a4e"}.get(v, ""),
                                    subset=["result"]), hide_index=True, width="stretch")
        if st.button("Clear history", key="clear_button"):
            st.session_state["history"], st.session_state["last_run"] = [], None
            st.rerun()

with tab_about:
    st.markdown(f"""
**Models.** Naive (predicts tomorrow = today, the random-walk baseline); Ridge regression on lag/rolling-mean
features (plain OLS was tried first and collapsed to a near-constant prediction because those features are
correlated above 0.99 -- Ridge's regularization fixes that); random forest on the same features; ARIMA(5,1,0), a
rolling one-step-ahead forecast on the close series directly.

**Metrics.** RMSE/MAE/MAPE measure price-level error. Directional accuracy is whether the model got tomorrow's
up/down right -- a coin flip is 50%, and the naive model sits at ~0% by construction (it only ever predicts
"no change", so the rare flat trading day aside, it never has a direction to be right or wrong about).

**Specs.** {SPECS['beats_naive_rmse']['text']}. {SPECS['directional_accuracy']['text']}. Expect most models to
FAIL most of the time -- that's the project's central, honest finding: liquid stock prices are close enough to
a random walk that "tomorrow equals today" is a hard baseline to beat, and none of these models finds a
reliable directional edge either.

**Recommended config.** {IDEAL['why']}

**Known limitation.** Random forest cannot extrapolate past the highest price value it saw during training --
visible directly in the test-period chart above as a flat line once the actual price breaks into new territory.
This is a structural property of tree ensembles on trending series, not a tuning problem.

**The FUTURE tab** fits every model on *all* available history and projects past today, instead of holding
out a known test period. There's no real outcome to score these against, so no RMSE or spec applies there --
only the projection and, where available, an uncertainty band. Naive's band comes from historical volatility
scaled by sqrt(time), the standard random-walk result; ARIMA's comes from its own fitted model. Notice ARIMA's
band is usually *narrower* than naive's despite ARIMA backtesting worse (see Findings) -- a model's own
parametric confidence interval isn't automatically more trustworthy than a simple empirical one. Random
forest and Ridge have no native uncertainty estimate and are forecast recursively (each day's prediction
feeds the next day's lag features), which compounds error with every step.
""")
