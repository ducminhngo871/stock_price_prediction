# Stock Price Prediction

A price-forecasting project combining a Kaggle historical dump with live
yfinance data to get a long, clean, up-to-date daily OHLCV series.

## Data sources

* **Kaggle: [`camnugent/sandp500`](https://www.kaggle.com/datasets/camnugent/sandp500)**
  — daily OHLCV for 505 S&P 500 tickers, 2013-02-08 to 2018-02-07. Verified
  clean (0 tickers with missing data). Chosen over the more recent
  `andrewmvd/sp-500-stocks` dataset, which turned out to have 330 of 502
  tickers (66%) entirely null, including AAPL — unusable as a backbone.
* **yfinance** — extends the series from 2018-02-08 to the present for the
  same tickers, fetched live and cached locally under `data/raw/yfinance/`.

Both are loaded in `src/stock_price_prediction/data.py::load_combined`.

### The split-adjustment bug (read before trusting the data)

yfinance's OHLC columns are always back-adjusted for every stock split up to
today, regardless of the `auto_adjust` flag (that flag only controls dividend
adjustment). The Kaggle file was scraped around its 2018-02-07 cutoff, so its
prices are back-adjusted only for splits that had already happened by then
(e.g. AAPL's 2014 7-for-1), not later ones (e.g. AAPL's 2020 4-for-1, NVDA's
2021 4-for-1 and 2024 10-for-1).

Naively concatenating the two sources, or re-adjusting both for full split
history, creates a fake multi-x price jump exactly at the 2018-02-08 stitch
point. `load_combined` instead adjusts only the Kaggle rows, and only for
splits dated after the Kaggle cutoff. This was caught by inspecting the merged
AAPL series before building anything on top of it, and is now locked in by
`tests/test_data.py` (checks for no >15% jump at the stitch date or at known
NVDA split dates).

## Layout

```
src/stock_price_prediction/
  data.py       # load_kaggle, fetch_yfinance, load_combined (+ split adjustment)
  features.py   # lag/rolling features, next-day target, chronological train/test split
  model.py      # naive, Ridge, random forest, ARIMA + evaluation (RMSE/MAE/MAPE/directional accuracy)
  pipeline.py   # run_forecast: data -> features -> split -> all models, shared by the script and the app
  plotting.py   # static (matplotlib) plots used by scripts/run_model.py
  specs.py       # app vocabulary: run IDs, recipe names, pass/fail specs
  ideal.py      # recommended demo config + reasoning
data/raw/       # not committed; kaggle_sandp500/ + yfinance/ (fetched, cached)
notebooks/      # exploration
scripts/        # run_model.py: reproducible end-to-end data+model pipeline
app/            # streamlit_app.py: interactive ticker/model picker, run/compare, run history
.streamlit/     # dark theme
tests/          # data integrity, model sanity and app smoke tests
results/        # per-ticker model outputs: figures, predictions.csv, metrics.json
```

## Models (`scripts/run_model.py`)

Four next-day close-price forecasters, in increasing complexity, each backtested on a
chronological (no-shuffle) 80/20 train/test split so no future price ever leaks into training:

| Model | What it does | Why it's here |
|---|---|---|
| Naive (random walk) | predicts tomorrow = today | the baseline every other model must beat; liquid stock prices are close to a random walk, so this is a genuinely hard baseline |
| Ridge regression | lag/rolling-mean features, L2-regularized | plain OLS was tried first and **collapsed to a near-constant prediction** (~$65-70 against an actual range of $125-198) because the lag and rolling-mean features are >0.99 correlated, making the OLS normal equations numerically degenerate. Ridge's regularization fixes this without dropping features. |
| Random forest | same features, 300 trees, depth 6 | nonlinear baseline for comparison |
| ARIMA(5,1,0) | rolling one-step-ahead forecast on the close series itself | classic time-series model, no engineered features |

Run with `PYTHONPATH=src python scripts/run_model.py AAPL` (any ticker, optional end date).
Saves to `results/price_forecast_<TICKER>/`: `price_history.png`, `predictions.png`,
`model_comparison.png`, `predictions.csv`, `metrics.json`.

### Findings (AAPL and AMZN, 2013-2024)

* **Naive is nearly unbeatable on price-level RMSE.** Ridge and ARIMA land within ~1-2% of
  naive's RMSE; none of the models meaningfully beat "tomorrow equals today." This is the
  expected signature of a near-random-walk series, not a failure to tune the models harder.
* **Random forest is the worst model, consistently.** Its RMSE is 4-5x naive's. The reason is
  visible directly in `predictions.png`: it cannot extrapolate past the highest price value it
  saw during training (e.g. AAPL: capped at ~$153 while the actual test-period price reached
  $198), a structural limitation of tree ensembles on trending, non-stationary series.
* **Directional accuracy sits at ~50% for every model** (a coin flip), and is *exactly* 0% for
  naive by construction -- it only ever predicts "no change," so it never has a direction to be
  right or wrong about. None of the feature-based models find a genuine edge on up/down direction
  either.
* **Takeaway:** this is an honest negative result, consistent with the efficient-market
  hypothesis, not a bug to chase. Price-level forecasts track the market well enough to *look*
  good in a plot (see `predictions.png`), but that's naive's doing, not genuine predictive skill.

## The app

`streamlit run app/streamlit_app.py` (from the project root, so `.streamlit/config.toml` applies).

* **Sidebar config:** ticker (any of the 505 in the Kaggle dataset), test-period size, an
  **⭐ Load recommended** button (AAPL, 20% test, see Findings above for why), and an **Advanced**
  panel to pin an "as of" date instead of always using the latest available data.
* **▶ Run forecast** fits all four models and shows: the full price history (interactive
  zoom/pan/hover, train/test split marked); a scrubbable test-period chart (the `Day` slider
  reveals actual vs. predicted day by day, defaulting to the full period); a metrics table with a
  PASS/FAIL spec column; CSV/JSON export.
* **⇄ Compare with recommended** runs your config and the recommended one side by side with an
  explicit RMSE-difference table. Disabled when they already match.
* **RUN HISTORY:** every (run, model) result gets an ID (`SPX-0001-R03`), logged with PASS/FAIL and
  a trend chart. Expect it to fill up with FAIL -- see Findings.
* **ABOUT:** the same findings/limitations as this README, in-app.

Specs (`src/stock_price_prediction/specs.py`) are deliberately hard to pass: RMSE must beat naive,
and directional accuracy must clear 55%. Both usually FAIL, which is the point (see Findings).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`data/raw/kaggle_sandp500/all_stocks_5yr.csv` (the Kaggle file, CC0-1.0 licensed, ~28 MB) is
committed to the repo so a fresh checkout -- including a Streamlit Community Cloud deploy --
works with no Kaggle credentials and no manual download step. Everything else under `data/raw/`
(the yfinance cache) is regenerated on first use and gitignored. To refetch the Kaggle file
yourself instead: `kaggle datasets download -d camnugent/sandp500 -f all_stocks_5yr.csv --unzip`.

## Deploy your own copy

The app needs no API keys or secrets; the Kaggle CSV it depends on is committed (see above).

1. Fork this repository on GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the GitHub account
   that owns the fork.
3. **Create app** -> paste the GitHub URL of `app/streamlit_app.py` on your fork's `main` branch
   (e.g. `https://github.com/<you>/stock_price_prediction/blob/main/app/streamlit_app.py`).
4. Keep Python 3.12 in Advanced settings. No secrets required.
5. Deploy. Pushes to `main` update the deployed app automatically.

## Status

Data pipeline, forecasting models and the Streamlit app are built and validated (see Findings
above; the app was driven end-to-end with Playwright and its logic covered by
`streamlit.testing.v1.AppTest` in `tests/test_app.py`). Not yet done: a notebook walkthrough, a
research report, a user guide, and a developer guide; extending beyond next-day-close (e.g.
multi-day horizons, other targets like returns/volatility).
