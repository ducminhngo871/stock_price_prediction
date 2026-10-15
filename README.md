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
data/raw/       # not committed; kaggle_sandp500/ + yfinance/ (fetched, cached)
notebooks/      # exploration
scripts/        # reproducible data/model pipelines
tests/          # data integrity tests
results/        # model outputs, figures
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Kaggle data must be downloaded once into `data/raw/kaggle_sandp500/all_stocks_5yr.csv`
(`kaggle datasets download -d camnugent/sandp500 -f all_stocks_5yr.csv --unzip`).
yfinance data fetches and caches automatically on first use.

## Status

Data pipeline validated. Modeling (price forecasting) not yet started.
