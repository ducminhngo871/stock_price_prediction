"""Run the full forecasting showcase for one ticker: load data, fit all models, save plots + metrics.

Usage: PYTHONPATH=src python scripts/run_model.py [TICKER] [END_DATE]
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_price_prediction.pipeline import run_forecast  # noqa: E402
from stock_price_prediction.plotting import (  # noqa: E402
    plot_metrics_comparison,
    plot_predictions,
    plot_price_history,
)


def main(ticker: str = "AAPL", end: str | None = None) -> None:
    out_dir = ROOT / "results" / f"price_forecast_{ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_forecast(ticker, end=end, test_size=0.2)
    prices, train_df, test_df = run["prices"], run["train"], run["test"]
    split_date = test_df["date"].iloc[0]

    plot_price_history(prices, ticker, split_date=split_date, save_path=out_dir / "price_history.png")
    plot_predictions(test_df["date"], run["y_true"], run["predictions"], ticker,
                      save_path=out_dir / "predictions.png")
    plot_metrics_comparison(run["metrics"], ticker, save_path=out_dir / "model_comparison.png")

    pred_df = pd.DataFrame({"date": test_df["date"].to_numpy(), "actual": run["y_true"]})
    for name, preds in run["predictions"].items():
        pred_df[name] = preds
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps({
        "ticker": ticker,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split_date": str(split_date.date()),
        "metrics": run["metrics"],
    }, indent=2))

    print(f"{ticker}: {len(train_df)} train / {len(test_df)} test rows, split at {split_date.date()}")
    for name, m in run["metrics"].items():
        print(f"  {name:>18s}  RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  "
              f"MAPE={m['mape']:5.2f}%  dir.acc={m['directional_accuracy']:.1%}")
    print(f"-> saved to {out_dir}")


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    end_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(ticker_arg, end_arg)
