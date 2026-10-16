"""Forecast past the end of known history for one ticker: fit every model on all available
data, project forward, save a plot + the raw numbers.

Usage: PYTHONPATH=src python scripts/run_future.py [TICKER] [HORIZON_DAYS] [END_DATE]
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_price_prediction.future import all_future_forecasts  # noqa: E402
from stock_price_prediction.plotting import plot_future_forecast  # noqa: E402


def main(ticker: str = "AAPL", horizon: int = 252, end: str | None = None) -> None:
    out_dir = ROOT / "results" / f"future_forecast_{ticker}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = all_future_forecasts(ticker, end=end, horizon=horizon)
    plot_future_forecast(result["prices"], result, ticker, save_path=out_dir / "forecast.png")

    out = pd.DataFrame({"date": result["dates"]})
    for name, f in result["forecasts"].items():
        out[f"{name}_point"] = f["point"]
        if "lo" in f:
            out[f"{name}_lo"], out[f"{name}_hi"] = f["lo"], f["hi"]
    out.to_csv(out_dir / "forecast.csv", index=False)

    (out_dir / "settings.json").write_text(json.dumps({
        "ticker": ticker, "horizon": horizon, "last_date": str(result["last_date"].date()),
        "last_close": result["last_close"],
    }, indent=2))

    print(f"{ticker}: last known close {result['last_close']:.2f} on {result['last_date'].date()}")
    for name, f in result["forecasts"].items():
        tail = f"[{f['lo'][-1]:.2f}, {f['hi'][-1]:.2f}]" if "lo" in f else "(no band)"
        print(f"  {name:>18s}  {horizon}-day forecast = {f['point'][-1]:7.2f}  {tail}")
    print(f"-> saved to {out_dir}")


if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    horizon_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 252
    end_arg = sys.argv[3] if len(sys.argv) > 3 else None
    main(ticker_arg, horizon_arg, end_arg)
