"""Shared plots: full price history, train/test split, prediction overlays, model comparison."""

import matplotlib.pyplot as plt
import pandas as pd

MODEL_COLORS = {
    "naive": "#7b8a99",
    "linear_regression": "#3ddc84",
    "random_forest": "#f5a623",
    "arima": "#ff5a4e",
}
MODEL_LABELS = {
    "naive": "Naive (random walk)",
    "linear_regression": "Ridge regression",
    "random_forest": "Random forest",
    "arima": "ARIMA(5,1,0)",
}


def plot_price_history(df: pd.DataFrame, ticker: str, split_date=None, save_path=None):
    """Close price and volume over the full combined (Kaggle + yfinance) history."""
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_price.plot(df["date"], df["close"], color="#38d6e8", linewidth=0.9)
    ax_price.set(ylabel="Close price ($)", title=f"{ticker}: close price, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    ax_price.grid(True, alpha=0.3)

    ax_vol.bar(df["date"], df["volume"], color="#4b5866", width=1.0)
    ax_vol.set(ylabel="Volume", xlabel="Date")
    ax_vol.grid(True, alpha=0.3)

    if split_date is not None:
        for ax in (ax_price, ax_vol):
            ax.axvline(split_date, color="#ff5a4e", linestyle="--", linewidth=1, label="train / test split")
        ax_price.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_predictions(dates, y_true, predictions: dict, ticker: str, save_path=None):
    """Actual vs. predicted close price on the test period, one line per model."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_true, color="black", linewidth=1.4, label="Actual", zorder=5)
    for name, preds in predictions.items():
        ax.plot(dates, preds, color=MODEL_COLORS.get(name, "gray"), linewidth=1,
                alpha=0.85, label=MODEL_LABELS.get(name, name))
    ax.set(xlabel="Date", ylabel="Close price ($)", title=f"{ticker}: test-period forecasts vs. actual")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_future_forecast(history: pd.DataFrame, result: dict, ticker: str, history_days: int = 252,
                         save_path=None):
    """Recent history plus every model's forward projection, with bands where available.

    There is no "actual" line in the projected region -- by definition, nobody knows it yet.
    """
    tail = history.tail(history_days)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(tail["date"], tail["close"], color="black", linewidth=1.3, label="Actual (history)")

    for name, f in result["forecasts"].items():
        color = MODEL_COLORS.get(name, "gray")
        ax.plot(result["dates"], f["point"], color=color, linewidth=1.3, linestyle="--",
                label=MODEL_LABELS.get(name, name))
        if "lo" in f:
            ax.fill_between(result["dates"], f["lo"], f["hi"], color=color, alpha=0.08)

    ax.axvline(result["last_date"], color="#7b8a99", linestyle=":", linewidth=1)
    ax.set(xlabel="Date", ylabel="Close price ($)",
          title=f"{ticker}: {len(result['dates'])}-trading-day forecast past {result['last_date']:%Y-%m-%d}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_metrics_comparison(metrics: dict, ticker: str, save_path=None):
    """Bar charts of RMSE and directional accuracy across models."""
    names = list(metrics.keys())
    rmse = [metrics[n]["rmse"] for n in names]
    directional = [metrics[n]["directional_accuracy"] * 100 for n in names]
    colors = [MODEL_COLORS.get(n, "gray") for n in names]
    labels = [MODEL_LABELS.get(n, n) for n in names]

    fig, (ax_rmse, ax_dir) = plt.subplots(1, 2, figsize=(11, 4))
    ax_rmse.bar(labels, rmse, color=colors)
    ax_rmse.set(ylabel="RMSE ($)", title=f"{ticker}: price-level error (lower is better)")
    ax_rmse.tick_params(axis="x", rotation=20)
    ax_rmse.grid(True, axis="y", alpha=0.3)

    ax_dir.bar(labels, directional, color=colors)
    ax_dir.axhline(50, color="#ff5a4e", linestyle="--", linewidth=1, label="coin flip")
    ax_dir.set(ylabel="Directional accuracy (%)", title=f"{ticker}: up/down accuracy")
    ax_dir.tick_params(axis="x", rotation=20)
    ax_dir.legend(fontsize=8)
    ax_dir.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
