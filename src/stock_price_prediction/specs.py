"""App-facing vocabulary: run IDs, recipe names, and pass/fail specs.

The specs are deliberately hard to pass: "beats naive" and "directional accuracy
above a coin flip" are the two things a genuinely useful forecaster should do,
and the project's own finding (see README) is that none of these models
reliably does either. Expect FAIL to be the common case -- that is the honest
result, not a bug in the spec.
"""

SPECS = {
    "beats_naive_rmse": {"text": "RMSE must improve on the naive (random-walk) baseline"},
    "directional_accuracy": {"min": 0.55, "text": "Directional accuracy >= 55% (a coin flip is 50%)"},
}


def recipe_name(ticker: str, start: str, end: str, test_size: float) -> str:
    """Short, readable run ID, e.g. 'AAPL-2013:2024-20pct-test'."""
    start_year = start[:4]
    end_year = end[:4]
    return f"{ticker}-{start_year}:{end_year}-{int(test_size * 100)}pct-test"


def run_id(lot: str, number: int) -> str:
    return f"{lot}-R{number:02d}"


def spec_check(model_metrics: dict, naive_metrics: dict) -> str:
    """'PASS' only if a model beats naive's RMSE and clears the directional-accuracy bar."""
    beats_naive = model_metrics["rmse"] < naive_metrics["rmse"]
    directional_ok = model_metrics["directional_accuracy"] >= SPECS["directional_accuracy"]["min"]
    return "PASS" if (beats_naive and directional_ok) else "FAIL"
