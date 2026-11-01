"""
Unit Tests for Model Arena and Time-Series Forecasting
"""

import pytest
import pandas as pd
import numpy as np
from src.features import create_features
from src.models import TimeSeriesForecaster, ModelArena
from src.evaluator import calculate_metrics


def test_model_training_and_metrics():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    y_values = 500 + np.sin(np.arange(120) * 2 * np.pi / 7) * 50 + np.random.normal(0, 10, 120)
    df = pd.DataFrame({"ds": dates, "y": y_values})

    arena = ModelArena(test_days=20)
    leaderboard = arena.train_and_benchmark(df, model_names=["HistGradientBoosting", "Ridge Regression"])

    assert len(leaderboard) >= 2
    assert "WAPE (%)" in leaderboard.columns
    assert "RMSE ($)" in leaderboard.columns
    assert arena.best_model_name is not None

    best = arena.get_best_model()
    forecast = best.forecast_future(df, horizon_days=14)
    assert len(forecast) == 14
    assert "y_pred" in forecast.columns
    assert "y_lower" in forecast.columns
    assert "y_upper" in forecast.columns
    assert (forecast["y_pred"] >= 0).all()
