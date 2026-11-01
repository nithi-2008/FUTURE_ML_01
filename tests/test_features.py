"""
Unit Tests for Time-Based Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from src.features import create_features, get_feature_names


def test_create_features():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    y_values = np.linspace(100, 300, 60) + np.random.normal(0, 10, 60)
    df = pd.DataFrame({"ds": dates, "y": y_values})

    featured = create_features(df, date_col="ds", target_col="y", drop_na=True)

    # Check calendar features
    assert "day_of_week" in featured.columns
    assert "month" in featured.columns
    assert "quarter" in featured.columns
    assert "is_weekend" in featured.columns

    # Check cyclical features
    assert "sin_day_of_week" in featured.columns
    assert "cos_day_of_week" in featured.columns

    # Check lag & rolling features
    assert "lag_1" in featured.columns
    assert "lag_7" in featured.columns
    assert "rolling_mean_7" in featured.columns

    # Ensure no NaN values remain
    assert not featured.isnull().any().any()

    # Feature names helper
    names = get_feature_names(featured)
    assert "ds" not in names
    assert "y" not in names
    assert len(names) > 15
