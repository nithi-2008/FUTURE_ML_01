"""
Time-Based Feature Engineering Module
Constructs calendar, cyclical, lag, rolling, and trend features for forecasting models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def create_features(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    lags: List[int] = [1, 2, 7, 14, 21, 30],
    rolling_windows: List[int] = [7, 14, 30],
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Creates rich time-series features from a dataframe containing a datetime column and a target value.
    
    Features created:
    1. Calendar: year, quarter, month, day, day_of_week, day_of_year, week_of_year,
                 is_weekend, is_month_start, is_month_end, is_quarter_end
    2. Cyclical encodings: sin/cos transforms for day_of_week, month, day_of_year
    3. Trend: integer linear step
    4. Lags: historical values at specified offsets (e.g. lag_1, lag_7)
    5. Rolling stats: rolling mean, std, min, max over sliding windows (shifted to prevent leakage!)
    """
    df = df.copy()

    # Ensure datetime
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).reset_index(drop=True)

    dt = df[date_col].dt

    # 1. Calendar Features
    df["year"] = dt.year
    df["quarter"] = dt.quarter
    df["month"] = dt.month
    df["day"] = dt.day
    df["day_of_week"] = dt.day_of_week
    df["day_of_year"] = dt.day_of_year
    # In newer pandas, isocalendar().week is used
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["is_weekend"] = dt.day_of_week.isin([5, 6]).astype(int)
    df["is_month_start"] = (dt.day <= 3).astype(int)
    df["is_month_end"] = (dt.is_month_end | (dt.day >= 28)).astype(int)
    df["is_quarter_end"] = (dt.is_quarter_end).astype(int)

    # 2. Cyclical Encodings (sin / cos)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # 3. Monotonic Trend Feature
    df["trend_step"] = np.arange(len(df))

    # 4. Lag Features
    # IMPORTANT: Lags must be computed strictly on historical target values
    if target_col in df.columns:
        for lag in lags:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        # 5. Rolling Statistics
        # Note: shift(1) is CRITICAL to prevent data leakage (only use past observed values!)
        shifted_y = df[target_col].shift(1)
        for w in rolling_windows:
            df[f"rolling_mean_{w}"] = shifted_y.rolling(window=w, min_periods=1).mean()
            df[f"rolling_std_{w}"] = shifted_y.rolling(window=w, min_periods=1).std().fillna(0.0)
            df[f"rolling_max_{w}"] = shifted_y.rolling(window=w, min_periods=1).max()
            df[f"rolling_min_{w}"] = shifted_y.rolling(window=w, min_periods=1).min()

        # Exponential moving averages
        df["ema_7"] = shifted_y.ewm(span=7, adjust=False).mean()
        df["ema_14"] = shifted_y.ewm(span=14, adjust=False).mean()

    if drop_na:
        # Drop initial rows where lags are undefined
        max_lag = max(lags) if lags else 0
        df = df.iloc[max_lag:].reset_index(drop=True)

    return df


def get_feature_names(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Returns list of predictor feature columns, excluding identifier or target columns.
    """
    if exclude_cols is None:
        exclude_cols = ["ds", "y", "Order Date", "Sales", "orders_count", "Quantity", "Profit"]
    return [col for col in df.columns if col not in exclude_cols]
