"""
Unit Tests for Business Planning, Inventory ROP, and Scenario Simulator
"""

import pytest
import pandas as pd
import numpy as np
from src.business_planner import (
    calculate_inventory_plan,
    calculate_cashflow_projections,
    calculate_staffing_plan,
    simulate_scenario
)


def test_business_inventory_calculations():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    forecast_df = pd.DataFrame({
        "ds": dates,
        "y_pred": [2000.0] * 30,
        "y_lower": [1600.0] * 30,
        "y_upper": [2400.0] * 30
    })

    inv = calculate_inventory_plan(forecast_df, lead_time_days=5, service_level=0.95)
    assert inv["avg_daily_demand"] == 2000.0
    assert inv["lead_time_demand"] == 10000.0
    assert inv["reorder_point"] >= inv["lead_time_demand"]

    cf = calculate_cashflow_projections(forecast_df)
    assert cf["total_projected_revenue"] == 60000.0
    assert cf["conservative_revenue"] == 48000.0
    assert cf["optimistic_revenue"] == 72000.0

    sim = simulate_scenario(forecast_df, marketing_uplift_pct=20.0)
    assert sim["simulated_forecast"].iloc[0] == 2400.0
    assert sim["pct_change"].iloc[0] == 20.0
