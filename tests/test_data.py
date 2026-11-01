"""
Unit Tests for Data Ingestion, Cleaning, and Aggregation
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    load_superstore_data,
    clean_sales_data,
    aggregate_time_series,
    generate_synthetic_sales_data,
    validate_uploaded_csv
)


def test_clean_sales_data():
    raw_dict = {
        "Order Date": ["2023-01-01", "2023-01-02", None, "2023-01-03", "invalid_date"],
        "Sales": ["$100.50", "200.00", 200, None, "50"],
        "Quantity": [2, 1, 1, 3, 4]
    }
    raw_df = pd.DataFrame(raw_dict)
    cleaned = clean_sales_data(raw_df)
    assert len(cleaned) == 2
    assert cleaned["Sales"].iloc[0] == 100.50
    assert cleaned["Sales"].iloc[1] == 200.00
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Order Date"])


def test_aggregate_time_series():
    synth = generate_synthetic_sales_data(days=90, random_seed=42)
    daily = aggregate_time_series(synth, freq="Daily")
    assert len(daily) == 90
    assert "ds" in daily.columns
    assert "y" in daily.columns
    assert (daily["y"] >= 0).all()


def test_validate_uploaded_csv():
    good_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Sales": [150.0, 200.0]
    })
    is_valid, msg, date_col, sales_col = validate_uploaded_csv(good_df)
    assert is_valid is True
    assert date_col == "Date"
    assert sales_col == "Sales"

    bad_df = pd.DataFrame({"ColA": [1, 2], "ColB": [3, 4]})
    is_valid_bad, msg_bad, _, _ = validate_uploaded_csv(bad_df)
    assert is_valid_bad is False
