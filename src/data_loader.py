"""
Data Loading, Cleaning, and Aggregation Module
Handles historical sales transaction data and converts it into uniform time-series datasets.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List


def load_superstore_data(filepath: str = "data/superstore_sales.csv") -> pd.DataFrame:
    """
    Loads the Superstore Sales dataset.
    If the file does not exist locally, attempts to fetch it or returns a synthesized baseline.
    """
    if os.path.exists(filepath):
        # Read with latin1 or utf-8
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin1")
    else:
        # Fallback to direct URL or generate synthetic
        url = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
        try:
            df = pd.read_csv(url, encoding="latin1")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
        except Exception:
            df = generate_synthetic_sales_data(days=1095)
    return clean_sales_data(df)


def clean_sales_data(
    df: pd.DataFrame,
    date_col: str = "Order Date",
    sales_col: str = "Sales",
    quantity_col: str = "Quantity"
) -> pd.DataFrame:
    """
    Cleans raw transaction data:
    - Removes empty rows or missing values in key columns
    - Parses date strings into pandas datetime objects
    - Standardizes column names and formats numeric values
    - Sorts chronologically
    """
    df = df.copy()

    # Find matching date column if not exact
    if date_col not in df.columns:
        date_candidates = [c for c in df.columns if any(w in c.lower() for w in ["date", "time", "day"])]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise KeyError(f"Date column '{date_col}' not found. Available columns: {list(df.columns)}")

    # Find matching sales column if not exact
    if sales_col not in df.columns:
        sales_candidates = [c for c in df.columns if any(w in c.lower() for w in ["sale", "revenue", "demand", "amount", "total"])]
        if sales_candidates:
            sales_col = sales_candidates[0]
        else:
            raise KeyError(f"Sales/Demand column '{sales_col}' not found. Available columns: {list(df.columns)}")

    # Drop rows with null date or null sales
    df = df.dropna(subset=[date_col, sales_col]).copy()

    # Convert date to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    df = df.dropna(subset=[date_col])

    # Convert sales to numeric
    if df[sales_col].dtype == object:
        df[sales_col] = df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0.0)

    # Positive sales check
    df[sales_col] = df[sales_col].clip(lower=0.0)

    # Convert quantity if present
    if quantity_col in df.columns:
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(1.0)

    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df


def aggregate_time_series(
    df: pd.DataFrame,
    freq: str = "D",
    date_col: str = "Order Date",
    sales_col: str = "Sales",
    quantity_col: Optional[str] = "Quantity",
    profit_col: Optional[str] = "Profit",
    filter_category: Optional[str] = None,
    filter_region: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregates transaction data into uniform time series (Daily 'D', Weekly 'W-MON', Monthly 'MS').
    Ensures complete contiguous time index without missing date gaps.
    """
    filtered = df.copy()

    if filter_category and "Category" in filtered.columns and filter_category != "All Categories":
        filtered = filtered[filtered["Category"] == filter_category]

    if filter_region and "Region" in filtered.columns and filter_region != "All Regions":
        filtered = filtered[filtered["Region"] == filter_region]

    if filtered.empty:
        # Return empty with proper structure
        return pd.DataFrame(columns=["ds", "y", "orders_count"])

    # Map aggregation frequency
    freq_map = {
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "MS"
    }
    pandas_freq = freq_map.get(freq, freq)

    # Aggregation dictionary
    agg_dict = {sales_col: "sum"}
    if quantity_col and quantity_col in filtered.columns:
        agg_dict[quantity_col] = "sum"
    if profit_col and profit_col in filtered.columns:
        agg_dict[profit_col] = "sum"

    # Set index for resample
    ts_df = filtered.set_index(date_col).resample(pandas_freq).agg(agg_dict)
    ts_df["orders_count"] = filtered.set_index(date_col).resample(pandas_freq).size()

    # Fill any gap intervals with 0
    ts_df[sales_col] = ts_df[sales_col].fillna(0.0)
    if quantity_col and quantity_col in ts_df.columns:
        ts_df[quantity_col] = ts_df[quantity_col].fillna(0.0)
    if profit_col and profit_col in ts_df.columns:
        ts_df[profit_col] = ts_df[profit_col].fillna(0.0)
    ts_df["orders_count"] = ts_df["orders_count"].fillna(0)

    # Standardize column naming for modeling: ds (date), y (target sales)
    ts_df = ts_df.reset_index().rename(columns={date_col: "ds", sales_col: "y"})
    ts_df = ts_df.sort_values("ds").reset_index(drop=True)
    return ts_df


def generate_synthetic_sales_data(
    days: int = 1095,
    base_sales: float = 2500.0,
    annual_growth_rate: float = 0.12,
    weekly_seasonality: bool = True,
    annual_seasonality: bool = True,
    noise_level: float = 0.15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic synthetic retail sales data with:
    - Long-term upward business growth trend
    - Strong weekly seasonality (higher weekend sales)
    - Strong annual seasonality (Q4 Black Friday & Christmas peaks, mid-summer bump)
    - Product categories (Technology, Furniture, Office Supplies)
    - Regions (East, West, Central, South)
    """
    np.random.seed(random_seed)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    records = []
    categories = ["Technology", "Furniture", "Office Supplies"]
    category_multipliers = {"Technology": 1.4, "Furniture": 1.1, "Office Supplies": 0.7}
    regions = ["East", "West", "Central", "South"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]

    for i, dt in enumerate(dates):
        # 1. Trend component
        trend_factor = 1.0 + (annual_growth_rate * (i / 365.25))

        # 2. Day of week effect (Friday, Saturday, Sunday higher)
        dow = dt.weekday()
        if weekly_seasonality:
            dow_multipliers = [0.85, 0.90, 0.95, 1.05, 1.25, 1.35, 1.10]
            weekly_factor = dow_multipliers[dow]
        else:
            weekly_factor = 1.0

        # 3. Annual seasonality (Quarter 4 surge, November/December holiday shopping)
        doy = dt.timetuple().tm_yday
        if annual_seasonality:
            # Sine wave for summer bump + Gaussian spike around Nov 25 - Dec 24
            annual_cycle = 1.0 + 0.15 * np.sin(2 * np.pi * doy / 365.25)
            # Holiday shopping bump (days 320 to 358)
            if 315 <= doy <= 358:
                holiday_bump = 1.0 + 0.55 * np.exp(-((doy - 335) ** 2) / (2 * (15 ** 2)))
            else:
                holiday_bump = 1.0
            annual_factor = annual_cycle * holiday_bump
        else:
            annual_factor = 1.0

        # Generate 2 to 6 transactions per day
        n_tx = np.random.randint(2, 7)
        for _ in range(n_tx):
            cat = np.random.choice(categories, p=[0.25, 0.35, 0.40])
            reg = np.random.choice(regions, p=[0.30, 0.30, 0.22, 0.18])
            mode = np.random.choice(ship_modes, p=[0.60, 0.20, 0.15, 0.05])

            noise = np.random.normal(1.0, noise_level)
            noise = max(0.4, noise)

            tx_sales = (base_sales / 4.0) * trend_factor * weekly_factor * annual_factor * category_multipliers[cat] * noise
            tx_sales = max(5.0, round(tx_sales, 2))
            quantity = max(1, int(np.random.poisson(3)))
            discount = round(float(np.random.choice([0.0, 0.1, 0.15, 0.2, 0.3], p=[0.55, 0.20, 0.12, 0.08, 0.05])), 2)
            margin = float(np.random.uniform(0.12, 0.38)) - (discount * 0.7)
            profit = round(tx_sales * margin, 2)

            records.append({
                "Order Date": dt.strftime("%Y-%m-%d"),
                "Ship Date": (dt + timedelta(days=np.random.randint(2, 6))).strftime("%Y-%m-%d"),
                "Ship Mode": mode,
                "Category": cat,
                "Sub-Category": f"{cat} Item",
                "Region": reg,
                "Sales": tx_sales,
                "Quantity": quantity,
                "Discount": discount,
                "Profit": profit
            })

    synth_df = pd.DataFrame(records)
    synth_df["Order Date"] = pd.to_datetime(synth_df["Order Date"])
    return synth_df.sort_values("Order Date").reset_index(drop=True)


def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """
    Validates user-uploaded CSV file.
    Returns: (is_valid, message, detected_date_col, detected_sales_col)
    """
    if df.empty:
        return False, "The uploaded CSV file is empty.", None, None

    date_candidates = [c for c in df.columns if any(w in c.lower() for w in ["date", "time", "day", "timestamp"])]
    sales_candidates = [c for c in df.columns if any(w in c.lower() for w in ["sale", "revenue", "demand", "amount", "total", "quantity", "volume", "y"])]

    if not date_candidates:
        return False, "Could not automatically identify a date column. Please ensure your CSV has a column with 'Date' or 'Time'.", None, None

    if not sales_candidates:
        return False, "Could not automatically identify a sales or demand column. Please ensure your CSV has a column like 'Sales', 'Revenue', or 'Demand'.", None, None

    date_col = date_candidates[0]
    sales_col = sales_candidates[0]

    # Test date conversion
    sample_dates = pd.to_datetime(df[date_col].head(10), errors="coerce")
    if sample_dates.isnull().all():
        return False, f"Column '{date_col}' could not be parsed as valid dates.", None, None

    return True, f"Successfully identified Date column ('{date_col}') and Sales/Demand column ('{sales_col}').", date_col, sales_col
