"""
Sales & Demand Forecasting - Future Interns ML Task 1
Package Initialization
"""

from .data_loader import (
    load_superstore_data,
    clean_sales_data,
    aggregate_time_series,
    generate_synthetic_sales_data,
    validate_uploaded_csv
)
from .features import create_features, get_feature_names
from .models import ModelArena, TimeSeriesForecaster
from .evaluator import calculate_metrics, plot_residuals, compute_feature_importance
from .business_planner import (
    calculate_inventory_plan,
    calculate_cashflow_projections,
    calculate_staffing_plan,
    simulate_scenario,
    generate_executive_summary
)
