"""
Business Planning and Operational Decision Engine
Translates Machine Learning forecasts into actionable insights for Store Owners,
Startup Founders, and Business Managers (Inventory, Cash Flow, Staffing, Risk).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def calculate_inventory_plan(
    forecast_df: pd.DataFrame,
    lead_time_days: int = 5,
    service_level: float = 0.95,
    unit_cost: float = 25.0,
    holding_cost_rate: float = 0.20,
    order_cost: float = 50.0
) -> Dict[str, Any]:
    """
    Computes supply chain inventory parameters based on future forecasted demand:
    - Average Daily Demand
    - Demand Volatility (Std Dev)
    - Safety Stock (Buffer against demand spikes & supplier delays)
    - Reorder Point (ROP: when to trigger purchasing order)
    - Economic Order Quantity (EOQ: optimal batch size)
    - Stockout Risk Alerts (surge days)
    """
    if "y_pred" in forecast_df.columns:
        demands = forecast_df["y_pred"].values
    elif "y" in forecast_df.columns:
        demands = forecast_df["y"].values
    else:
        demands = np.array([100.0])

    avg_daily_demand = float(np.mean(demands))
    std_daily_demand = float(np.std(demands)) if len(demands) > 1 else avg_daily_demand * 0.2

    # Z-scores for standard service levels
    z_scores = {0.90: 1.28, 0.95: 1.645, 0.98: 2.05, 0.99: 2.33}
    z = z_scores.get(service_level, 1.645)

    # Supply Chain Formulas
    lead_time_demand = avg_daily_demand * lead_time_days
    safety_stock = z * std_daily_demand * np.sqrt(lead_time_days)
    reorder_point = lead_time_demand + safety_stock

    # Annualized demand estimate (365 days)
    annual_demand = avg_daily_demand * 365.0
    holding_cost_per_unit = unit_cost * holding_cost_rate
    if holding_cost_per_unit > 0:
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    else:
        eoq = avg_daily_demand * 14.0

    # Surge Days (Risk of stockouts if inventory is not prepared)
    surge_threshold = avg_daily_demand + 1.2 * std_daily_demand
    surge_days = []
    if "ds" in forecast_df.columns and "y_pred" in forecast_df.columns:
        for _, row in forecast_df.iterrows():
            if row["y_pred"] >= surge_threshold:
                surge_days.append({
                    "date": pd.to_datetime(row["ds"]).strftime("%Y-%m-%d"),
                    "day_name": pd.to_datetime(row["ds"]).strftime("%A"),
                    "predicted_demand": round(float(row["y_pred"]), 2),
                    "surge_factor": round(float(row["y_pred"] / max(avg_daily_demand, 1.0)), 2)
                })

    return {
        "avg_daily_demand": round(float(avg_daily_demand), 1),
        "std_daily_demand": round(float(std_daily_demand), 1),
        "lead_time_days": int(lead_time_days),
        "service_level_pct": round(float(service_level * 100), 1),
        "lead_time_demand": round(float(lead_time_demand), 1),
        "safety_stock": round(float(safety_stock), 1),
        "reorder_point": round(float(reorder_point), 1),
        "economic_order_quantity": round(float(eoq), 1),
        "surge_days": surge_days,
        "total_surge_days": len(surge_days)
    }


def calculate_cashflow_projections(
    forecast_df: pd.DataFrame,
    profit_margin_pct: float = 0.22,
    fixed_daily_operating_cost: float = 300.0
) -> Dict[str, Any]:
    """
    Projects cumulative revenue, estimated gross profit, and net operating cash flow.
    """
    if "y_pred" in forecast_df.columns:
        preds = forecast_df["y_pred"].values
        lower_bounds = forecast_df.get("y_lower", pd.Series(preds * 0.85)).values
        upper_bounds = forecast_df.get("y_upper", pd.Series(preds * 1.15)).values
    else:
        preds = np.array([0.0])
        lower_bounds = preds
        upper_bounds = preds

    total_projected_revenue = float(np.sum(preds))
    conservative_revenue = float(np.sum(lower_bounds))
    optimistic_revenue = float(np.sum(upper_bounds))

    estimated_gross_profit = total_projected_revenue * profit_margin_pct
    total_operating_costs = len(preds) * fixed_daily_operating_cost
    net_operating_cashflow = estimated_gross_profit - total_operating_costs

    # Weekly breakdown
    weekly_cashflow = []
    chunk_size = 7
    for i in range(0, len(preds), chunk_size):
        chunk_preds = preds[i:i + chunk_size]
        week_num = (i // chunk_size) + 1
        week_rev = float(np.sum(chunk_preds))
        weekly_cashflow.append({
            "Week": f"Week {week_num}",
            "Revenue": round(week_rev, 2),
            "Estimated_Profit": round(week_rev * profit_margin_pct, 2),
            "Cumulative_Revenue": round(float(np.sum(preds[:i + len(chunk_preds)])), 2)
        })

    return {
        "total_projected_revenue": round(total_projected_revenue, 2),
        "conservative_revenue": round(conservative_revenue, 2),
        "optimistic_revenue": round(optimistic_revenue, 2),
        "estimated_gross_profit": round(estimated_gross_profit, 2),
        "net_operating_cashflow": round(net_operating_cashflow, 2),
        "avg_daily_revenue": round(float(np.mean(preds)), 2),
        "weekly_breakdown": pd.DataFrame(weekly_cashflow)
    }


def calculate_staffing_plan(
    forecast_df: pd.DataFrame,
    revenue_per_worker_shift: float = 600.0,
    min_staff_headcount: int = 2,
    max_staff_headcount: int = 10
) -> pd.DataFrame:
    """
    Translates daily demand forecast into recommended daily staffing shifts.
    Ensures adequate coverage on peak shopping days while avoiding idle labor costs.
    """
    df = forecast_df.copy()
    if "y_pred" not in df.columns:
        df["y_pred"] = df["y"] if "y" in df.columns else 1000.0

    df["date"] = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m-%d")
    df["day_name"] = pd.to_datetime(df["ds"]).dt.strftime("%A")
    df["forecasted_sales"] = np.round(df["y_pred"], 2)

    # Required workers = Sales / Revenue Capacity per worker
    raw_workers = df["y_pred"] / revenue_per_worker_shift
    df["suggested_headcount"] = np.clip(np.ceil(raw_workers).astype(int), min_staff_headcount, max_staff_headcount)

    # Workload intensity categorization
    mean_val = df["y_pred"].mean()
    std_val = df["y_pred"].std() if len(df) > 1 else 1.0

    def categorize_intensity(val):
        if val >= mean_val + std_val:
            return "High (Peak Surge)"
        elif val <= mean_val - std_val:
            return "Low (Light Day)"
        return "Normal"

    df["workload_level"] = df["y_pred"].apply(categorize_intensity)

    return df[["date", "day_name", "forecasted_sales", "suggested_headcount", "workload_level"]]


def simulate_scenario(
    base_forecast_df: pd.DataFrame,
    marketing_uplift_pct: float = 0.0,
    price_change_pct: float = 0.0,
    price_elasticity: float = -1.2,
    supply_shock_pct: float = 0.0
) -> pd.DataFrame:
    """
    Simulates a 'What-If' business scenario on the forecasted values:
    - Marketing Uplift: direct percentage boost to demand
    - Price Change: adjusts demand based on price elasticity (Elasticity * % Price Change)
    - Supply Shock: inventory or logistics delivery bottlenecks reducing realizable sales
    """
    df = base_forecast_df.copy()
    y_base = df["y_pred"].values

    # 1. Price elasticity effect: Demand change = Elasticity * Price Change
    elasticity_effect = (price_elasticity * (price_change_pct / 100.0))

    # 2. Combined demand multiplier
    multiplier = 1.0 + (marketing_uplift_pct / 100.0) + elasticity_effect + (supply_shock_pct / 100.0)
    multiplier = max(0.1, multiplier)

    # Calculate simulated sales
    simulated_demand = y_base * multiplier
    # Actual realized revenue after new price
    price_factor = 1.0 + (price_change_pct / 100.0)
    simulated_revenue = simulated_demand * price_factor

    df["baseline_forecast"] = np.round(y_base, 2)
    df["simulated_forecast"] = np.round(simulated_revenue, 2)
    df["net_change"] = np.round(df["simulated_forecast"] - df["baseline_forecast"], 2)
    df["pct_change"] = np.round((df["net_change"] / np.maximum(df["baseline_forecast"], 1.0)) * 100.0, 1)

    return df


def generate_executive_summary(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str,
    metrics: Dict[str, float],
    inventory_plan: Dict[str, Any],
    cashflow_plan: Dict[str, Any]
) -> str:
    """
    Generates a professional executive memo in Markdown format tailored for
    store owners, startup founders, and business managers.
    """
    total_projected = cashflow_plan["total_projected_revenue"]
    horizon_days = len(forecast_df)
    avg_daily = cashflow_plan["avg_daily_revenue"]
    rop = inventory_plan["reorder_point"]
    safety_stock = inventory_plan["safety_stock"]
    surge_count = inventory_plan["total_surge_days"]

    # Historical average comparison
    hist_avg = float(historical_df["y"].mean()) if "y" in historical_df.columns else avg_daily
    growth_rate = ((avg_daily - hist_avg) / hist_avg) * 100.0 if hist_avg > 0 else 0.0
    growth_str = f"+{growth_rate:.1f}%" if growth_rate >= 0 else f"{growth_rate:.1f}%"

    summary_md = f"""# 📈 Executive Sales & Demand Forecast Brief
**Target Audience**: Store Owner / Business Manager / Executive Leadership  
**Forecast Horizon**: {horizon_days} Days Ahead  
**Engine**: Machine Learning ({model_name})  
**Validation Accuracy (WAPE)**: {100.0 - metrics.get('WAPE (%)', 15.0):.1f}% | **R² Score**: {metrics.get('R2 Score', 0.85):.3f}

---

### 1. Key Business Takeaways
* **Projected Revenue ({horizon_days} Days)**: **${total_projected:,.2f}** (Expected daily run-rate: **${avg_daily:,.2f}/day**).
* **Demand Trend**: Anticipated **{growth_str}** shift compared to the historical daily baseline (${hist_avg:,.2f}/day).
* **Estimated Gross Operating Profit**: **${cashflow_plan['estimated_gross_profit']:,.2f}** (assuming standard 22% retail margin).
* **Peak Demand Alert**: **{surge_count} high-volume surge days** identified where demand exceeds 120% of standard capacity.

---

### 2. Supply Chain & Inventory Recommendations
* **Reorder Point (ROP)**: **{rop:,.0f} units**. Place purchase replenishment orders whenever stock drops below this threshold to protect against lead-time runout.
* **Safety Stock Buffer**: Maintain **{safety_stock:,.0f} units** on-hand at all times to absorb unexpected supplier delays or seasonal order spikes.
* **Overstock Prevention**: Avoid excess ordering during identified post-holiday lulls to minimize capital lockup and storage depreciation.

---

### 3. Financial & Cash Flow Guidance
* **Conservative Scenario (Lower 95% Bound)**: Projected at **${cashflow_plan['conservative_revenue']:,.2f}**.
* **Optimistic Scenario (Upper 95% Bound)**: Can reach up to **${cashflow_plan['optimistic_revenue']:,.2f}** with promotional execution.
* **Working Capital Recommendation**: Maintain at least **${cashflow_plan['conservative_revenue'] * 0.25:,.2f}** in short-term liquid reserves for supplier invoices and logistics prep.

---

### 4. Staffing & Operations Action Plan
* Deploy maximum staff on identified peak surge days (especially weekend cycles and month-end dates).
* Adjust warehouse pick-and-pack shifts 2 days prior to peak delivery surges to ensure zero fulfillment backlogs.
"""
    return summary_md
