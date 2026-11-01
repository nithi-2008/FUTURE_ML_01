"""
Sales & Demand Forecasting Dashboard
Future Interns - Machine Learning Task 1 (2026)
Author: Sujan Shetty / Future Interns Showcase
Executive-Grade AI Forecasting & Business Decision System
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import our custom modular forecasting packages
from src.data_loader import (
    load_superstore_data,
    clean_sales_data,
    aggregate_time_series,
    generate_synthetic_sales_data,
    validate_uploaded_csv
)
from src.features import create_features, get_feature_names
from src.models import ModelArena, TimeSeriesForecaster
from src.evaluator import calculate_metrics, compute_residuals, compute_feature_importance
from src.business_planner import (
    calculate_inventory_plan,
    calculate_cashflow_projections,
    calculate_staffing_plan,
    simulate_scenario,
    generate_executive_summary
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales & Demand Forecasting | Future Interns ML Task 1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Enterprise Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Font & App Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #312e81 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #93c5fd, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 8px;
    }
    
    .badge-intern {
        background-color: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    .badge-task {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-status {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fde68a;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    
    .kpi-subtext {
        font-size: 0.8rem;
        font-weight: 500;
        color: #059669;
    }

    /* Custom Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 10px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Info Callouts */
    .biz-callout {
        background: #f8fafc;
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("DemandAI Studio")
    st.caption("Future Interns ML Task 1 • 2026")

    st.markdown("---")
    st.subheader("📁 Data Source Selection")
    
    data_source_mode = st.radio(
        "Choose Data Source:",
        ["Superstore Retail Dataset (Recommended)", "Simulate Live Business Scenario", "Upload Custom CSV File"],
        index=0
    )

    uploaded_file = None
    if data_source_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        st.info("Ensure your file has a date column and sales/revenue column.")
    elif data_source_mode == "Simulate Live Business Scenario":
        st.markdown("**Simulation Parameters**")
        sim_growth = st.slider("Annual Business Growth Rate (%)", 0, 40, 15) / 100.0
        sim_volatility = st.slider("Demand Volatility (Noise)", 0.05, 0.40, 0.15, step=0.05)
        sim_days = st.select_slider("Historical History (Days)", options=[365, 730, 1095], value=730)

    st.markdown("---")
    st.subheader("⚙️ Forecasting Configuration")

    horizon_days = st.slider("Forecast Horizon (Days Ahead)", min_value=7, max_value=90, value=30, step=7)
    freq_choice = st.selectbox("Aggregation Granularity", ["Daily", "Weekly", "Monthly"], index=0)

    st.markdown("---")
    st.subheader("📦 Supply Chain Settings")
    lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1, max_value=30, value=5)
    service_level = st.selectbox("Desired Service Level", [0.90, 0.95, 0.98, 0.99], index=1, format_func=lambda x: f"{int(x*100)}% Fill Rate")

    st.markdown("---")
    st.markdown("""
    **Created for**:  
    **Future Interns ML Program**  
    [Task 1: Sales & Demand Forecasting](https://futureinterns.com/)  
    **GitHub**: [nithi-2008/FUTURE_ML_01](https://github.com/nithi-2008/FUTURE_ML_01.git)
    """)


# ---------------------------------------------------------
# Data Ingestion & Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing business dataset...")
def get_dataset(source_mode, sim_params=None):
    if source_mode == "Superstore Retail Dataset (Recommended)":
        return load_superstore_data("data/superstore_sales.csv")
    elif source_mode == "Simulate Live Business Scenario":
        growth = sim_params.get("growth", 0.15) if sim_params else 0.15
        vol = sim_params.get("vol", 0.15) if sim_params else 0.15
        days = sim_params.get("days", 730) if sim_params else 730
        return generate_synthetic_sales_data(days=days, annual_growth_rate=growth, noise_level=vol)
    return None

# Load Raw Data
raw_df = None
if data_source_mode == "Upload Custom CSV File":
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            custom_df = pd.read_csv(uploaded_file, encoding="latin1")
        is_valid, msg, date_col, sales_col = validate_uploaded_csv(custom_df)
        if not is_valid:
            st.error(msg)
            st.stop()
        else:
            raw_df = clean_sales_data(custom_df, date_col=date_col, sales_col=sales_col)
            st.sidebar.success(msg)
    else:
        st.info("Please upload a CSV file in the sidebar to begin analysis.")
        st.stop()
elif data_source_mode == "Simulate Live Business Scenario":
    sim_params = {"growth": sim_growth, "vol": sim_volatility, "days": sim_days}
    raw_df = get_dataset(data_source_mode, sim_params)
else:
    raw_df = get_dataset(data_source_mode)

if raw_df is None or raw_df.empty:
    st.error("Dataset could not be loaded. Please check data files.")
    st.stop()

# Filter options based on available columns
category_filter = "All Categories"
region_filter = "All Regions"

if "Category" in raw_df.columns:
    categories = ["All Categories"] + sorted(raw_df["Category"].dropna().unique().tolist())
    category_filter = st.sidebar.selectbox("Filter Category", categories, index=0)

if "Region" in raw_df.columns:
    regions = ["All Regions"] + sorted(raw_df["Region"].dropna().unique().tolist())
    region_filter = st.sidebar.selectbox("Filter Region", regions, index=0)

# Aggregation
ts_daily = aggregate_time_series(
    raw_df,
    freq=freq_choice,
    filter_category=category_filter,
    filter_region=region_filter
)

if len(ts_daily) < 40:
    st.warning("Selected filter resulted in insufficient time points for reliable forecasting. Please broaden your selection.")
    st.stop()


# ---------------------------------------------------------
# Cached Model Arena Training
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training and benchmarking ML forecasting models...")
def train_forecasting_models(ts_data, test_horizon=45):
    arena = ModelArena(test_days=test_horizon)
    leaderboard = arena.train_and_benchmark(ts_data)
    best_forecaster = arena.get_best_model()
    return arena, leaderboard, best_forecaster

arena, leaderboard, best_model = train_forecasting_models(ts_daily, test_horizon=min(45, int(len(ts_daily) * 0.2)))

# Generate Future Forecast
@st.cache_data(show_spinner="Generating future recursive forecast and prediction intervals...")
def generate_forecast(_model, historical_series, horizon):
    return _model.forecast_future(historical_series, horizon_days=horizon)

forecast_df = generate_forecast(best_model, ts_daily, horizon_days)

# Calculate Business Plans
inventory_plan = calculate_inventory_plan(forecast_df, lead_time_days=lead_time, service_level=service_level)
cashflow_plan = calculate_cashflow_projections(forecast_df)
staffing_plan = calculate_staffing_plan(forecast_df)
best_metrics = arena.results[arena.best_model_name]["metrics"]
exec_summary_md = generate_executive_summary(ts_daily, forecast_df, arena.best_model_name, best_metrics, inventory_plan, cashflow_plan)


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div>
        <span class="badge-pill badge-intern">Future Interns ML Task 1</span>
        <span class="badge-pill badge-task">Sales & Demand Forecasting</span>
        <span class="badge-pill badge-status">Active Model: {arena.best_model_name}</span>
    </div>
    <h1 class="hero-title">Enterprise Sales & Demand Forecasting System</h1>
    <p class="hero-subtitle">
        Empowering Store Owners, Startup Founders, and Operations Managers with Machine Learning, 
        Inventory Optimization, and Cash Flow Projections.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# High-Level KPI Ribbon
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_hist_rev = float(ts_daily["y"].sum())
avg_daily_sales = float(ts_daily["y"].mean())
proj_total_rev = cashflow_plan["total_projected_revenue"]
hist_vs_proj_growth = ((cashflow_plan["avg_daily_revenue"] - avg_daily_sales) / avg_daily_sales) * 100.0 if avg_daily_sales > 0 else 0.0

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Historical Revenue</div>
        <div class="kpi-value">${total_hist_rev:,.0f}</div>
        <div class="kpi-subtext" style="color:#64748b;">{len(ts_daily)} periods analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Daily Run Rate</div>
        <div class="kpi-value">${avg_daily_sales:,.0f}</div>
        <div class="kpi-subtext" style="color:#64748b;">Historical baseline</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    growth_color = "#059669" if hist_vs_proj_growth >= 0 else "#dc2626"
    arrow = "▲" if hist_vs_proj_growth >= 0 else "▼"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{horizon_days}-Day Forecast</div>
        <div class="kpi-value">${proj_total_rev:,.0f}</div>
        <div class="kpi-subtext" style="color:{growth_color};">{arrow} {abs(hist_vs_proj_growth):.1f}% vs past avg</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Reorder Point (ROP)</div>
        <div class="kpi-value">{inventory_plan['reorder_point']:,.0f} <span style="font-size:0.9rem;font-weight:500;">units</span></div>
        <div class="kpi-subtext" style="color:#2563eb;">Buffer: {inventory_plan['safety_stock']:,.0f} units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    model_acc = max(0.0, 100.0 - best_metrics.get("WAPE (%)", 15.0))
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{model_acc:.1f}%</div>
        <div class="kpi-subtext" style="color:#7c3aed;">WAPE: {best_metrics.get('WAPE (%)', 15.0):.1f}% | R²: {best_metrics.get('R2 Score', 0.85):.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ---------------------------------------------------------
# Tabbed Workspace
# ---------------------------------------------------------
tab_overview, tab_eda, tab_features, tab_arena, tab_forecast, tab_scenario, tab_planning, tab_report = st.tabs([
    "📊 Executive Overview",
    "🔍 Exploratory Data (EDA)",
    "⚙️ Feature Engineering",
    "🤖 Model Arena & Benchmark",
    "🔮 Future Projections",
    "🎛️ What-If Simulator",
    "📦 Inventory & Staffing Hub",
    "📑 Executive Memo & Export"
])


# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tab_overview:
    st.markdown('<div class="section-title">📈 Historical Sales Timeline & Forecasting Trajectory</div>', unsafe_allow_html=True)

    # Plotly interactive historical + future line
    fig_overview = go.Figure()

    # Historical Sales
    fig_overview.add_trace(go.Scatter(
        x=ts_daily["ds"],
        y=ts_daily["y"],
        mode="lines",
        name="Historical Sales ($)",
        line=dict(color="#2563eb", width=2),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Sales</b>: $%{y:,.2f}<extra></extra>"
    ))

    # Rolling 14-day average for trend clarity
    rolling_14 = ts_daily["y"].rolling(window=14, min_periods=1).mean()
    fig_overview.add_trace(go.Scatter(
        x=ts_daily["ds"],
        y=rolling_14,
        mode="lines",
        name="14-Period Moving Average",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>14D MA</b>: $%{y:,.2f}<extra></extra>"
    ))

    # Future Forecast Line
    fig_overview.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["y_pred"],
        mode="lines+markers",
        name="Future Prediction ($)",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=4),
        hovertemplate="<b>Forecast Date</b>: %{x|%Y-%m-%d}<br><b>Expected</b>: $%{y:,.2f}<extra></extra>"
    ))

    # 95% Confidence Band
    fig_overview.add_trace(go.Scatter(
        x=list(forecast_df["ds"]) + list(forecast_df["ds"])[::-1],
        y=list(forecast_df["y_upper"]) + list(forecast_df["y_lower"])[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))

    fig_overview.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Sales Revenue ($)", showgrid=True, gridcolor="#f1f5f9"),
        hovermode="x unified"
    )

    st.plotly_chart(fig_overview, use_container_width=True)

    # 2 Sub-charts: Category & Regional Distribution (if available in raw_df)
    col_sub1, col_sub2 = st.columns(2)
    
    with col_sub1:
        if "Category" in raw_df.columns:
            st.markdown('<div class="section-title">📦 Revenue by Product Category</div>', unsafe_allow_html=True)
            cat_summary = raw_df.groupby("Category")["Sales"].sum().reset_index()
            fig_cat = px.pie(
                cat_summary, 
                names="Category", 
                values="Sales", 
                hole=0.45,
                color_discrete_sequence=["#3b82f6", "#10b981", "#8b5cf6"]
            )
            fig_cat.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.markdown('<div class="section-title">📅 Sales by Day of Week</div>', unsafe_allow_html=True)
            dow_df = ts_daily.copy()
            dow_df["Day"] = dow_df["ds"].dt.day_name()
            dow_summary = dow_df.groupby("Day")["y"].mean().reindex([
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ]).reset_index()
            fig_dow = px.bar(dow_summary, x="Day", y="y", color="y", color_continuous_scale="Viridis")
            fig_dow.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
            st.plotly_chart(fig_dow, use_container_width=True)

    with col_sub2:
        if "Region" in raw_df.columns:
            st.markdown('<div class="section-title">🗺️ Revenue by Sales Territory</div>', unsafe_allow_html=True)
            reg_summary = raw_df.groupby("Region")["Sales"].sum().reset_index()
            fig_reg = px.bar(
                reg_summary, 
                x="Region", 
                y="Sales", 
                color="Region",
                color_discrete_sequence=["#6366f1", "#ec4899", "#14b8a6", "#f59e0b"]
            )
            fig_reg.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300, showlegend=False)
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.markdown('<div class="section-title">📊 Monthly Sales Seasonality</div>', unsafe_allow_html=True)
            monthly_df = ts_daily.copy()
            monthly_df["Month"] = monthly_df["ds"].dt.strftime("%B")
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            monthly_summary = monthly_df.groupby("Month")["y"].mean().reindex(month_order).dropna().reset_index()
            fig_mo = px.line(monthly_summary, x="Month", y="y", markers=True)
            fig_mo.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=300)
            st.plotly_chart(fig_mo, use_container_width=True)


# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_eda:
    st.markdown('<div class="section-title">🔍 Historical Sales Behavior, Trends & Seasonality</div>', unsafe_allow_html=True)
    st.write("Understand consumer purchasing rhythms, weekday vs weekend patterns, and annual seasonal peaks.")

    eda_c1, eda_c2 = st.columns(2)

    with eda_c1:
        st.markdown("**Sales Distribution & Density (Histogram + Box)**")
        fig_dist = px.histogram(
            ts_daily, 
            x="y", 
            marginal="box", 
            nbins=40,
            color_discrete_sequence=["#3b82f6"],
            labels={"y": "Daily Sales Revenue ($)"}
        )
        fig_dist.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    with eda_c2:
        st.markdown("**Weekday vs Weekend Revenue Variance**")
        dow_analysis = ts_daily.copy()
        dow_analysis["Day"] = dow_analysis["ds"].dt.day_name()
        dow_analysis["Is Weekend"] = dow_analysis["ds"].dt.dayofweek.isin([5, 6]).map({True: "Weekend", False: "Weekday"})
        fig_box = px.box(
            dow_analysis, 
            x="Day", 
            y="y", 
            color="Is Weekend",
            category_orders={"Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
            color_discrete_map={"Weekday": "#3b82f6", "Weekend": "#10b981"},
            labels={"y": "Sales ($)", "Day": "Day of Week"}
        )
        fig_box.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    # Seasonality Heatmap
    st.markdown("**Seasonal Month vs Day-of-Week Revenue Heatmap**")
    heat_df = ts_daily.copy()
    heat_df["Month"] = heat_df["ds"].dt.strftime("%b")
    heat_df["Month_Num"] = heat_df["ds"].dt.month
    heat_df["Day"] = heat_df["ds"].dt.strftime("%a")
    heat_df["Day_Num"] = heat_df["ds"].dt.dayofweek

    pivot_table = heat_df.pivot_table(
        index="Day", 
        columns="Month", 
        values="y", 
        aggfunc="mean"
    )
    # Order axes
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_table = pivot_table.reindex(index=[d for d in day_order if d in pivot_table.index], 
                                     columns=[m for m in month_order if m in pivot_table.columns])

    fig_heat = px.imshow(
        pivot_table,
        labels=dict(x="Month", y="Day of Week", color="Avg Sales ($)"),
        color_continuous_scale="Tealgrn",
        aspect="auto"
    )
    fig_heat.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📄 View Historical Transaction Data Sample (First 100 Rows)"):
        st.dataframe(raw_df.head(100), use_container_width=True)


# =========================================================
# TAB 3: FEATURE ENGINEERING
# =========================================================
with tab_features:
    st.markdown('<div class="section-title">⚙️ Machine Learning Feature Engineering Laboratory</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="biz-callout">
        <b>Why Feature Engineering is Critical for Time-Series:</b><br>
        Standard regression algorithms do not inherently understand the flow of time. By constructing 
        <b>calendar indicators</b> (weekends, month ends), <b>cyclical trigonometric encodings</b> (sine/cosine annual waves),
        <b>lag features</b> (what did we sell 7 days ago?), and <b>rolling moving averages</b> (past 7-day momentum), 
        we transform raw timestamp data into high-signal predictive features.
    </div>
    """, unsafe_allow_html=True)

    sample_featured_df = create_features(ts_daily, date_col="ds", target_col="y", drop_na=True)
    feat_names = get_feature_names(sample_featured_df)

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    f_col1.metric("Total Engineered Features", len(feat_names))
    f_col2.metric("Lag Offsets", "1, 2, 7, 14, 21, 30 days")
    f_col3.metric("Rolling Windows", "7, 14, 30 days (Mean/Std/Max/Min)")
    f_col4.metric("Cyclical Transforms", "Sin & Cos (DOW, Month, DOY)")

    st.markdown("**Engineered Feature Matrix Preview**")
    st.dataframe(sample_featured_df.tail(15), use_container_width=True)

    # Correlation Matrix of top features with target
    st.markdown("**Correlation with Target Sales (y)**")
    numeric_df = sample_featured_df[feat_names + ["y"]]
    corr_series = numeric_df.corr()["y"].drop("y").sort_values(ascending=False)
    
    fig_corr = px.bar(
        x=corr_series.values,
        y=corr_series.index,
        orientation="h",
        labels={"x": "Pearson Correlation with Sales", "y": "Feature"},
        color=corr_series.values,
        color_continuous_scale="RdBu_r"
    )
    fig_corr.update_layout(template="plotly_white", height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)


# =========================================================
# TAB 4: MODEL ARENA & BENCHMARK
# =========================================================
with tab_arena:
    st.markdown('<div class="section-title">🤖 Model Arena: Algorithm Shootout & Validation Leaderboard</div>', unsafe_allow_html=True)
    st.write("We evaluate multiple machine learning paradigms chronologically on an out-of-time holdout set to prevent data leakage.")

    # Leaderboard Table
    st.dataframe(
        leaderboard.style.highlight_min(subset=["MAE ($)", "RMSE ($)", "MAPE (%)", "WAPE (%)"], color="#d1fae5")
                         .highlight_max(subset=["R2 Score"], color="#d1fae5")
                         .format({
                             "MAE ($)": "${:,.2f}",
                             "RMSE ($)": "${:,.2f}",
                             "MAPE (%)": "{:.2f}%",
                             "WAPE (%)": "{:.2f}%",
                             "R2 Score": "{:.4f}",
                             "Forecast Bias ($)": "${:,.2f}"
                         }),
        use_container_width=True
    )

    st.success(f"🏆 Top-Performing Architecture: **{arena.best_model_name}** (Lowest validation error).")

    # Selected Model Inspector
    active_test_res = arena.results[arena.best_model_name]
    y_test_actual = active_test_res["test_df"]["y"].values
    y_test_pred = active_test_res["y_pred"]
    test_dates = active_test_res["test_df"]["ds"].values

    col_ar1, col_ar2 = st.columns(2)

    with col_ar1:
        st.markdown(f"**Actual vs Predicted on Holdout Set ({arena.best_model_name})**")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=test_dates, y=y_test_actual, mode="lines", name="Actual Sales", line=dict(color="#0f172a", width=2)))
        fig_eval.add_trace(go.Scatter(x=test_dates, y=y_test_pred, mode="lines+markers", name="Model Predicted", line=dict(color="#10b981", width=2, dash="dot")))
        fig_eval.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig_eval, use_container_width=True)

    with col_ar2:
        st.markdown("**Residual Error Diagnostics (Prediction Variance)**")
        residuals = y_test_actual - y_test_pred
        fig_resid = px.histogram(
            x=residuals, 
            nbins=25, 
            color_discrete_sequence=["#6366f1"],
            labels={"x": "Residual Error ($) [Actual - Predicted]"}
        )
        fig_resid.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_resid, use_container_width=True)

    # Feature Importance (if applicable)
    if hasattr(best_model.model, "feature_importances_") or hasattr(best_model.model, "coef_"):
        st.markdown(f"**Top Predictive Drivers & Feature Importance ({arena.best_model_name})**")
        imp_df = compute_feature_importance(best_model.model, best_model.feature_names, top_n=12)
        if not imp_df.empty:
            fig_imp = px.bar(
                imp_df, 
                x="Importance (%)", 
                y="Feature", 
                orientation="h",
                color="Importance (%)",
                color_continuous_scale="Purp"
            )
            fig_imp.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# TAB 5: FUTURE FORECAST
# =========================================================
with tab_forecast:
    st.markdown(f'<div class="section-title">🔮 Future Demand Projections ({horizon_days} Days Ahead)</div>', unsafe_allow_html=True)
    st.write(f"Generating recursive multi-step forecasting using **{arena.best_model_name}** with 80% and 95% confidence intervals.")

    fig_f = go.Figure()

    # Historical context (last 90 days for zoom clarity)
    hist_subset = ts_daily.tail(90)
    fig_f.add_trace(go.Scatter(
        x=hist_subset["ds"], 
        y=hist_subset["y"], 
        mode="lines", 
        name="Recent History", 
        line=dict(color="#64748b", width=1.5)
    ))

    # Forecast line
    fig_f.add_trace(go.Scatter(
        x=forecast_df["ds"], 
        y=forecast_df["y_pred"], 
        mode="lines+markers", 
        name="Projected Forecast", 
        line=dict(color="#10b981", width=3)
    ))

    # 80% confidence interval
    fig_f.add_trace(go.Scatter(
        x=list(forecast_df["ds"]) + list(forecast_df["ds"])[::-1],
        y=list(forecast_df["y_upper_80"]) + list(forecast_df["y_lower_80"])[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="80% Confidence Interval"
    ))

    # 95% confidence interval
    fig_f.add_trace(go.Scatter(
        x=list(forecast_df["ds"]) + list(forecast_df["ds"])[::-1],
        y=list(forecast_df["y_upper"]) + list(forecast_df["y_lower"])[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Confidence Interval"
    ))

    fig_f.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Daily Demand / Sales ($)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Forecast Data Table
    st.markdown("**Detailed Daily Forecast Table**")
    display_forecast = forecast_df.copy()
    display_forecast["Date"] = pd.to_datetime(display_forecast["ds"]).dt.strftime("%Y-%m-%d (%a)")
    display_forecast["Expected Sales"] = display_forecast["y_pred"].map("${:,.2f}".format)
    display_forecast["Conservative (95% Low)"] = display_forecast["y_lower"].map("${:,.2f}".format)
    display_forecast["Optimistic (95% High)"] = display_forecast["y_upper"].map("${:,.2f}".format)

    st.dataframe(
        display_forecast[["Date", "Expected Sales", "Conservative (95% Low)", "Optimistic (95% High)"]],
        use_container_width=True,
        height=280
    )


# =========================================================
# TAB 6: WHAT-IF SCENARIO SIMULATOR
# =========================================================
with tab_scenario:
    st.markdown('<div class="section-title">🎛️ Interactive "What-If" Business Scenario Simulator</div>', unsafe_allow_html=True)
    st.write("Test strategic decisions before committing capital. Adjust marketing campaigns, prices, or supply chain disruptions.")

    sc_c1, sc_c2, sc_c3 = st.columns(3)
    with sc_c1:
        sim_marketing = st.slider("Marketing Campaign Uplift (%)", 0, 50, 15, help="Projected boost from paid ad spend, social campaigns, or influencer partnerships.")
    with sc_c2:
        sim_price = st.slider("Price Adjustment (%)", -25, 25, 0, help="Simulate a price hike or promotional markdown.")
        sim_elasticity = st.number_input("Price Elasticity of Demand", value=-1.2, step=0.1, help="Typical retail elasticity is -1.0 to -1.5 (negative number).")
    with sc_c3:
        sim_shock = st.slider("Supply Chain Bottleneck (%)", -50, 0, 0, help="Simulate delivery delays or inventory shortage constraints.")

    # Run Simulation
    sim_df = simulate_scenario(
        forecast_df,
        marketing_uplift_pct=sim_marketing,
        price_change_pct=sim_price,
        price_elasticity=sim_elasticity,
        supply_shock_pct=sim_shock
    )

    base_rev = float(sim_df["baseline_forecast"].sum())
    scen_rev = float(sim_df["simulated_forecast"].sum())
    delta_rev = scen_rev - base_rev
    pct_delta = (delta_rev / base_rev) * 100.0 if base_rev > 0 else 0.0

    # Impact Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Total Forecast", f"${base_rev:,.2f}")
    m2.metric("Simulated Scenario Revenue", f"${scen_rev:,.2f}")
    m3.metric("Net Financial Impact", f"${delta_rev:+,.2f}", f"{pct_delta:+.1f}%")

    # Comparison Plot
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=sim_df["ds"], y=sim_df["baseline_forecast"], mode="lines", name="Baseline ML Forecast", line=dict(color="#64748b", dash="dash")))
    fig_sim.add_trace(go.Scatter(x=sim_df["ds"], y=sim_df["simulated_forecast"], mode="lines+markers", name="Simulated Scenario", line=dict(color="#6366f1", width=2.5)))
    fig_sim.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="Forecasted Revenue ($)"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# =========================================================
# TAB 7: INVENTORY & STAFFING HUB
# =========================================================
with tab_planning:
    st.markdown('<div class="section-title">📦 Operational Decision Hub (Inventory, Cash Flow & Staffing)</div>', unsafe_allow_html=True)
    st.write("Convert predictions directly into purchase orders, financial liquidity reserves, and labor shifts.")

    inv_col1, inv_col2 = st.columns(2)

    with inv_col1:
        st.markdown("### 🏷️ Inventory & Replenishment Rules")
        st.markdown(f"""
        - **Average Daily Demand ($\\\\mu_d$)**: **{inventory_plan['avg_daily_demand']:,.1f} units/day**
        - **Demand Volatility ($\\\\sigma_d$)**: **{inventory_plan['std_daily_demand']:,.1f} units/day**
        - **Supplier Lead Time ($L$)**: **{inventory_plan['lead_time_days']} days**
        - **Target Service Level ($Z$)**: **{inventory_plan['service_level_pct']}% fill rate**
        - **Required Safety Stock (Buffer)**: **{inventory_plan['safety_stock']:,.0f} units**
        - **Recommended Reorder Point (ROP)**: **{inventory_plan['reorder_point']:,.0f} units**
        - **Economic Order Quantity (EOQ)**: **{inventory_plan['economic_order_quantity']:,.0f} units per purchase batch**
        """)
        st.info("💡 **Operational Directive**: Place a restocking PO immediately whenever inventory drops below the Reorder Point.")

    with inv_col2:
        st.markdown("### ⚠️ Critical Surge Days (Stockout Risk Alert)")
        if inventory_plan["total_surge_days"] > 0:
            st.warning(f"Detected **{inventory_plan['total_surge_days']} Peak Surge Days** exceeding 120% of standard capacity. Prepare warehouse buffers!")
            surge_table = pd.DataFrame(inventory_plan["surge_days"])
            st.dataframe(
                surge_table.rename(columns={
                    "date": "Date",
                    "day_name": "Day",
                    "predicted_demand": "Predicted Demand ($)",
                    "surge_factor": "Surge Index (x Avg)"
                }),
                use_container_width=True,
                height=220
            )
        else:
            st.success("No extreme demand surges detected during this forecast horizon. Inventory demand remains steady.")

    st.markdown("---")
    st.markdown("### 👥 Daily Staffing & Shift Workload Scheduler")
    st.write("Ensures optimal customer service coverage during high-volume periods without paying idle labor during quiet days.")
    
    st.dataframe(
        staffing_plan.rename(columns={
            "date": "Date",
            "day_name": "Day of Week",
            "forecasted_sales": "Forecasted Sales ($)",
            "suggested_headcount": "Recommended Shift Staff",
            "workload_level": "Workload Intensity"
        }),
        use_container_width=True,
        height=260
    )


# =========================================================
# TAB 8: EXECUTIVE MEMO & EXPORT STUDIO
# =========================================================
with tab_report:
    st.markdown('<div class="section-title">📑 Executive Decision Memo & Data Export</div>', unsafe_allow_html=True)
    st.write("Submission-ready brief tailored for store owners, investors, and leadership.")

    # Render Executive Memo in UI
    st.markdown(exec_summary_md)

    st.markdown("---")
    st.markdown("### 📥 Export Predictions & Decision Data")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # 1. Download Forecast CSV
    with exp_col1:
        csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv_forecast,
            file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. Download Inventory & Staffing Plan CSV
    with exp_col2:
        merged_plan = forecast_df.copy()
        merged_plan["suggested_headcount"] = staffing_plan["suggested_headcount"].values
        merged_plan["workload_level"] = staffing_plan["workload_level"].values
        csv_plan = merged_plan.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Staffing & Inventory (CSV)",
            data=csv_plan,
            file_name=f"operational_plan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 3. Download Executive Memo Markdown
    with exp_col3:
        st.download_button(
            label="⬇️ Download Executive Memo (.md)",
            data=exec_summary_md.encode("utf-8"),
            file_name=f"executive_forecast_memo_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
