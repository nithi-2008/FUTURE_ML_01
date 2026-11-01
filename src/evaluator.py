"""
Model Evaluation and Error Diagnostics Module
Provides quantitative metrics and residual diagnostics for business forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression and time-series error metrics:
    - MAE: Mean Absolute Error ($)
    - RMSE: Root Mean Squared Error ($)
    - MAPE: Mean Absolute Percentage Error (%)
    - WAPE: Weighted Absolute Percentage Error (%) - robust to zeros
    - R2: Coefficient of Determination
    - Bias: Mean forecast error (positive means underforecasting, negative means overforecasting)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Clip predictions to non-negative
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: avoid divide by zero
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    # WAPE (Weighted Absolute Percentage Error)
    sum_true = np.sum(y_true)
    if sum_true > 0:
        wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0)
    else:
        wape = 0.0

    # Forecast bias
    bias = float(np.mean(y_true - y_pred))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "R2 Score": round(r2, 4),
        "Bias ($)": round(bias, 2)
    }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Computes detailed residual metrics and returns a dataframe with true, predicted, and errors.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    return pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred,
        "Residual": residuals,
        "Abs_Error": np.abs(residuals),
        "Pct_Error": np.where(y_true > 0, np.abs(residuals) / y_true * 100, 0.0)
    })


def compute_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extracts normalized feature importance rankings from tree-based or linear models.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    elif hasattr(model, "named_steps"):
        # Pipeline handling
        final_estimator = model.steps[-1][1]
        return compute_feature_importance(final_estimator, feature_names, top_n=top_n)

    if importances is not None and len(importances) == len(feature_names):
        # Normalize to sum to 100%
        total = np.sum(importances)
        if total > 0:
            norm_imp = (importances / total) * 100.0
        else:
            norm_imp = importances

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": np.round(norm_imp, 2)
        }).sort_values("Importance (%)", ascending=False).head(top_n).reset_index(drop=True)
        return df_imp

    return pd.DataFrame(columns=["Feature", "Importance (%)"])


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Returns summary statistics of residuals for diagnostic visualization.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "skewness": float(pd.Series(residuals).skew()),
        "kurtosis": float(pd.Series(residuals).kurtosis()),
        "p95": float(np.percentile(np.abs(residuals), 95))
    }
