"""
Forecasting Models and Model Arena
Includes Ridge, Random Forest, Gradient Boosting, Extra Trees, and Holt-Winters Exponential Smoothing,
along with temporal train/test splitting, multi-model benchmarking, and recursive multi-step future forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import timedelta

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import create_features, get_feature_names
from .evaluator import calculate_metrics


class TimeSeriesForecaster:
    """
    Wrapper for individual forecasting models with unified train, evaluate, and recursive future forecast methods.
    """
    def __init__(self, model_name: str = "HistGradientBoosting", params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.params = params or {}
        self.model = self._init_model()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.residual_std: float = 0.0
        self.is_statistical: bool = (model_name == "Holt-Winters Exponential Smoothing")
        self.stats_model = None

    def _init_model(self):
        if self.model_name == "Ridge Regression":
            alpha = self.params.get("alpha", 1.0)
            return Ridge(alpha=alpha)
        elif self.model_name == "Random Forest":
            n_estimators = self.params.get("n_estimators", 150)
            max_depth = self.params.get("max_depth", 12)
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif self.model_name == "HistGradientBoosting":
            max_iter = self.params.get("max_iter", 150)
            learning_rate = self.params.get("learning_rate", 0.08)
            return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, random_state=42)
        elif self.model_name == "Extra Trees":
            n_estimators = self.params.get("n_estimators", 150)
            return ExtraTreesRegressor(n_estimators=n_estimators, max_depth=12, random_state=42, n_jobs=-1)
        elif self.model_name == "Holt-Winters Exponential Smoothing":
            return None
        else:
            return HistGradientBoostingRegressor(random_state=42)

    def fit(self, train_df: pd.DataFrame, feature_cols: Optional[List[str]] = None, target_col: str = "y"):
        """
        Fits the model on training data.
        """
        if self.is_statistical:
            # Holt-Winters on raw target series
            y_train = train_df[target_col].values
            # Handle zeros or negatives safely
            y_train_safe = np.maximum(y_train, 1e-3)
            seasonal_periods = self.params.get("seasonal_periods", 7)
            try:
                self.stats_model = ExponentialSmoothing(
                    y_train_safe,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated"
                ).fit()
            except Exception:
                # Fallback to simple additive trend
                self.stats_model = ExponentialSmoothing(
                    y_train_safe,
                    trend="add",
                    initialization_method="estimated"
                ).fit()
            self.is_fitted = True
            residuals = y_train - self.stats_model.fittedvalues
            self.residual_std = float(np.std(residuals))
            return self

        if feature_cols is None:
            feature_cols = get_feature_names(train_df)
        self.feature_names = feature_cols

        X_train = train_df[self.feature_names].values
        y_train = train_df[target_col].values

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate in-sample residual standard deviation for prediction intervals
        y_train_pred = np.clip(self.model.predict(X_train), 0, None)
        residuals = y_train - y_train_pred
        self.residual_std = float(np.std(residuals)) if len(residuals) > 0 else 1.0

        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for a dataframe with features.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if self.is_statistical:
            n_steps = len(test_df)
            preds = self.stats_model.forecast(steps=n_steps)
            return np.clip(preds, 0, None)

        X_test = test_df[self.feature_names].values
        preds = self.model.predict(X_test)
        return np.clip(preds, 0, None)

    def forecast_future(
        self,
        historical_df: pd.DataFrame,
        horizon_days: int = 30,
        freq: str = "D",
        target_col: str = "y"
    ) -> pd.DataFrame:
        """
        Generates recursive multi-step future forecasts with 80% and 95% confidence intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating future forecasts.")

        last_date = pd.to_datetime(historical_df["ds"].iloc[-1])
        future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]

        if self.is_statistical:
            raw_forecast = self.stats_model.forecast(steps=horizon_days)
            preds = np.clip(raw_forecast, 0, None)

            # Growth in uncertainty with horizon
            uncertainty_growth = np.sqrt(1.0 + np.arange(horizon_days) * 0.05)
            z_80 = 1.28
            z_95 = 1.96

            lower_80 = np.clip(preds - z_80 * self.residual_std * uncertainty_growth, 0, None)
            upper_80 = preds + z_80 * self.residual_std * uncertainty_growth
            lower_95 = np.clip(preds - z_95 * self.residual_std * uncertainty_growth, 0, None)
            upper_95 = preds + z_95 * self.residual_std * uncertainty_growth

            return pd.DataFrame({
                "ds": future_dates,
                "y_pred": np.round(preds, 2),
                "y_lower_80": np.round(lower_80, 2),
                "y_upper_80": np.round(upper_80, 2),
                "y_lower": np.round(lower_95, 2),
                "y_upper": np.round(upper_95, 2)
            })

        # Recursive Forecaster for ML regression models
        # Maintain a dynamic rolling history
        sim_history = historical_df[["ds", target_col]].copy()

        future_predictions = []
        for dt in future_dates:
            # 1. Add future date row with dummy y
            next_row = pd.DataFrame([{"ds": dt, target_col: np.nan}])
            augmented = pd.concat([sim_history, next_row], ignore_index=True)

            # 2. Compute features on augmented data (lags will pull from recent history)
            feat_df = create_features(augmented, date_col="ds", target_col=target_col, drop_na=False)

            # 3. Extract the feature row for the current target date
            current_x = feat_df.iloc[[-1]][self.feature_names].values

            # 4. Predict
            pred_val = float(np.clip(self.model.predict(current_x)[0], 0, None))
            future_predictions.append(pred_val)

            # 5. Fill prediction back into sim_history so subsequent lags can use it
            sim_history = pd.concat([sim_history, pd.DataFrame([{"ds": dt, target_col: pred_val}])], ignore_index=True)

        preds = np.array(future_predictions)

        # Confidence intervals with expanding uncertainty horizon
        uncertainty_growth = np.sqrt(1.0 + np.arange(horizon_days) * 0.04)
        z_80 = 1.28
        z_95 = 1.96

        lower_80 = np.clip(preds - z_80 * self.residual_std * uncertainty_growth, 0, None)
        upper_80 = preds + z_80 * self.residual_std * uncertainty_growth
        lower_95 = np.clip(preds - z_95 * self.residual_std * uncertainty_growth, 0, None)
        upper_95 = preds + z_95 * self.residual_std * uncertainty_growth

        return pd.DataFrame({
            "ds": future_dates,
            "y_pred": np.round(preds, 2),
            "y_lower_80": np.round(lower_80, 2),
            "y_upper_80": np.round(upper_80, 2),
            "y_lower": np.round(lower_95, 2),
            "y_upper": np.round(upper_95, 2)
        })


class ModelArena:
    """
    Automated Model Arena: Trains multiple forecasting models, validates on holdout test set,
    computes evaluation metrics, and ranks models for business stakeholders.
    """
    def __init__(self, test_days: int = 60):
        self.test_days = test_days
        self.models: Dict[str, TimeSeriesForecaster] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically (train on past, test on recent future).
        """
        if len(df) <= self.test_days + 30:
            test_size = max(14, int(len(df) * 0.2))
        else:
            test_size = self.test_days

        train_df = df.iloc[:-test_size].copy().reset_index(drop=True)
        test_df = df.iloc[-test_size:].copy().reset_index(drop=True)
        return train_df, test_df

    def train_and_benchmark(
        self,
        daily_ts: pd.DataFrame,
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Executes feature engineering, temporal train/test split, trains models,
        and generates benchmark leaderboard.
        """
        if model_names is None:
            model_names = [
                "HistGradientBoosting",
                "Random Forest",
                "Ridge Regression",
                "Extra Trees",
                "Holt-Winters Exponential Smoothing"
            ]

        # 1. Feature Engineering on the contiguous series
        df_featured = create_features(daily_ts, date_col="ds", target_col="y", drop_na=True)

        # 2. Chronological Split
        train_df, test_df = self.temporal_split(df_featured)
        feature_cols = get_feature_names(train_df)

        y_test = test_df["y"].values

        leaderboard = []

        for name in model_names:
            try:
                forecaster = TimeSeriesForecaster(model_name=name)
                forecaster.fit(train_df, feature_cols=feature_cols)
                y_pred = forecaster.predict(test_df)

                metrics = calculate_metrics(y_test, y_pred)
                self.models[name] = forecaster
                self.results[name] = {
                    "forecaster": forecaster,
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "test_df": test_df
                }

                leaderboard.append({
                    "Model": name,
                    "MAE ($)": metrics["MAE"],
                    "RMSE ($)": metrics["RMSE"],
                    "MAPE (%)": metrics["MAPE (%)"],
                    "WAPE (%)": metrics["WAPE (%)"],
                    "R2 Score": metrics["R2 Score"],
                    "Forecast Bias ($)": metrics["Bias ($)"]
                })
            except Exception as e:
                print(f"Error training {name}: {e}")

        df_leaderboard = pd.DataFrame(leaderboard)
        if not df_leaderboard.empty:
            # Sort by WAPE (or RMSE)
            df_leaderboard = df_leaderboard.sort_values(by=["WAPE (%)", "RMSE ($)"]).reset_index(drop=True)
            self.best_model_name = df_leaderboard.iloc[0]["Model"]

        return df_leaderboard

    def get_best_model(self) -> Optional[TimeSeriesForecaster]:
        """
        Returns the top-performing model instance.
        """
        if self.best_model_name and self.best_model_name in self.models:
            return self.models[self.best_model_name]
        return None
