import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ARIMABaselineForecaster:
    """Statistical ARIMA Baseline.
    Fallback to simple exponential smoothing if statsmodels is unavailable or fitting fails.
    """
    def __init__(self, order=(1, 1, 0)):
        self.order = order
        self.user_models = {}
        
    def fit(self, df: pd.DataFrame):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self._use_statsmodels = True
        except ImportError:
            self._use_statsmodels = False
            
        for user_cat, group in df.groupby(['user_id', 'category']):
            # Sort by time
            ts = group.sort_values('month')['target'].values
            
            if len(ts) < 3:
                # Too short for ARIMA
                self.user_models[user_cat] = np.mean(ts) if len(ts) > 0 else 0.0
                continue
                
            if self._use_statsmodels:
                from statsmodels.tsa.arima.model import ARIMA
                try:
                    model = ARIMA(ts, order=self.order)
                    res = model.fit()
                    # Store the forecasted next step
                    self.user_models[user_cat] = res.forecast(steps=1)[0]
                except Exception:
                    # Fallback to naive lag
                    self.user_models[user_cat] = ts[-1]
            else:
                # Simple exponential smoothing fallback
                alpha = 0.3
                s = ts[0]
                for x in ts[1:]:
                    s = alpha * x + (1 - alpha) * s
                self.user_models[user_cat] = s
                
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = []
        for _, row in df.iterrows():
            key = (row['user_id'], row['category'])
            pred = self.user_models.get(key, row.get('lag_1', 0.0))
            preds.append(max(0.0, pred))
        return pd.Series(preds, index=df.index)

class XGBoostBaselineForecaster:
    """XGBoost Baseline."""
    def __init__(self):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            self._available = True
        except ImportError:
            # Fallback to HistGradientBoosting
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
            self._available = False
            
    def fit(self, df: pd.DataFrame):
        # We assume df has the standard lag/roll features from `build_forecast_frame`
        from src.features import feature_columns
        
        self.cat_means = df.groupby("category")["target"].mean().to_dict()
        self.global_mean = df["target"].mean()
        
        X = df[feature_columns]
        y = df["target"]
        
        if len(X) > 0:
            self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        from src.features import feature_columns
        X = df[feature_columns]
        if len(X) == 0:
            return pd.Series([], index=df.index)
            
        preds = self.model.predict(X)
        return pd.Series(np.maximum(0.0, preds), index=df.index)
