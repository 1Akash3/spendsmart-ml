import numpy as np
import pandas as pd
from typing import Tuple, List, Callable

class QuantileForecaster:
    """A. Quantile prediction without post-hoc calibration.
    Uses basic pinball loss logic or quantile gradient boosting."""
    def __init__(self, model_class, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {q: model_class(quantile=q) for q in quantiles}
        
    def fit(self, X, y):
        for q, model in self.models.items():
            model.fit(X, y)
        return self
        
    def predict(self, X) -> pd.DataFrame:
        preds = {f"P{int(q*100)}": self.models[q].predict(X) for q in self.quantiles}
        return pd.DataFrame(preds)

class CalibratedQuantileForecaster:
    """B. Quantile prediction + calibration (e.g. Isotonic Regression on held-out data)."""
    def __init__(self, base_quantile_forecaster):
        self.base = base_quantile_forecaster
        self.calibrators = {}
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        from sklearn.isotonic import IsotonicRegression
        
        # Fit base models on train
        self.base.fit(X_train, y_train)
        
        # Predict on cal
        cal_preds = self.base.predict(X_cal)
        
        # Fit calibrator per quantile
        for col in cal_preds.columns:
            # We calibrate the predicted quantile values against the empirical CDF or 
            # adjust the values to guarantee coverage.
            # Simplified isotonic mapping for values:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(cal_preds[col], y_cal)
            self.calibrators[col] = iso
            
        return self
        
    def predict(self, X) -> pd.DataFrame:
        raw_preds = self.base.predict(X)
        cal_preds = {}
        for col in raw_preds.columns:
            cal_preds[col] = self.calibrators[col].predict(raw_preds[col])
        return pd.DataFrame(cal_preds)

class ConformalPredictor:
    """C. Conformal-style calibration.
    Uses nonconformity scores on a calibration set to construct valid intervals."""
    def __init__(self, base_point_forecaster, alpha=0.1):
        self.base = base_point_forecaster
        self.alpha = alpha
        self.q_hat = None
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        self.base.fit(X_train, y_train)
        preds_cal = self.base.predict(X_cal)
        
        # Absolute residuals (nonconformity scores)
        residuals = np.abs(y_cal - preds_cal)
        n = len(residuals)
        
        # Compute the quantile q_hat
        val = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(residuals, min(val, 1.0))
        return self
        
    def predict(self, X) -> pd.DataFrame:
        point_preds = self.base.predict(X)
        # Returns [lower, point, upper]
        return pd.DataFrame({
            "P10": point_preds - self.q_hat,
            "P50": point_preds,
            "P90": point_preds + self.q_hat
        })

def evaluate_uncertainty(y_true: np.ndarray, y_pred_df: pd.DataFrame) -> dict:
    """
    Evaluates probabilistic predictions.
    y_pred_df must contain 'P10', 'P50', 'P90'.
    """
    p10, p50, p90 = y_pred_df["P10"], y_pred_df["P50"], y_pred_df["P90"]
    
    # Empirical Coverage
    coverage_80 = np.mean((y_true >= p10) & (y_true <= p90))
    
    # Sharpness (interval width)
    width = np.mean(p90 - p10)
    
    # Pinball loss
    def pinball(y, q_pred, q):
        err = y - q_pred
        return np.mean(np.maximum(q * err, (q - 1) * err))
        
    loss_10 = pinball(y_true, p10, 0.1)
    loss_50 = pinball(y_true, p50, 0.5)
    loss_90 = pinball(y_true, p90, 0.9)
    
    # Calibration error (target 80% coverage)
    cal_err = np.abs(coverage_80 - 0.80)
    
    return {
        "empirical_coverage": float(coverage_80),
        "sharpness": float(width),
        "calibration_error": float(cal_err),
        "pinball_10": float(loss_10),
        "pinball_50": float(loss_50),
        "pinball_90": float(loss_90),
    }
