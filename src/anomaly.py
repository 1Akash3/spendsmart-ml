import numpy as np
import pandas as pd

class BehavioralAnomalyInjector:
    """Injects controlled behavioral anomalies if real labels are unavailable.
    Never refer to these as 'fraud'."""
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        
    def inject(self, df: pd.DataFrame, injection_rate=0.01) -> pd.DataFrame:
        """
        Injects sudden spikes or out-of-distribution transactions.
        Creates a 'is_injected_anomaly' label.
        """
        out = df.copy()
        if 'is_injected_anomaly' not in out.columns:
            out['is_injected_anomaly'] = False
            
        n_inject = int(len(out) * injection_rate)
        indices = self.rng.choice(out.index, n_inject, replace=False)
        
        for idx in indices:
            # 50% chance to multiply amount by 10x
            # 50% chance to change category to something unseen for the user
            if self.rng.random() < 0.5:
                out.loc[idx, 'amount'] *= self.rng.uniform(5.0, 15.0)
            else:
                out.loc[idx, 'category'] = "injected_anomalous_category"
                
            out.loc[idx, 'is_injected_anomaly'] = True
            
        return out

class ForecastResidualAnomalyDetector:
    """Detects anomalies by comparing actual spend to forecast intervals."""
    def __init__(self, z_threshold=3.0):
        self.z_threshold = z_threshold
        
    def predict(self, y_true: pd.Series, p50: pd.Series, p90: pd.Series, p10: pd.Series) -> pd.Series:
        """
        Uses the distance from the median normalized by the interval width (p90 - p10) as a Z-score equivalent.
        """
        width = np.maximum(p90 - p10, 1.0) # Avoid div by zero
        z_equiv = np.abs(y_true - p50) / (width / 2.0)
        
        return z_equiv > self.z_threshold
