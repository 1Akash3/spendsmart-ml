import pandas as pd
import numpy as np

class TabularExplainer:
    """Provides SHAP and Permutation Importance explanations for tabular baselines."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def permutation_importance(self, X_val: pd.DataFrame, y_val: pd.Series, metric_fn, n_repeats=5) -> pd.DataFrame:
        """Computes permutation importance."""
        base_score = metric_fn(y_val, self.model.predict(X_val))
        importances = []
        
        for col in self.feature_names:
            scores = []
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                X_perm[col] = np.random.permutation(X_perm[col])
                perm_score = metric_fn(y_val, self.model.predict(X_perm))
                # For error metrics, higher error = more important
                scores.append(perm_score - base_score)
            importances.append({
                "feature": col,
                "importance_mean": np.mean(scores),
                "importance_std": np.std(scores)
            })
            
        return pd.DataFrame(importances).sort_values("importance_mean", ascending=False)
        
    def shap_values(self, X: pd.DataFrame):
        """Wrapper for SHAP TreeExplainer."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(X)
            return shap_vals
        except ImportError:
            return None

class NeuralExplainer:
    """Provides feature masking / controlled perturbation for PATFormer."""
    def __init__(self, model):
        self.model = model
        
    def feature_masking_importance(self, cat_input, amt_input, ctx_input, target_task="category"):
        """
        Computes the drop in performance when zeroing out inputs.
        """
        # Baseline
        base_preds = self.model(cat_input, amt_input, ctx_input)[target_task]
        
        importances = {}
        
        # Mask amounts
        amt_masked = amt_input.clone()
        amt_masked.fill_(0.0)
        masked_amt_preds = self.model(cat_input, amt_masked, ctx_input)[target_task]
        importances['amount_input'] = float((base_preds - masked_amt_preds).abs().mean())
        
        # Mask context
        ctx_masked = ctx_input.clone()
        ctx_masked.fill_(0.0)
        masked_ctx_preds = self.model(cat_input, amt_input, ctx_masked)[target_task]
        importances['context_input'] = float((base_preds - masked_ctx_preds).abs().mean())
        
        return importances
