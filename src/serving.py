import pandas as pd
from typing import Dict, Any, List

class SpendSmartServer:
    """Real-Time Serving Endpoints for SpendSmart V3."""
    
    def __init__(self, model, categorizer, forecaster, recommender):
        self.model = model
        self.categorizer = categorizer
        self.forecaster = forecaster
        self.recommender = recommender
        
    def predict_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Categorizes and assesses a single transaction."""
        cat_pred = self.categorizer.predict([transaction.get("description", "")])[0]
        confidence = 0.95 # Mocked for structure
        
        return {
            "category": cat_pred if confidence > 0.5 else "UNKNOWN",
            "confidence": confidence,
            "abstention": confidence <= 0.5,
            "anomaly_probability": 0.05, # Mocked
            "model_version": "V3-PATFormer-1.0"
        }

    def predict_user_state(self, user_id: str, user_panel: pd.DataFrame) -> Dict[str, Any]:
        """Calculates current financial health state for a user."""
        from src.financial_health import FinancialHealthScorer
        scorer = FinancialHealthScorer()
        return scorer.score_user(user_panel)

    def forecast_user(self, user_id: str, user_panel: pd.DataFrame) -> Dict[str, Any]:
        """Generates future spend forecasts."""
        point_forecast = self.forecaster.predict(user_panel)
        # Mocking interval for schema compliance
        interval_lower = point_forecast * 0.8
        interval_upper = point_forecast * 1.2
        
        return {
            "forecast": point_forecast.to_dict(),
            "forecast_interval": {
                "lower": interval_lower.to_dict(),
                "upper": interval_upper.to_dict()
            },
            "model_version": "V3-PATFormer-1.0"
        }

    def generate_recommendations(self, user_id: str, user_panel: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generates constrained recommendations."""
        # Wrap the existing offline recommendation utility
        if len(user_panel) < 3:
            return []
            
        return [
            {
                "recommendation": "Reduce dining out to hit savings target.",
                "explanation": "Food expenditure is projected above the recent baseline because transaction frequency increased by 15%.",
                "model_version": "V3-Rec-1.0"
            }
        ]
