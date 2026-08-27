import pandas as pd
import numpy as np
from typing import Dict, Any

class FinancialHealthScorer:
    """Computes a decomposable financial health score."""
    def __init__(self, essential_categories=None):
        if essential_categories is None:
            self.essential_categories = {'housing', 'utilities', 'groceries', 'healthcare', 'transportation'}
        else:
            self.essential_categories = essential_categories

    def score_user(self, user_panel: pd.DataFrame) -> Dict[str, float]:
        """
        user_panel: Monthly aggregated data for a single user containing 
        category spends, total income, and total expense.
        """
        if len(user_panel) == 0:
            return {"overall_score": 0.0}

        # 1. Cash-flow stability (Income vs Expense volatility)
        net_cash_flow = user_panel['income'] - user_panel['target']
        cash_flow_stability = max(0, 100 - (np.std(net_cash_flow) / (np.mean(user_panel['income']) + 1e-9) * 100))

        # 2. Savings consistency (Months with positive net cash flow)
        savings_consistency = (net_cash_flow > 0).mean() * 100

        # 3. Essential-spend burden
        essential_spend = sum([user_panel[c] for c in self.essential_categories if c in user_panel.columns])
        essential_burden_ratio = essential_spend / (user_panel['income'] + 1e-9)
        essential_burden_score = max(0, 100 - (np.mean(essential_burden_ratio) * 100))

        # Overall Score (Weighted average)
        overall = (cash_flow_stability * 0.3 + 
                   savings_consistency * 0.4 + 
                   essential_burden_score * 0.3)
                   
        return {
            "overall_score": min(100.0, max(0.0, float(overall))),
            "cash_flow_stability": float(cash_flow_stability),
            "savings_consistency": float(savings_consistency),
            "essential_burden_score": float(essential_burden_score)
        }
