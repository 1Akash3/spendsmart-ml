"""Personalized recommendation engine + real-time optimizer.

Turns forecasts + a user's own history + their cohort into ranked, money-quantified actions.
Nothing here is a generic rule: every threshold is derived from *this* user's baseline mean and
volatility, and peer comparisons use *their* cohort's norms.

Two layers:
  • `PersonalizedRecommender.recommend(...)` — batch: deviation detection, peer context,
    subscription-creep detection, and a savings-goal reallocation optimizer.
  • `RealTimeState` — online: O(#categories) update as each new transaction lands, projecting
    month-end spend and re-firing alerts without any retrain.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from config import (
    CURRENCY,
    DISCRETIONARY_CATEGORIES,
    EXPENSE_CATEGORIES,
    CATEGORY_LABELS,
    FLEX_STD_MULT,
    OVERSPEND_Z,
    TOP_K_RECOMMENDATIONS,
)

BASELINE_WINDOW = 6  # months of recent history used to define "your normal"


@dataclass
class Recommendation:
    kind: str            # overspend | peer | subscription_creep | reallocation | positive
    category: str
    title: str
    detail: str
    monthly_impact: float   # ₹ potential monthly saving (>=0)
    confidence: float       # 0..1
    priority: float = field(default=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category_label"] = CATEGORY_LABELS.get(self.category, self.category)
        return d


def _fmt(amount: float) -> str:
    return f"{CURRENCY}{amount:,.0f}"


class PersonalizedRecommender:
    def __init__(self, currency: str = CURRENCY) -> None:
        self.currency = currency

    # ------------------------------------------------------------------ baselines
    @staticmethod
    def _baselines(panel_user: pd.DataFrame) -> tuple[dict, dict, float]:
        """Return (mean, std) per category over the recent window + mean income."""
        g = panel_user.sort_values("month").tail(BASELINE_WINDOW)
        mean = {c: float(g[c].mean()) for c in EXPENSE_CATEGORIES}
        std = {c: float(g[c].std(ddof=0)) for c in EXPENSE_CATEGORIES}
        income = float(g["income"].mean())
        return mean, std, income

    # ------------------------------------------------------------------ main API
    def recommend(
        self,
        panel_user: pd.DataFrame,
        forecast: dict[str, float],
        cohort_norms_row: pd.Series | None = None,
        savings_goal_rate: float | None = None,
        history_months: int | None = None,
    ) -> dict:
        """Produce a personalized recommendation bundle for one user.

        panel_user        : that user's monthly panel rows.
        forecast          : {category: predicted next-month spend}.
        cohort_norms_row  : optional Series of cohort median category *shares* ("users like you").
        savings_goal_rate : optional target savings as a fraction of income (e.g. 0.25).
        """
        mean, std, income = self._baselines(panel_user)
        n_months = history_months if history_months is not None else len(panel_user)
        # Confidence grows with available history (more months → more trustworthy baseline).
        hist_conf = float(np.clip(n_months / 12.0, 0.3, 1.0))

        recs: list[Recommendation] = []
        forecast = {c: float(forecast.get(c, mean[c])) for c in EXPENSE_CATEGORIES}
        proj_expense = sum(forecast.values())

        # (1) Deviation from the user's OWN normal.
        for c in EXPENSE_CATEGORIES:
            threshold = mean[c] + OVERSPEND_Z * std[c]
            if std[c] > 0 and forecast[c] > threshold and forecast[c] - mean[c] > 0.02 * income:
                impact = forecast[c] - mean[c]
                z = (forecast[c] - mean[c]) / (std[c] + 1e-9)
                recs.append(Recommendation(
                    kind="overspend", category=c,
                    title=f"{CATEGORY_LABELS[c]} trending above your usual",
                    detail=(f"Projected {_fmt(forecast[c])} next month vs your typical "
                            f"{_fmt(mean[c])} ({z:.1f}σ high). Trimming here saves ~{_fmt(impact)}."),
                    monthly_impact=impact,
                    confidence=hist_conf * float(np.clip(z / 3, 0.3, 1.0)),
                ))

        # (2) Peer context — only for discretionary categories, framed against the user's cohort.
        if cohort_norms_row is not None and proj_expense > 0:
            for c in DISCRETIONARY_CATEGORIES:
                user_share = forecast[c] / proj_expense
                peer_share = float(cohort_norms_row.get(c, user_share))
                if peer_share > 0 and user_share > peer_share * 1.3 and forecast[c] > 0.03 * income:
                    impact = (user_share - peer_share) * proj_expense
                    recs.append(Recommendation(
                        kind="peer", category=c,
                        title=f"{CATEGORY_LABELS[c]} is high for users like you",
                        detail=(f"You put {user_share*100:.0f}% of spend here vs "
                                f"{peer_share*100:.0f}% for your cohort. Aligning frees ~{_fmt(impact)}/mo."),
                        monthly_impact=impact,
                        confidence=hist_conf * 0.7,
                    ))

        # (3) Subscription creep — compare forecast vs 3-months-ago level.
        g = panel_user.sort_values("month")
        if len(g) >= 4:
            past = float(g["subscriptions"].iloc[-4:-1].mean())
            if past > 0 and forecast["subscriptions"] > past * 1.25:
                impact = forecast["subscriptions"] - past
                recs.append(Recommendation(
                    kind="subscription_creep", category="subscriptions",
                    title="Subscription spend is creeping up",
                    detail=(f"Recurring charges up ~{(forecast['subscriptions']/past-1)*100:.0f}% "
                            f"vs 3 months ago. Audit unused subscriptions to reclaim ~{_fmt(impact)}."),
                    monthly_impact=impact,
                    confidence=hist_conf * 0.8,
                ))

        # (4) Savings-goal reallocation optimizer (bounded by the user's OWN volatility).
        plan = None
        if savings_goal_rate is not None and income > 0:
            plan = self._reallocation_plan(forecast, mean, std, income, savings_goal_rate)
            if plan["shortfall_closed"] > 0:
                recs.append(Recommendation(
                    kind="reallocation", category="misc",
                    title=f"Plan to hit a {savings_goal_rate*100:.0f}% savings goal",
                    detail=plan["summary"],
                    monthly_impact=plan["shortfall_closed"],
                    confidence=hist_conf * 0.9,
                ))

        # (5) Positive reinforcement if genuinely on track.
        proj_savings_rate = (income - proj_expense) / income if income > 0 else 0.0
        if not recs and proj_savings_rate > 0.2:
            recs.append(Recommendation(
                kind="positive", category="income",
                title="You're on track",
                detail=(f"Projected to save {proj_savings_rate*100:.0f}% of income next month "
                        f"with no category running hot. Consider auto-investing the surplus."),
                monthly_impact=0.0, confidence=hist_conf,
            ))

        # Rank by impact × confidence.
        for r in recs:
            r.priority = r.monthly_impact * (0.5 + 0.5 * r.confidence)
        recs.sort(key=lambda r: r.priority, reverse=True)
        recs = recs[:TOP_K_RECOMMENDATIONS]

        return {
            "summary": {
                "avg_income": round(income, 2),
                "projected_expense": round(proj_expense, 2),
                "projected_savings_rate": round(proj_savings_rate, 4),
                "history_months": n_months,
            },
            "forecast": {c: round(forecast[c], 2) for c in EXPENSE_CATEGORIES},
            "recommendations": [r.to_dict() for r in recs],
            "plan": plan,
        }

    # ------------------------------------------------------------------ optimizer
    @staticmethod
    def _reallocation_plan(forecast, mean, std, income, goal_rate) -> dict:
        """Greedy water-filling: cut discretionary categories within ±FLEX_STD_MULT·std to
        close the gap to the savings goal, preferring the most flexible categories first."""
        target_savings = goal_rate * income
        projected_savings = income - sum(forecast.values())
        gap = target_savings - projected_savings  # >0 → need to cut this much

        cuts: dict[str, float] = {}
        if gap <= 0:
            return {"target_savings": target_savings, "projected_savings": projected_savings,
                    "gap": gap, "cuts": cuts, "shortfall_closed": 0.0, "feasible": True,
                    "summary": "Already projected to meet the goal — no cuts needed."}

        # Flexibility per category = min(volatility headroom, sensible fraction of the forecast).
        flex = {}
        for c in DISCRETIONARY_CATEGORIES:
            headroom = FLEX_STD_MULT * std[c] + 0.10 * mean[c]
            flex[c] = float(min(headroom, 0.4 * forecast[c]))
        # Cut the most flexible first.
        remaining = gap
        for c in sorted(flex, key=flex.get, reverse=True):
            if remaining <= 0:
                break
            cut = min(flex[c], remaining)
            if cut > 0.005 * income:
                cuts[c] = round(cut, 2)
                remaining -= cut

        closed = gap - max(remaining, 0)
        feasible = remaining <= 1e-6
        parts = ", ".join(f"{CATEGORY_LABELS[c]} −{_fmt(v)}" for c, v in cuts.items())
        if feasible:
            summary = f"Reallocate {_fmt(closed)}/mo ({parts}) to reach {_fmt(target_savings)} saved."
        else:
            summary = (f"Cutting {parts} recovers {_fmt(closed)}/mo, but the goal needs "
                       f"{_fmt(gap)} — consider a gentler target or boosting income.")
        return {"target_savings": round(target_savings, 2),
                "projected_savings": round(projected_savings, 2),
                "gap": round(gap, 2), "cuts": cuts,
                "shortfall_closed": round(closed, 2), "feasible": feasible,
                "summary": summary}


class RealTimeState:
    """Online per-month tracker. Feed it transactions as they arrive; it projects month-end
    spend and fires alerts in O(#categories) — no retrain, suitable for live use."""

    def __init__(self, baseline_mean: dict, baseline_std: dict, income: float,
                 days_in_month: int = 30) -> None:
        self.mean = baseline_mean
        self.std = baseline_std
        self.income = income
        self.days_in_month = days_in_month
        self.month_spend = {c: 0.0 for c in EXPENSE_CATEGORIES}
        self.max_day_seen = 1

    def update(self, category: str, amount: float, day_of_month: int) -> list[dict]:
        """Register a transaction and return any alerts triggered for that category."""
        if category not in self.month_spend:
            category = "misc"
        self.month_spend[category] += float(amount)
        self.max_day_seen = max(self.max_day_seen, int(day_of_month))

        alerts = []
        projected = self._project(category, day_of_month)
        threshold = self.mean[category] + OVERSPEND_Z * self.std[category]
        if self.std[category] > 0 and projected > threshold:
            alerts.append({
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "message": (f"On pace for {_fmt(projected)} in {CATEGORY_LABELS.get(category, category)} "
                            f"this month — above your usual {_fmt(self.mean[category])}."),
                "projected": round(projected, 2),
                "baseline": round(self.mean[category], 2),
            })
        return alerts

    def _project(self, category: str, day_of_month: int) -> float:
        frac = max(int(day_of_month), self.max_day_seen) / self.days_in_month
        frac = min(max(frac, 1e-3), 1.0)
        return self.month_spend[category] / frac

    def projected_month_end(self) -> dict[str, float]:
        return {c: round(self._project(c, self.max_day_seen), 2) for c in EXPENSE_CATEGORIES}
