"""End-to-end training + evaluation orchestrator.

Run:  python -m src.train            (from the project root)
      python -m src.train --users 300 --months 15 --real-categorizer

Steps: generate/load data → categorizer → panel + profiles → segmentation → forecaster backtest
→ overspend-detector eval → fit final models → sample personalized recommendation → save
artifacts + reports/metrics.json.
"""
from __future__ import annotations

# --- path bootstrap so `import config` / `import features` work when run directly --------------
import os as _os
import sys as _sys
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:  # Windows consoles default to cp1252 and choke on ₹ / → ; force UTF-8 where supported.
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
# ----------------------------------------------------------------------------------------------

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from categorizer import TransactionCategorizer
from evaluate import backtest_forecaster, evaluate_overspend_detector
from features import (
    PROFILE_FEATURE_COLS,
    build_forecast_frame,
    build_monthly_panel,
    build_serving_frame,
    build_user_profiles,
)
from forecaster import PersonalizedForecaster
from recommender import PersonalizedRecommender
from segmentation import UserSegmenter
from synth import generate_transactions
from data_sources import (load_real_transactions, load_all_real_transactions,
                          load_all_labeled_descriptions)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_static_features(profiles, segmenter) -> pd.DataFrame:
    """profiles + cohort id, as a merge-ready frame keyed by user_id."""
    assigned = segmenter.assign(profiles)
    static = assigned[PROFILE_FEATURE_COLS + ["cohort"]].reset_index()
    return static


def run(source: str = "combined", users: int = config.SYNTH_N_USERS, months: int = config.SYNTH_MONTHS,
        demo_goal_rate: float = 0.25, seed: int = config.RANDOM_SEED,
        sample_users: int | None = None, real_categorizer: bool = False) -> dict:
    metrics: dict = {"config": {"source": source, "seed": seed}}

    # 1. Data ---------------------------------------------------------------------------------
    labeled = None
    if source == "combined":
        _log("Loading ALL REAL data (credit-card + personal-finance-tracker) …")
        txns = load_all_real_transactions(sample_users=sample_users)
        labeled = load_all_labeled_descriptions()  # best category labels from every source
    elif source == "real":
        _log("Loading REAL transactions (Kaggle credit-card dataset) …")
        txns = load_real_transactions(sample_users=sample_users)
    else:
        _log(f"Generating synthetic transactions: {users} users × {months} months …")
        txns = generate_transactions(n_users=users, months=months, seed=seed)
    _log(f"  {len(txns):,} transactions, {txns.user_id.nunique()} users, "
         f"{txns.category.nunique()} categories  [source={source}]")

    # 2. Categorizer --------------------------------------------------------------------------
    _log("Training transaction categorizer (TF-IDF word+char → logistic regression) …")
    cdf = (labeled if labeled is not None else txns[["description", "category"]]).dropna()
    # cap rows for a fast, memory-safe fit on large real datasets (merchants repeat)
    if len(cdf) > 200_000:
        cdf = cdf.sample(200_000, random_state=seed)
    desc, labels = cdf["description"].values, cdf["category"].values
    Xtr, Xte, ytr, yte = train_test_split(desc, labels, test_size=0.2,
                                          random_state=seed, stratify=labels)
    cat = TransactionCategorizer().fit(Xtr, ytr)
    cat_metrics = cat.evaluate(Xte, yte)
    metrics["categorizer"] = {k: cat_metrics[k] for k in
                              ("accuracy", "macro_f1", "weighted_f1", "n_test")}
    _log(f"  accuracy={cat_metrics['accuracy']:.3f}  macro_f1={cat_metrics['macro_f1']:.3f}")

    # 3. Panel + profiles ---------------------------------------------------------------------
    _log("Building monthly panel + user profiles …")
    panel = build_monthly_panel(txns)
    panel["income"] = panel["income"].fillna(0.0)  # real card data carries no income stream
    profiles = build_user_profiles(panel)
    profiles = profiles.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _log(f"  panel rows={len(panel):,}  users profiled={len(profiles)}")

    # 4. Segmentation -------------------------------------------------------------------------
    _log(f"Segmenting users into {config.N_COHORTS} cohorts (KMeans) …")
    seg = UserSegmenter().fit(profiles)
    metrics["segmentation"] = {"n_cohorts": seg.n_cohorts,
                               "silhouette": round(seg.silhouette_, 4)}
    static = build_static_features(profiles, seg)
    _log(f"  silhouette={seg.silhouette_:.3f}")

    # 5. Forecaster backtest (proven accuracy vs naive) ---------------------------------------
    _log("Back-testing personalized forecaster (train past → predict held-out months) …")
    fc_metrics = backtest_forecaster(panel, static)
    metrics["forecaster"] = fc_metrics
    _log(f"  MAE model={fc_metrics.get('model_mae')} vs naive={fc_metrics.get('naive_mae')} "
         f"| skill_vs_naive={fc_metrics.get('skill_vs_naive')}")

    # 6. Recommendation quality (overspend detector) ------------------------------------------
    _log("Evaluating overspend detector (recommendation quality) …")
    rec_metrics = evaluate_overspend_detector(panel, static)
    metrics["recommender_eval"] = rec_metrics
    if "model" in rec_metrics:
        _log(f"  overspend F1 model={rec_metrics['model']['f1']} "
             f"vs naive={rec_metrics['naive_baseline']['f1']}")

    # 7. Fit final forecaster on ALL history + build next-month forecasts ----------------------
    _log("Fitting final forecaster on full history …")
    full_frame = build_forecast_frame(panel, static_features=static)
    forecaster = PersonalizedForecaster().fit(full_frame)
    serving = build_serving_frame(panel, static_features=static)
    forecasts = forecaster.predict_by_user_category(serving)

    # 8. Sample personalized recommendation (prefer a user with actionable alerts) -------------
    recommender = PersonalizedRecommender()
    demo_uid, demo, demo_cohort = None, None, None
    for cand in profiles.index[:600]:
        cpanel = panel[panel["user_id"] == cand]
        if len(cpanel) < 6 or cand not in forecasts:
            continue
        ccoh = int(static.loc[static["user_id"] == cand, "cohort"].iloc[0])
        out = recommender.recommend(cpanel, forecasts.get(cand, {}),
                                    cohort_norms_row=seg.cohort_category_norms_.loc[ccoh],
                                    savings_goal_rate=demo_goal_rate)
        if demo is None:                       # first valid candidate = fallback
            demo_uid, demo, demo_cohort = cand, out, ccoh
        if any(r["kind"] in ("overspend", "peer") for r in out["recommendations"]):
            demo_uid, demo, demo_cohort = cand, out, ccoh
            break
    metrics["sample_recommendation"] = {"user_id": str(demo_uid), "cohort": demo_cohort, **demo}

    # 9. Persist ------------------------------------------------------------------------------
    _log("Saving artifacts + metrics …")
    cat.save()
    seg.save()
    forecaster.save()
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    _print_summary(metrics, demo)
    return metrics


def _print_summary(metrics: dict, demo: dict) -> None:
    print("\n" + "=" * 74)
    print("  RESULTS SUMMARY".center(74))
    print("=" * 74)
    c = metrics["categorizer"]
    print(f"  Categorizer     accuracy={c['accuracy']:.3f}  macro-F1={c['macro_f1']:.3f}")
    s = metrics["segmentation"]
    print(f"  Segmentation    {s['n_cohorts']} cohorts  silhouette={s['silhouette']:.3f}")
    f = metrics["forecaster"]
    if "model_mae" in f:
        print(f"  Forecaster      MAE={f['model_mae']}  (naive={f['naive_mae']})  "
              f"WAPE={f.get('model_wape')}  skill_vs_naive={f['skill_vs_naive']:+.1%}")
    r = metrics.get("recommender_eval", {})
    if "model" in r:
        print(f"  Overspend det.  F1={r['model']['f1']}  (naive baseline F1={r['naive_baseline']['f1']})")
    print("-" * 74)
    print(f"  Sample personalized recommendation (user {metrics['sample_recommendation']['user_id']}, "
          f"cohort {metrics['sample_recommendation']['cohort']}):")
    sm = demo["summary"]
    print(f"    income≈{config.CURRENCY}{sm['avg_income']:,.0f}  "
          f"projected save rate={sm['projected_savings_rate']*100:.1f}%")
    for i, rec in enumerate(demo["recommendations"], 1):
        print(f"    {i}. [{rec['kind']}] {rec['title']}")
        print(f"       {rec['detail']}")
    if not demo["recommendations"]:
        print("    (no actionable flags — user is on track)")
    print("=" * 74)
    print(f"  Full metrics → {config.REPORTS_DIR / 'metrics.json'}")
    print(f"  Artifacts    → {config.ARTIFACTS_DIR}")
    print("=" * 74 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the personalized spending recommender.")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic generator instead of REAL Kaggle data (offline, no creds)")
    ap.add_argument("--cc-only", action="store_true",
                    help="real credit-card dataset only (skip the personal-finance-tracker enrichment)")
    ap.add_argument("--sample-users", type=int, default=None,
                    help="cap number of real users (for a faster run)")
    ap.add_argument("--users", type=int, default=config.SYNTH_N_USERS, help="synthetic mode only")
    ap.add_argument("--months", type=int, default=config.SYNTH_MONTHS, help="synthetic mode only")
    ap.add_argument("--goal-rate", type=float, default=0.25,
                    help="demo savings-goal as a fraction of income")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()
    src = "synthetic" if args.synthetic else ("real" if args.cc_only else "combined")
    run(source=src, users=args.users, months=args.months, sample_users=args.sample_users,
        demo_goal_rate=args.goal_rate, seed=args.seed)


if __name__ == "__main__":
    main()
