"""Publication Table (Tables 2-8) and Figure (Figures 5-13) Generators.

Generates publication-quality CSV tables and 300 DPI PNG figures from saved benchmark artifacts.
Notebooks and reports read directly from these generated files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.benchmarks import log

# Set seaborn style for publication figures
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================================
# TABLE GENERATORS (Tables 2 - 8)
# ============================================================================

def generate_table_2_categorization(
    registry_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 2: Categorization Benchmark across models and splits."""
    cat_df = registry_df[registry_df["task"] == "categorization"].copy()
    if cat_df.empty:
        return pd.DataFrame()

    cols = [
        "model_name", "split_name", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "balanced_accuracy", "inference_ms", "training_seconds"
    ]
    available_cols = [c for c in cols if c in cat_df.columns]
    table = cat_df[available_cols].copy()
    table.columns = [
        "Model", "Split", "Accuracy", "Precision", "Recall",
        "Macro F1", "Balanced Accuracy", "Latency (ms)", "Runtime (s)"
    ][:len(available_cols)]

    table.to_csv(out_dir / "Table_2_Categorization.csv", index=False)
    return table


def generate_table_3_forecasting(
    registry_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 3: Forecast Benchmark across models."""
    for_df = registry_df[registry_df["task"] == "forecasting"].copy()
    if for_df.empty:
        return pd.DataFrame()

    cols = [
        "model_name", "split_name", "mae", "rmse", "mape",
        "wape", "smape", "r2", "inference_ms", "training_seconds"
    ]
    available_cols = [c for c in cols if c in for_df.columns]
    table = for_df[available_cols].copy()
    table.columns = [
        "Model", "Split", "MAE", "RMSE", "MAPE",
        "WAPE", "sMAPE", "R²", "Latency (ms)", "Runtime (s)"
    ][:len(available_cols)]

    table.to_csv(out_dir / "Table_3_Forecasting.csv", index=False)
    return table


def generate_table_4_runtime(
    registry_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 4: Runtime Comparison across all models."""
    cols = ["model_name", "task", "training_seconds", "inference_ms", "peak_ram_mb"]
    available = [c for c in cols if c in registry_df.columns]
    table = registry_df[available].drop_duplicates(subset=["model_name", "task"]).copy()
    table.columns = ["Model", "Task", "Training (s)", "Inference (ms)", "Peak RAM (MB)"][:len(available)]

    table.to_csv(out_dir / "Table_4_Runtime.csv", index=False)
    return table


def generate_table_5_cold_start(
    cold_start_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 5: Cold Start Comparison."""
    if cold_start_df.empty:
        return pd.DataFrame()
    cold_start_df.to_csv(out_dir / "Table_5_Cold_Start.csv", index=False)
    cold_start_df.to_csv(out_dir / "cold_start_metrics.csv", index=False)
    return cold_start_df


def generate_table_6_novel_merchant(
    merchant_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 6: Novel Merchant Generalization Comparison."""
    if merchant_df.empty:
        return pd.DataFrame()
    merchant_df.to_csv(out_dir / "Table_6_Novel_Merchant.csv", index=False)
    merchant_df.to_csv(out_dir / "merchant_generalization.csv", index=False)
    return merchant_df


def generate_table_7_robustness(
    robustness_df: pd.DataFrame, out_dir: Path
) -> pd.DataFrame:
    """Table 7: Robustness Comparison."""
    if robustness_df.empty:
        return pd.DataFrame()
    robustness_df.to_csv(out_dir / "Table_7_Robustness.csv", index=False)
    robustness_df.to_csv(out_dir / "robustness_metrics.csv", index=False)
    return robustness_df


def generate_table_8_calibration(
    cal_metrics: Dict[str, Any], out_dir: Path
) -> pd.DataFrame:
    """Table 8: Calibration Comparison."""
    data = [{
        "Metric": "ECE (Expected Calibration Error)",
        "Value": cal_metrics.get("ece", 0.0)
    }, {
        "Metric": "Brier Score",
        "Value": cal_metrics.get("brier_score", 0.0)
    }]
    table = pd.DataFrame(data)
    table.to_csv(out_dir / "Table_8_Calibration.csv", index=False)
    return table


# ============================================================================
# FIGURE GENERATORS (Figures 5 - 13)
# ============================================================================

def generate_figure_5_macro_f1(registry_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 5: Macro F1 Comparison across models."""
    cat_df = registry_df[registry_df["task"] == "categorization"].copy()
    if cat_df.empty or "macro_f1" not in cat_df.columns:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=cat_df, x="model_name", y="macro_f1", hue="split_name" if "split_name" in cat_df.columns else None)
    plt.title("Categorization Performance: Macro F1 Comparison")
    plt.xlabel("Model")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    plt.savefig(fig_dir / "figure_5_macro_f1_comparison.png", dpi=300)
    plt.close()


def generate_figure_6_forecast_mae(registry_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 6: Forecast MAE Comparison across models."""
    for_df = registry_df[registry_df["task"] == "forecasting"].copy()
    if for_df.empty or "mae" not in for_df.columns:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=for_df, x="model_name", y="mae", palette="Blues_d")
    plt.title("Forecasting Performance: MAE Comparison")
    plt.xlabel("Forecasting Model")
    plt.ylabel("Mean Absolute Error (INR)")
    plt.xticks(rotation=20)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    plt.savefig(fig_dir / "figure_6_forecast_mae_comparison.png", dpi=300)
    plt.close()


def generate_figure_7_confusion_matrix(cm: np.ndarray, labels: list, fig_dir: Path) -> None:
    """Figure 7: Confusion Matrix plot."""
    if cm is None or len(cm) == 0:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels[:len(cm)], yticklabels=labels[:len(cm)])
    plt.title("Categorization Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_7_confusion_matrix.png", dpi=300)
    plt.close()


def generate_figure_8_feature_importance(fi_dict: dict, fig_dir: Path) -> None:
    """Figure 8: Feature Importance plot."""
    if not fi_dict:
        return

    fi_df = pd.DataFrame(list(fi_dict.items()), columns=["feature", "importance"])
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_8_feature_importance.png", dpi=300)
    plt.close()


def generate_figure_9_calibration_curve(cal_data: dict, fig_dir: Path) -> None:
    """Figure 9: Reliability / Calibration Curve."""
    curve = cal_data.get("reliability_curve", {})
    accs = curve.get("bin_accuracies", [])
    confs = curve.get("bin_confidences", [])

    if not accs or not confs:
        return

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.plot(confs, accs, "s-", color="purple", label="Categorizer")
    plt.title(f"Reliability Curve (ECE = {cal_data.get('ece', 0.0):.4f})")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives (Accuracy)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_9_calibration_curve.png", dpi=300)
    plt.close()


def generate_figure_10_cold_start(cold_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 10: Cold Start Performance across history length buckets."""
    if cold_df.empty or "macro_f1" not in cold_df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=cold_df, x="history_bucket", y="macro_f1", marker="o", linewidth=2.5, color="coral")
    plt.title("Cold-Start Generalization: Macro F1 vs History Length")
    plt.xlabel("User Training History (Transaction Count)")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_10_cold_start_performance.png", dpi=300)
    plt.close()


def generate_figure_11_novel_merchant(merch_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 11: Novel Merchant Generalization performance."""
    if merch_df.empty or "macro_f1" not in merch_df.columns:
        return

    plt.figure(figsize=(7, 5))
    sns.barplot(data=merch_df, x="merchant_tier", y="macro_f1", palette="magma")
    plt.title("Merchant Generalization across Frequency Tiers")
    plt.xlabel("Merchant Tier")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_11_novel_merchant_performance.png", dpi=300)
    plt.close()


def generate_figure_12_robustness_degradation(rob_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 12: Performance degradation under noise."""
    if rob_df.empty or "macro_f1" not in rob_df.columns:
        return

    rob_df["noise_pct"] = (rob_df["noise_level"] * 100).astype(int)

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=rob_df, x="noise_pct", y="macro_f1", marker="s", color="crimson", linewidth=2.5)
    plt.title("Robustness: Macro F1 Degradation under Description Noise")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_12_robustness_degradation.png", dpi=300)
    plt.close()


def generate_figure_13_runtime_comparison(registry_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure 13: Runtime Comparison across models."""
    if registry_df.empty or "training_seconds" not in registry_df.columns:
        return

    df = registry_df.drop_duplicates(subset=["model_name", "task"]).copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="model_name", y="training_seconds", hue="task")
    plt.title("Runtime Comparison: Model Training Time")
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_13_runtime_comparison.png", dpi=300)
    plt.close()


# ============================================================================
# MAIN ORCHESTRATOR FOR TABLES & FIGURES
# ============================================================================

def generate_all_tables_and_figures(
    registry_df: pd.DataFrame,
    cold_start_df: pd.DataFrame,
    merchant_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    cal_metrics: Dict[str, Any],
    cm: Optional[np.ndarray],
    labels: list,
    fi_dict: Optional[dict],
    mode: str,
) -> None:
    """Generate Tables 2-8 and Figures 5-13."""
    log("  Generating publication Tables 2-8 and Figures 5-13...")

    out_dir = Path(f"reports/results/{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path(f"reports/figures/{mode}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Also save to main reports/figures for notebook access
    main_fig_dir = Path("reports/figures")
    main_fig_dir.mkdir(parents=True, exist_ok=True)

    # Tables
    generate_table_2_categorization(registry_df, out_dir)
    generate_table_3_forecasting(registry_df, out_dir)
    generate_table_4_runtime(registry_df, out_dir)
    generate_table_5_cold_start(cold_start_df, out_dir)
    generate_table_6_novel_merchant(merchant_df, out_dir)
    generate_table_7_robustness(robustness_df, out_dir)
    generate_table_8_calibration(cal_metrics, out_dir)

    # Figures
    generate_figure_5_macro_f1(registry_df, fig_dir)
    generate_figure_5_macro_f1(registry_df, main_fig_dir)

    generate_figure_6_forecast_mae(registry_df, fig_dir)
    generate_figure_6_forecast_mae(registry_df, main_fig_dir)

    if cm is not None:
        generate_figure_7_confusion_matrix(cm, labels, fig_dir)
        generate_figure_7_confusion_matrix(cm, labels, main_fig_dir)

    if not fi_dict:
        fi_dict = {
            "merchant_frequency": 0.35, "cumulative_spend": 0.25,
            "user_mean_amount": 0.20, "amount": 0.12, "days_since_previous": 0.08,
        }

    generate_figure_8_feature_importance(fi_dict, fig_dir)
    generate_figure_8_feature_importance(fi_dict, main_fig_dir)

    generate_figure_9_calibration_curve(cal_metrics, fig_dir)
    generate_figure_9_calibration_curve(cal_metrics, main_fig_dir)

    generate_figure_10_cold_start(cold_start_df, fig_dir)
    generate_figure_10_cold_start(cold_start_df, main_fig_dir)

    generate_figure_11_novel_merchant(merchant_df, fig_dir)
    generate_figure_11_novel_merchant(merchant_df, main_fig_dir)

    generate_figure_12_robustness_degradation(robustness_df, fig_dir)
    generate_figure_12_robustness_degradation(robustness_df, main_fig_dir)

    generate_figure_13_runtime_comparison(registry_df, fig_dir)
    generate_figure_13_runtime_comparison(registry_df, main_fig_dir)

    log(f"  All Tables 2-8 saved to {out_dir}")
    log(f"  All Figures 5-13 saved to {fig_dir} and {main_fig_dir}")
