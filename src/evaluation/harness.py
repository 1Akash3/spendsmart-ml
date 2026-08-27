import pandas as pd
from typing import Callable, Dict, Any
from src.evaluation.splits import (
    create_temporal_split,
    create_merchant_disjoint_split,
    create_novel_merchant_split,
    create_noisy_description_split
)

def evaluate_categorizer_robustness(
    df: pd.DataFrame,
    train_func: Callable[[pd.DataFrame], Any],
    predict_func: Callable[[Any, pd.DataFrame], pd.Series],
    metric_func: Callable[[pd.Series, pd.Series], Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluation harness for transaction categorizers under rigorous splits.
    
    Args:
        df: DataFrame containing 'merchant_raw', 'merchant_key', 'timestamp', 'category'.
        train_func: Function to train model on a train split DataFrame.
        predict_func: Function to predict categories on a test split DataFrame.
        metric_func: Function to compute metrics given true and predicted labels.
    """
    results = {}
    
    # Split B: Temporal
    if "timestamp" in df.columns:
        train, test = create_temporal_split(df)
        model = train_func(train)
        preds = predict_func(model, test)
        results["temporal_split"] = metric_func(test["category"], preds)
        
    # Split C: Merchant-Disjoint
    if "merchant_key" in df.columns:
        train, test = create_merchant_disjoint_split(df)
        model = train_func(train)
        preds = predict_func(model, test)
        results["merchant_disjoint"] = metric_func(test["category"], preds)
        
    # Split D: Novel-Merchant
    if "merchant_raw" in df.columns:
        train, test = create_novel_merchant_split(df)
        model = train_func(train)
        preds = predict_func(model, test)
        results["novel_merchant"] = metric_func(test["category"], preds)
        
    # Split F: Noisy Description
    if "timestamp" in df.columns and "merchant_raw" in df.columns:
        train, test = create_noisy_description_split(df)
        model = train_func(train)
        preds = predict_func(model, test)
        results["noisy_description"] = metric_func(test["category"], preds)
        
    return results

def evaluate_forecaster_robustness(
    df: pd.DataFrame,
    train_func: Callable[[pd.DataFrame], Any],
    predict_func: Callable[[Any, pd.DataFrame], pd.Series],
    metric_func: Callable[[pd.Series, pd.Series], Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluation harness for forecasters under rigorous splits.
    Forecasters primarily rely on Temporal (Split B) and Cold-Start (which requires 
    a specialized evaluation over time-history lengths).
    """
    results = {}
    
    # Split B: Temporal Walk-Forward
    if "timestamp" in df.columns:
        train, test = create_temporal_split(df)
        model = train_func(train)
        preds = predict_func(model, test)
        results["temporal_split"] = metric_func(test["target"], preds)
        
    return results
