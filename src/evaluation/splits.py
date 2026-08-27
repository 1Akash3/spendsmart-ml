import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import hashlib

def create_temporal_split(df: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split B: Temporal Walk-Forward Split.
    Sorts by timestamp. Trains on earlier data, tests on later data.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' for temporal splitting.")
        
    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * train_fraction)
    
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    return train, test

def create_merchant_disjoint_split(df: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split C: Merchant-Disjoint Generalization.
    Assigns merchant entities exclusively to either train or test.
    No merchant entity appears in both partitions.
    """
    if "merchant_key" not in df.columns:
        raise ValueError("DataFrame must contain 'merchant_key' for disjoint split.")
        
    unique_merchants = df['merchant_key'].dropna().unique()
    np.random.seed(42) # Deterministic split
    np.random.shuffle(unique_merchants)
    
    split_idx = int(len(unique_merchants) * train_fraction)
    train_merchants = set(unique_merchants[:split_idx])
    
    train = df[df['merchant_key'].isin(train_merchants)].copy()
    test = df[~df['merchant_key'].isin(train_merchants)].copy()
    return train, test

def create_novel_merchant_split(df: pd.DataFrame, test_fraction: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split D: Novel-Merchant Robustness.
    Test set contains explicitly unseen merchant *strings* (raw descriptions)
    that did not occur in the training set.
    """
    if "merchant_raw" not in df.columns:
        raise ValueError("DataFrame must contain 'merchant_raw' for novel merchant split.")
        
    unique_raw = df['merchant_raw'].dropna().unique()
    np.random.seed(42)
    np.random.shuffle(unique_raw)
    
    test_size = int(len(unique_raw) * test_fraction)
    test_raw = set(unique_raw[:test_size])
    
    test = df[df['merchant_raw'].isin(test_raw)].copy()
    train = df[~df['merchant_raw'].isin(test_raw)].copy()
    return train, test

def create_noisy_description_split(df: pd.DataFrame, noise_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split F: Noisy Description Split.
    Injects noise (casing, punctuation, OCR errors) into a fraction of test descriptions.
    """
    train, test = create_temporal_split(df)
    
    # Simple noise injection
    def _inject_noise(text: str) -> str:
        if not isinstance(text, str): return text
        r = np.random.random()
        if r < 0.25: return text.lower()
        if r < 0.50: return text.upper()
        if r < 0.75: return text.replace(" ", "")
        return text + " X"
        
    np.random.seed(42)
    mask = np.random.random(len(test)) < noise_fraction
    test.loc[mask, 'merchant_raw'] = test.loc[mask, 'merchant_raw'].apply(_inject_noise)
    
    return train, test
