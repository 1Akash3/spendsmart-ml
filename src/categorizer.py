"""Transaction categorization: free-text description -> category.

A proven, fast, CPU-only text classifier (TF-IDF word + char n-grams → logistic regression).
Char n-grams make it robust to merchant spelling noise ("PAYTM*Swiggy", "UPI/Zomato/12345").
Reports accuracy + macro-F1 on a held-out split — this is the model's *proven* accuracy.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from config import ARTIFACTS_DIR, RANDOM_SEED


def _build_pipeline() -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return Pipeline([
        ("tfidf", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(max_iter=1000, C=4.0,
                                   class_weight="balanced", random_state=RANDOM_SEED)),
    ])


class TransactionCategorizer:
    """Wraps the text pipeline with fit / predict / evaluate / save / load."""

    def __init__(self) -> None:
        self.pipe = _build_pipeline()
        self.classes_: np.ndarray | None = None

    def fit(self, descriptions, labels) -> "TransactionCategorizer":
        self.pipe.fit(descriptions, labels)
        self.classes_ = self.pipe.named_steps["clf"].classes_
        return self

    def predict(self, descriptions):
        return self.pipe.predict(descriptions)

    def predict_with_confidence(self, descriptions):
        """Return (labels, confidence) — max class probability per row."""
        proba = self.pipe.predict_proba(descriptions)
        idx = proba.argmax(axis=1)
        return self.classes_[idx], proba[np.arange(len(idx)), idx]

    def evaluate(self, descriptions, labels) -> dict:
        pred = self.predict(descriptions)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
            "n_test": int(len(labels)),
            "report": classification_report(labels, pred, zero_division=0, output_dict=True),
        }

    def save(self, path=None):
        path = path or (ARTIFACTS_DIR / "categorizer.joblib")
        joblib.dump(self.pipe, path)
        return path

    @classmethod
    def load(cls, path=None) -> "TransactionCategorizer":
        path = path or (ARTIFACTS_DIR / "categorizer.joblib")
        obj = cls()
        obj.pipe = joblib.load(path)
        obj.classes_ = obj.pipe.named_steps["clf"].classes_
        return obj
