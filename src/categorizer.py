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
from sentence_transformers import SentenceTransformer
import torch

from config import ARTIFACTS_DIR, RANDOM_SEED


def _build_pipeline() -> Pipeline:
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return Pipeline([
        ("tfidf", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(max_iter=1000, C=4.0,
                                   class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)),
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

# Hybrid categorizer combining TF-IDF + transformer embeddings
class HybridTransactionCategorizer:
    """Hybrid model that concatenates TF-IDF features with transformer embeddings,
    then trains a logistic regression classifier. Provides confidence-aware predictions."""
    def __init__(self, transformer_name: str = "all-MiniLM-L6-v2"):
        # TF-IDF extractor (no classifier) for feature generation
        self.tfidf_extractor = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])
        self.transformer_name = transformer_name
        self.transformer = SentenceTransformer(transformer_name)
        self.clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)
        self.classes_: np.ndarray | None = None

    def _embed(self, texts):
        # Returns numpy array of shape (n_samples, embed_dim)
        return self.transformer.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def fit(self, descriptions, labels):
        # Fit TF-IDF extractor (no labels needed)
        self.tfidf_extractor.fit(descriptions)
        tfidf_features = self.tfidf_extractor.transform(descriptions)
        # Compute transformer embeddings
        embed_features = self._embed(descriptions)
        # Concatenate TF‑IDF and transformer embeddings
        X = np.hstack([tfidf_features.toarray(), embed_features])
        self.clf.fit(X, labels)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, descriptions):
        tfidf_features = self.tfidf_extractor.transform(descriptions)
        embed_features = self._embed(descriptions)
        X = np.hstack([tfidf_features.toarray(), embed_features])
        return self.clf.predict(X)

    def predict_with_confidence(self, descriptions):
        tfidf_features = self.tfidf_extractor.transform(descriptions)
        embed_features = self._embed(descriptions)
        X = np.hstack([tfidf_features.toarray(), embed_features])
        probs = self.clf.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx], probs[np.arange(len(idx)), idx]

    def evaluate(self, descriptions, labels):
        pred = self.predict(descriptions)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
            "n_test": int(len(labels)),
            "report": classification_report(labels, pred, zero_division=0, output_dict=True),
        }

    def save(self, path=None):
        path = path or (ARTIFACTS_DIR / "hybrid_categorizer.joblib")
        joblib.dump({
            "tfidf_extractor": self.tfidf_extractor,
            "transformer_name": self.transformer_name,
            "clf": self.clf,
            "classes_": self.classes_
        }, path)
        return path

    @classmethod
    def load(cls, path=None) -> "HybridTransactionCategorizer":
        path = path or (ARTIFACTS_DIR / "hybrid_categorizer.joblib")
        data = joblib.load(path)
        obj = cls(transformer_name=data["transformer_name"])
        obj.tfidf_extractor = data["tfidf_extractor"]
        obj.clf = data["clf"]
        obj.classes_ = data["classes_"]
        return obj
