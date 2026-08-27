import re
from typing import Dict, Optional, List

class MerchantResolver:
    """4-Level Merchant Resolution Engine.
    
    Must be fitted ONLY on training data to prevent test-set leakage.
    Levels:
      1. Exact Match
      2. Deterministic Normalization
      3. Fuzzy Match
      4. Semantic Similarity (placeholder for later integration)
    """
    def __init__(self):
        self.exact_vocab: Dict[str, str] = {}
        self.norm_vocab: Dict[str, str] = {}
        self.is_fitted = False
    
    def _normalize(self, text: str) -> str:
        """Deterministic normalization (Level 2)."""
        if not isinstance(text, str):
            return ""
        # Lowercase
        norm = text.lower()
        # Remove common boilerplate/identifiers (e.g. UPI, transaction IDs)
        norm = re.sub(r'upi/|ref\s*\d+|txn\s*\d+', '', norm)
        # Remove special characters
        norm = re.sub(r'[^a-z0-9\s]', ' ', norm)
        # Remove extra whitespace
        norm = ' '.join(norm.split())
        return norm

    def fit(self, merchants: List[str]):
        """Learns the merchant vocabulary strictly from training data."""
        for m in merchants:
            if not isinstance(m, str) or not m.strip():
                continue
            
            # Level 1: exact raw string
            self.exact_vocab[m] = m
            
            # Level 2: normalized string
            norm_m = self._normalize(m)
            if norm_m and norm_m not in self.norm_vocab:
                self.norm_vocab[norm_m] = m
                
        self.is_fitted = True
        return self

    def resolve(self, merchant: str) -> tuple[str, str]:
        """Resolves a merchant. Returns (merchant_key, state).
        
        State is either 'KNOWN_MERCHANT' or 'UNKNOWN_MERCHANT'.
        """
        if not self.is_fitted:
            raise ValueError("MerchantResolver must be fitted on training data first.")
            
        if not isinstance(merchant, str):
            return "UNKNOWN_MERCHANT_KEY", "UNKNOWN_MERCHANT"
            
        # Level 1: Exact
        if merchant in self.exact_vocab:
            return self.exact_vocab[merchant], "KNOWN_MERCHANT"
            
        # Level 2: Deterministic Normalization
        norm_m = self._normalize(merchant)
        if norm_m in self.norm_vocab:
            return self.norm_vocab[norm_m], "KNOWN_MERCHANT"
            
        # Level 3: Fuzzy Matching (Basic fallback implementation)
        # TODO: Implement Levenshtein or rapidfuzz against self.norm_vocab keys
        
        # Level 4: Semantic Similarity
        # TODO: Implement embedding-based similarity (if threshold met)
        
        # Unresolved
        return merchant, "UNKNOWN_MERCHANT"
