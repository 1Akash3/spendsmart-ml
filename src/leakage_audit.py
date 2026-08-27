from dataclasses import dataclass
from typing import List

@dataclass
class FeatureMetadata:
    feature_name: str
    source: str
    calculation_window: str # e.g. "30d", "lifetime"
    available_at: str # e.g. "transaction_time", "end_of_month"
    uses_future_data: bool

class LeakageAuditor:
    def __init__(self):
        self.features: List[FeatureMetadata] = []
        
    def register_feature(self, meta: FeatureMetadata):
        self.features.append(meta)
        
    def audit(self):
        """Audits all registered features for temporal leakage."""
        violations = []
        for feature in self.features:
            if feature.uses_future_data:
                violations.append(f"Leakage detected in feature '{feature.feature_name}': explicitly uses future data.")
            if feature.available_at == "future":
                violations.append(f"Leakage detected in feature '{feature.feature_name}': available only in the future.")
        
        if violations:
            error_msg = "\n".join(violations)
            raise RuntimeError(f"Temporal Leakage Audit FAILED:\n{error_msg}")
        return True

# Singleton auditor
auditor = LeakageAuditor()

def run_leakage_audit():
    return auditor.audit()
