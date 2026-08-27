# spendsmart_ml/features/__init__.py
"""Feature extraction package.
Provides utilities to generate temporal and behavioral features from the
canonical transaction DataFrame.
"""

from .temporal import add_cyclical_time_features, add_inter_transaction_features
from .behavioral import add_behavioral_features
