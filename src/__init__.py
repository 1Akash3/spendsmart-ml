"""SpendSmart-ML: standalone personalized spending recommendation engine.

Path bootstrap: put the project root (for `config`) and this `src/` dir (for sibling modules
like `features`, `forecaster`) on sys.path, so the same absolute imports work whether the code is
run as `python -m src.train`, imported as `from src.train import main` (Colab), or a file is run
directly.
"""
import os as _os
import sys as _sys

_SRC = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_SRC)
for _p in (_ROOT, _SRC):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
