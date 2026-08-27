# spendsmart_ml/data/adapters/__init__.py
"""Adapter utilities for converting various source formats to the canonical transaction schema.
Each adapter function returns a pandas DataFrame that conforms to the columns defined in
`canonical_schema.Transaction`. Missing columns are added with `None` values.
"""
