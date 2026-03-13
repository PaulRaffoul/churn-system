"""Data Generator Service — public API.

Usage:
    from services.data_generator import generate_churn_dataset
    from services.data_generator import validate_or_raise, validate_raw_dataset
"""

from services.data_generator.generator import generate_churn_dataset
from services.data_generator.validator import (
    validate_or_raise,
    validate_raw_dataset,
    validate_scored_dataset,
)

__all__ = [
    "generate_churn_dataset",
    "validate_or_raise",
    "validate_raw_dataset",
    "validate_scored_dataset",
]
