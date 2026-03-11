"""Data Generator Service — public API.

Usage:
    from services.data_generator import generate_churn_dataset
"""

from services.data_generator.generator import generate_churn_dataset

__all__ = ["generate_churn_dataset"]
