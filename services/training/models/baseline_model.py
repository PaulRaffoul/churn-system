"""Champion model — Logistic Regression.

Uses a scikit-learn Pipeline that bundles:
1. ColumnTransformer (one-hot for categoricals, pass-through for numerics)
2. LogisticRegression

This means the saved model artifact handles its own preprocessing —
no separate encoding step needed at scoring time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from services.training.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

MODEL_NAME = "logistic_regression_v1"


def build_champion_pipeline() -> Pipeline:
    """Create an untrained champion model pipeline.

    Returns:
        A scikit-learn Pipeline ready to be fit.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]
    )

    return pipeline


def train_champion(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """Train the champion model.

    Args:
        X_train: Feature DataFrame (must contain NUMERIC + CATEGORICAL columns).
        y_train: Binary target array (0/1).

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = build_champion_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Get churn probabilities from a trained pipeline.

    Args:
        pipeline: Fitted model pipeline.
        X: Feature DataFrame.

    Returns:
        1-D array of churn probabilities.
    """
    return pipeline.predict_proba(X)[:, 1]
