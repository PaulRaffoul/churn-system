"""Challenger model — Random Forest.

Same pipeline pattern as the champion: ColumnTransformer + classifier
bundled into a single scikit-learn Pipeline, so the saved artifact
handles its own preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from services.training.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

CHALLENGER_MODEL_NAME = "random_forest_v1"


def build_challenger_pipeline() -> Pipeline:
    """Create an untrained challenger model pipeline.

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
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    return pipeline


def train_challenger(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """Train the challenger model.

    Args:
        X_train: Feature DataFrame (must contain NUMERIC + CATEGORICAL columns).
        y_train: Binary target array (0/1).

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = build_challenger_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline
