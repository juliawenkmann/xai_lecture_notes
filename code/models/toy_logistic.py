"""Simple helper to train a logistic regression model for notebook demos."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_toy_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Any:
    """Fit and return a quick logistic regression model with standard scaling."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=random_state),
    )
    model.fit(X_train, y_train)
    return model
