from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_model(name: str = "random_forest", random_state: int = 42) -> Any:
    model_name = (name or "").lower()
    if model_name in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1)
    if model_name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=1000, random_state=random_state, solver="liblinear")
    if model_name == "ridge":
        return Ridge()
    raise ValueError(f"Unknown model: {name}")


def train_toy_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver="liblinear",
        ),
    )
    model.fit(X_train, y_train)
    return model
