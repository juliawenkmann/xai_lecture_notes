from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_model(name: str = "random_forest", random_state: int = 42) -> Any:
    name = (name or "").lower()
    if name in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Unknown model: {name}")
