from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .plotting import BLUE, CHAPTER_GRAY, ORANGE, apply_plot_style, blue_orange_cmap


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _to_frame(values, reference, feature_names: Sequence[str] | None = None):
    if hasattr(reference, "columns"):
        return pd.DataFrame(values, columns=list(reference.columns))
    if feature_names is not None:
        return pd.DataFrame(values, columns=list(feature_names))
    return values


def _subsample_rows(values, max_rows: Optional[int], random_state: int = 0):
    if max_rows is None or len(values) <= max_rows:
        return values
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(values), size=max_rows, replace=False)
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]


def plot_lime_explanation(
    model,
    X_train,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    instance,
    num_features: int = 8,
    random_state: int = 0,
    discretize_continuous: bool = True,
):
    try:
        from lime.lime_tabular import LimeTabularExplainer
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("LIME is required. Install it with `pip install lime`.") from exc

    training_array = _to_numpy(X_train)
    instance_array = _to_numpy(instance).reshape(-1)

    def predict_fn(rows):
        formatted = _to_frame(rows, X_train, feature_names)
        return model.predict_proba(formatted)

    explainer = LimeTabularExplainer(
        training_array,
        feature_names=list(feature_names),
        class_names=list(class_names),
        discretize_continuous=discretize_continuous,
        random_state=random_state,
        verbose=False,
        mode="classification",
    )

    explanation = explainer.explain_instance(
        instance_array,
        predict_fn,
        num_features=int(num_features),
    )

    pairs = explanation.as_list()
    labels = [label for label, _ in pairs]
    values = np.asarray([float(value) for _, value in pairs], dtype=float)
    order = np.argsort(np.abs(values))[::-1]
    labels = [labels[index] for index in order]
    values = values[order]

    apply_plot_style()
    colors = [ORANGE if value >= 0 else BLUE for value in values]
    fig, axis = plt.subplots()
    axis.barh(range(len(labels)), values, color=colors)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.axvline(0, color=CHAPTER_GRAY, linewidth=1.0)
    axis.invert_yaxis()
    axis.set_xlabel("Local contribution")
    axis.set_title("LIME local feature contributions")
    return fig


def plot_shap_summary(
    model,
    X_background,
    X_explain,
    feature_names: Sequence[str],
    *,
    max_background: int = 200,
    max_display: int = 10,
    random_state: int = 0,
):
    try:
        import matplotlib.pyplot as plt
        import shap
    except Exception as exc:
        raise ImportError("SHAP is required. Install it with `pip install shap`.") from exc

    background = _subsample_rows(X_background, max_background, random_state=random_state)
    explain_data = _to_frame(X_explain, background, feature_names)

    if hasattr(model, "predict_proba"):
        def predict_fn(rows):
            formatted = _to_frame(rows, background, feature_names)
            return model.predict_proba(formatted)[:, 1]
    elif hasattr(model, "decision_function"):
        def predict_fn(rows):
            formatted = _to_frame(rows, background, feature_names)
            return model.decision_function(formatted)
    else:
        def predict_fn(rows):
            formatted = _to_frame(rows, background, feature_names)
            return model.predict(formatted)

    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_data)
    except Exception:
        explainer = shap.Explainer(predict_fn, background)
        shap_values = explainer(explain_data)

    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    if isinstance(values, list):
        values = values[-1]

    apply_plot_style()
    plt.close("all")
    shap.summary_plot(
        values,
        features=explain_data,
        feature_names=list(feature_names),
        max_display=max_display,
        cmap=blue_orange_cmap(),
        show=False,
    )
    return plt.gcf()
