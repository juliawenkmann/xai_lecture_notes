from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from code.plots.style import apply_mpl_style

def _subsample(X: np.ndarray, max_samples: Optional[int], random_state: int = 0):
    if max_samples is None or X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx]

def explain_with_lime(
    model,
    X_train: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    instance: np.ndarray,
    num_features: int = 8,
    random_state: int = 0,
    discretize_continuous: bool = True,
):
    """Return a Matplotlib figure with LIME feature importances for one instance."""
    try:
        from lime import lime_tabular
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise ImportError(
            "LIME is required for explain_with_lime(). Install via: pip install lime"
        ) from e

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(feature_names),
        class_names=list(class_names),
        discretize_continuous=discretize_continuous,
        random_state=random_state,
        verbose=False,
        mode="classification",
    )
    apply_mpl_style()
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=int(num_features),
    )
    fig = exp.as_pyplot_figure()
    return fig

def explain_with_shap(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    max_background: int = 200,
    random_state: int = 0,
):
    """Return a Matplotlib figure with SHAP summary plot for a set of instances.

    Uses KernelExplainer to stay within perturbation-based methods.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError(
            "SHAP is required for explain_with_shap(). Install via: pip install shap"
        ) from e

    background = _subsample(X_background, max_background, random_state=random_state)

    def f(X):
        # probability of the positive class
        return model.predict_proba(X)[:, 1]

    try:
        explainer = shap.KernelExplainer(f, background)
    except Exception:
        # Fallback to the unified API if available
        explainer = shap.Explainer(f, background)

    shap_values = explainer.shap_values(X_explain)

    # KernelExplainer returns a list for classification in some versions
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    apply_mpl_style()
    plt.close("all")
    shap.summary_plot(sv, features=X_explain, feature_names=list(feature_names), show=False)
    fig = plt.gcf()
    return fig
