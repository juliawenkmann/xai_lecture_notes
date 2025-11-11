from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from utils.style import apply_style, BLUE, ORANGE

def explain_sample_with_lime(model, X_train, X_test, feature_names: Sequence[str], class_names: Sequence[str],
                             sample_index: int = 0, num_features: int = 10):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise RuntimeError("LIME is not installed. Please install it with `pip install lime`.") from e

    apply_style()

    X_train_np = X_train.values if hasattr(X_train, "values") else np.array(X_train)
    X_test_np = X_test.values if hasattr(X_test, "values") else np.array(X_test)

    explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(
        X_test_np[sample_index],
        model.predict_proba,
        num_features=num_features
    )

    # Build our own consistent bar plot (blue for negative, orange for positive)
    pairs = exp.as_list()
    labels = [p[0] for p in pairs]
    values = np.array([float(p[1]) for p in pairs], dtype=float)
    order = np.argsort(np.abs(values))[::-1]
    labels = [labels[i] for i in order]
    values = values[order]

    colors = [ORANGE if v >= 0 else BLUE for v in values]
    fig, ax = plt.subplots()
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#E5E7EB", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Local contribution")
    ax.set_title("LIME — local feature contributions")
    return fig
