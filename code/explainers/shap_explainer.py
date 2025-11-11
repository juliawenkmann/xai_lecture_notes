import matplotlib.pyplot as plt
from utils.style import apply_style, blue_orange_cmap

def kernel_shap_summary_plot(model, X_train, X_test, max_display: int = 10, nsamples: int = 100):
    try:
        import shap
    except ImportError as e:
        raise RuntimeError("SHAP is not installed. Please install it with `pip install shap`.") from e

    apply_style()

    # Limit background size for speed
    if hasattr(X_train, "sample"):
        background = X_train.sample(min(len(X_train), nsamples), random_state=0)
    else:
        background = X_train[:nsamples]

    # Pick a prediction function
    f = getattr(model, "predict_proba", None)
    if f is None:
        if hasattr(model, "decision_function"):
            f = model.decision_function
        else:
            f = model.predict

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_test, nsamples="auto")

    # Handle binary/multiclass/regression
    if isinstance(shap_values, list):
        sv = shap_values[-1]  # pick last class for display
    else:
        sv = shap_values

    plt.figure()
    shap.summary_plot(sv, X_test, show=False, max_display=max_display, cmap=blue_orange_cmap())
    fig = plt.gcf()
    return fig
