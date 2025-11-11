from utils.paths import FIGURES_DIR
from utils.io import save_fig
from utils.style import apply_style
from data.datasets import load_tabular, load_timeseries, make_lagged
from models.model_zoo import get_model
from explainers.lime_explainer import explain_sample_with_lime
from explainers.shap_explainer import kernel_shap_summary_plot

def run_lime_plot(model_name: str = "random_forest", dataset: str = "breast_cancer",
                  chapter: str = "02_perturbation_based", output_prefix: str = "lime_example",
                  sample_index: int = 0):
    apply_style()
    X_train, X_test, y_train, y_test, feature_names, class_names = load_tabular(dataset)
    model = get_model(model_name)
    model.fit(X_train, y_train)

    fig = explain_sample_with_lime(model, X_train, X_test, feature_names, class_names, sample_index=sample_index)
    out_dir = FIGURES_DIR / chapter
    out_path = save_fig(fig, out_dir, output_prefix)
    return out_path

def run_shap_plot(model_name: str = "random_forest", dataset: str = "breast_cancer",
                  chapter: str = "02_perturbation_based", output_prefix: str = "shap_kernel_summary",
                  max_display: int = 10):
    apply_style()
    X_train, X_test, y_train, y_test, feature_names, class_names = load_tabular(dataset)
    model = get_model(model_name)
    model.fit(X_train, y_train)

    fig = kernel_shap_summary_plot(model, X_train, X_test, max_display=max_display)
    out_dir = FIGURES_DIR / chapter
    out_path = save_fig(fig, out_dir, output_prefix)
    return out_path

def run_shap_timeseries_plot(dataset: str = "sunspots", n_lags: int = 12,
                             chapter: str = "02_perturbation_based", output_prefix: str = "shap_ts_summary"):
    apply_style()
    ts = load_timeseries(dataset)
    X_df, y = make_lagged(ts, n_lags=n_lags)
    # Split simple train/test (last 20% as test)
    split = int(len(X_df) * 0.8)
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = get_model("ridge")  # regression
    model.fit(X_train, y_train)

    fig = kernel_shap_summary_plot(model, X_train, X_test, max_display=min(10, n_lags))
    out_dir = FIGURES_DIR / chapter
    out_path = save_fig(fig, out_dir, output_prefix)
    return out_path
