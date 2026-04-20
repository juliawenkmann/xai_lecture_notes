from __future__ import annotations

from . import __doc__  # noqa: F401

from ..datasets import (
    load_synthetic_classification,
    load_tabular_dataset,
    load_timeseries_dataset,
    make_lagged_frame,
)
from ..models import get_model, train_toy_logistic
from ..paths import chapter_figure_path
from ..perturbation import plot_lime_explanation, plot_shap_summary
from ..plotting import save_figure


def run_synthetic_lime_plot(
    *,
    chapter: str = "03_perturbation_based",
    output_name: str = "lime_example",
    sample_index: int = 0,
    num_features: int = 8,
    output_dir=None,
):
    X_train, y_train, X_test, _, feature_names, class_names = load_synthetic_classification()
    model = train_toy_logistic(X_train, y_train)
    figure = plot_lime_explanation(
        model,
        X_train,
        feature_names,
        class_names,
        X_test[sample_index],
        num_features=num_features,
    )
    return save_figure(figure, chapter_figure_path(chapter, output_name, output_dir=output_dir))


def run_synthetic_shap_plot(
    *,
    chapter: str = "03_perturbation_based",
    output_name: str = "shap_summary",
    explain_samples: int = 100,
    max_background: int = 200,
    max_display: int = 10,
    output_dir=None,
):
    X_train, y_train, X_test, _, feature_names, _ = load_synthetic_classification()
    model = train_toy_logistic(X_train, y_train)
    figure = plot_shap_summary(
        model,
        X_train,
        X_test[:explain_samples],
        feature_names,
        max_background=max_background,
        max_display=max_display,
    )
    return save_figure(figure, chapter_figure_path(chapter, output_name, output_dir=output_dir))


def run_lime_plot(
    model_name: str = "random_forest",
    dataset: str = "breast_cancer",
    chapter: str = "03_perturbation_based",
    output_name: str = "lime_tabular",
    sample_index: int = 0,
    num_features: int = 10,
    output_dir=None,
):
    X_train, X_test, y_train, _, feature_names, class_names = load_tabular_dataset(dataset)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    figure = plot_lime_explanation(
        model,
        X_train,
        feature_names,
        class_names,
        X_test.iloc[sample_index],
        num_features=num_features,
    )
    return save_figure(figure, chapter_figure_path(chapter, output_name, output_dir=output_dir))


def run_shap_plot(
    model_name: str = "random_forest",
    dataset: str = "breast_cancer",
    chapter: str = "03_perturbation_based",
    output_name: str = "shap_kernel_summary",
    explain_samples: int = 100,
    max_background: int = 200,
    max_display: int = 10,
    output_dir=None,
):
    X_train, X_test, y_train, _, feature_names, _ = load_tabular_dataset(dataset)
    model = get_model(model_name)
    model.fit(X_train, y_train)
    figure = plot_shap_summary(
        model,
        X_train,
        X_test.iloc[:explain_samples],
        feature_names,
        max_background=max_background,
        max_display=max_display,
    )
    return save_figure(figure, chapter_figure_path(chapter, output_name, output_dir=output_dir))


def run_shap_timeseries_plot(
    dataset: str = "sunspots",
    n_lags: int = 12,
    chapter: str = "03_perturbation_based",
    output_name: str = "shap_timeseries_summary",
    explain_samples: int = 100,
    output_dir=None,
):
    series = load_timeseries_dataset(dataset)
    X, y = make_lagged_frame(series, n_lags=n_lags)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train = y.iloc[:split_index]

    model = get_model("ridge")
    model.fit(X_train, y_train)
    figure = plot_shap_summary(
        model,
        X_train,
        X_test.iloc[:explain_samples],
        X.columns,
        max_background=min(200, len(X_train)),
        max_display=min(n_lags, 10),
    )
    return save_figure(figure, chapter_figure_path(chapter, output_name, output_dir=output_dir))
