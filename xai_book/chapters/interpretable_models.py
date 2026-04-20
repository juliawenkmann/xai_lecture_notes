from __future__ import annotations

from ..interpretable_models import (
    compute_mdi_and_mda,
    compute_tree_mdi_from_scratch,
    fit_explainable_tree,
    load_diabetes_regression_data,
    plot_decision_tree_diagram,
    plot_feature_importance_bar,
    plot_mdi_vs_mda,
    plot_permutation_distribution,
)
from ..paths import chapter_figure_path
from ..plotting import ORANGE, save_figure


MDI_AND_MDA_FILENAMES = {
    "decision_tree": "decision_tree_diabetes",
    "mdi_from_scratch": "mdi_from_scratch",
    "mda_bar": "mda_bar",
    "mda_visualization": "mda_visualization",
    "mdi_vs_mda": "mdi_vs_mda",
}


def build_mdi_and_mda_figures():
    X, y = load_diabetes_regression_data()
    tree = fit_explainable_tree(X, y)
    manual_mdi = compute_tree_mdi_from_scratch(tree, X.columns)
    comparison_frame, mda_frame, distribution_frame, baseline_r2 = compute_mdi_and_mda(X, y)

    return {
        "decision_tree": plot_decision_tree_diagram(tree, X.columns),
        "mdi_from_scratch": plot_feature_importance_bar(
            manual_mdi,
            "MDI",
            title="Mean decrease in impurity",
            ylabel="MDI (from scratch)",
        ),
        "mda_bar": plot_feature_importance_bar(
            mda_frame,
            "MDA",
            title="Mean decrease in accuracy",
            ylabel="Average drop in $R^2$ when a feature is permuted",
            color=ORANGE,
        ),
        "mda_visualization": plot_permutation_distribution(distribution_frame, baseline_r2),
        "mdi_vs_mda": plot_mdi_vs_mda(comparison_frame),
    }


def run_mdi_and_mda_figures(
    *,
    chapter: str = "02_interpretable_models",
    output_dir=None,
):
    figures = build_mdi_and_mda_figures()
    outputs = {}
    for name, fig in figures.items():
        outputs[name] = save_figure(
            fig,
            chapter_figure_path(
                chapter,
                MDI_AND_MDA_FILENAMES[name],
                suffix=".pdf",
                output_dir=output_dir,
            ),
        )
    return outputs
