from __future__ import annotations

from typing import Iterable

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

from .plotting import BLUE, GRID, ORANGE, book_subplots


def load_diabetes_regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    dataset = load_diabetes()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    return X, y


def fit_explainable_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_depth: int = 3,
    random_state: int = 0,
) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    tree.fit(X, y)
    return tree


def compute_tree_mdi_from_scratch(
    tree: DecisionTreeRegressor,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    feature_names = list(feature_names)
    tree_ = tree.tree_
    raw_importance = {index: 0.0 for index in range(len(feature_names))}
    root_samples = tree_.n_node_samples[0]

    for node_id in range(tree_.node_count):
        feature_index = tree_.feature[node_id]
        if feature_index < 0:
            continue

        parent_samples = tree_.n_node_samples[node_id]
        left_id = tree_.children_left[node_id]
        right_id = tree_.children_right[node_id]
        left_samples = tree_.n_node_samples[left_id]
        right_samples = tree_.n_node_samples[right_id]

        impurity_parent = tree_.impurity[node_id]
        impurity_left = tree_.impurity[left_id]
        impurity_right = tree_.impurity[right_id]

        impurity_drop = impurity_parent
        impurity_drop -= (left_samples / parent_samples) * impurity_left
        impurity_drop -= (right_samples / parent_samples) * impurity_right

        node_weight = parent_samples / root_samples
        raw_importance[feature_index] += node_weight * impurity_drop

    frame = pd.DataFrame(
        {
            "Feature": [feature_names[index] for index in range(len(feature_names))],
            "MDI": [raw_importance[index] for index in range(len(feature_names))],
        }
    )
    return frame.sort_values("MDI", ascending=False).reset_index(drop=True)


def compute_mdi_and_mda(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    n_repeats: int = 30,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    forest.fit(X_train, y_train)

    mdi_frame = pd.DataFrame(
        {
            "Feature": X.columns,
            "MDI": forest.feature_importances_,
        }
    )

    baseline_r2 = r2_score(y_test, forest.predict(X_test))
    permutation = permutation_importance(
        forest,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    mda_frame = pd.DataFrame(
        {
            "Feature": X.columns,
            "MDA": permutation.importances_mean,
        }
    ).sort_values("MDA", ascending=False)

    distribution_frame = (
        pd.DataFrame(permutation.importances.T, columns=X.columns)
        .melt(var_name="Feature", value_name="Delta_R2")
    )
    distribution_frame["Permuted_R2"] = baseline_r2 - distribution_frame["Delta_R2"]

    comparison_frame = mdi_frame.merge(mda_frame, on="Feature").sort_values(
        "MDI",
        ascending=False,
    )

    return comparison_frame, mda_frame.reset_index(drop=True), distribution_frame, baseline_r2


def plot_decision_tree_diagram(
    tree: DecisionTreeRegressor,
    feature_names: Iterable[str],
):
    fig, axis = book_subplots(size="tree")
    plot_tree(
        tree,
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        ax=axis,
    )
    axis.set_title("Decision tree on the diabetes dataset")
    axis.grid(False)
    return fig


def plot_feature_importance_bar(
    frame: pd.DataFrame,
    value_column: str,
    *,
    title: str,
    ylabel: str,
    color: str = BLUE,
):
    fig, axis = book_subplots(size="wide")
    axis.bar(frame["Feature"], frame[value_column], color=color)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", color=GRID)
    return fig


def plot_permutation_distribution(
    distribution_frame: pd.DataFrame,
    baseline_r2: float,
):
    grouped = distribution_frame.groupby("Feature")["Permuted_R2"]
    ordered_features = list(grouped.mean().sort_values().index)
    values = [distribution_frame.loc[distribution_frame["Feature"] == feature, "Permuted_R2"].to_numpy() for feature in ordered_features]

    fig, axis = book_subplots(size="wide")
    boxplot = axis.boxplot(values, patch_artist=True, tick_labels=ordered_features)
    for patch in boxplot["boxes"]:
        patch.set_facecolor(GRID)
        patch.set_edgecolor(GRID)
    for median in boxplot["medians"]:
        median.set_color(BLUE)
    axis.axhline(
        baseline_r2,
        linestyle="--",
        color=ORANGE,
        label=f"Baseline $R^2$ = {baseline_r2:.3f}",
    )
    axis.set_title("Permutation importance distribution")
    axis.set_ylabel("$R^2$ after permutation")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    return fig


def plot_mdi_vs_mda(comparison_frame: pd.DataFrame):
    fig, axis = book_subplots(size="wide")
    axis.scatter(comparison_frame["Feature"], comparison_frame["MDI"], color=BLUE, s=80, label="MDI")
    axis.scatter(
        comparison_frame["Feature"],
        comparison_frame["MDA"],
        color=ORANGE,
        marker="x",
        s=100,
        label="MDA",
    )
    axis.set_title("MDI versus MDA")
    axis.set_ylabel("Importance")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    return fig
