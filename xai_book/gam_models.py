from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .plotting import ANNOTATION_EDGE, BLUE, CHAPTER_GRAY, ORANGE, book_subplots


GAM_FEATURE_COLUMNS = (
    "temperature_c",
    "marketing_spend_k",
    "discount_pct",
)
GAM_TARGET_COLUMN = "demand_index"

GAM_FIGURE_FILENAMES = {
    "data_overview": "gam_data_overview",
    "model_comparison": "gam_model_comparison",
    "feature_effects": "gam_feature_effects",
    "prediction_breakdown": "gam_prediction_breakdown",
}

GAM_TABLE_FILENAMES = {
    "metrics": "gam_metrics",
    "prediction_breakdown": "gam_prediction_breakdown",
}

FEATURE_DISPLAY_NAMES = {
    "temperature_c": r"Temperature ($^\circ$C)",
    "marketing_spend_k": r"Marketing spend (\$k)",
    "discount_pct": r"Discount (\%)",
}

MODEL_LABELS = {
    "linear": "Linear baseline",
    "gam": "Additive spline model",
}


def _wrap_label(text: str, *, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def generate_gam_demo_dataset(
    *,
    n_samples: int = 260,
    random_state: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    temperature_c = rng.uniform(4.0, 34.0, size=n_samples)
    marketing_spend_k = rng.uniform(0.5, 12.0, size=n_samples)
    discount_pct = rng.uniform(0.0, 30.0, size=n_samples)
    noise = rng.normal(loc=0.0, scale=1.7, size=n_samples)

    demand_index = (
        58.0
        + 9.5 * np.sin((temperature_c - 8.0) / 6.5)
        + 7.0 * np.log1p(marketing_spend_k)
        + 0.58 * discount_pct
        - 0.018 * discount_pct**2
        + noise
    )

    frame = pd.DataFrame(
        {
            "temperature_c": temperature_c,
            "marketing_spend_k": marketing_spend_k,
            "discount_pct": discount_pct,
            "demand_index": demand_index,
        }
    )
    return frame.round(4)


def ensure_gam_demo_dataset(
    data_path: Path | str,
    *,
    n_samples: int = 260,
    random_state: int = 7,
) -> pd.DataFrame:
    path = Path(data_path)
    if path.exists():
        return pd.read_csv(path)

    frame = generate_gam_demo_dataset(n_samples=n_samples, random_state=random_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _make_linear_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def _make_additive_spline_model(
    feature_columns: tuple[str, ...] = GAM_FEATURE_COLUMNS,
    *,
    n_knots: int = 6,
    alpha: float = 0.4,
) -> Pipeline:
    transformers = [
        (
            feature,
            SplineTransformer(
                n_knots=n_knots,
                degree=3,
                include_bias=False,
            ),
            [feature],
        )
        for feature in feature_columns
    ]
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _metrics_row(y_true: pd.Series, y_pred: np.ndarray, *, label: str) -> dict[str, float | str]:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "model": label,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def _effect_curve_frame(
    model: Any,
    x_reference: pd.DataFrame,
    x_context: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    base_row = x_reference.median().to_frame().T
    rows: list[pd.DataFrame] = []

    for feature in GAM_FEATURE_COLUMNS:
        grid = np.linspace(x_context[feature].min(), x_context[feature].max(), 140)
        grid_frame = pd.concat([base_row] * len(grid), ignore_index=True)
        grid_frame[feature] = grid
        predictions = model.predict(grid_frame)
        rows.append(
            pd.DataFrame(
                {
                    "model": label,
                    "feature": feature,
                    "feature_value": grid,
                    "prediction": predictions,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def gam_prediction_breakdown(model: Pipeline, row: pd.Series) -> pd.Series:
    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["ridge"]
    row_frame = row[list(GAM_FEATURE_COLUMNS)].to_frame().T
    transformed = preprocessor.transform(row_frame)
    if not isinstance(transformed, np.ndarray):
        transformed = transformed.toarray()

    contributions: dict[str, float] = {"Intercept": float(ridge.intercept_)}
    for feature in GAM_FEATURE_COLUMNS:
        feature_slice = preprocessor.output_indices_[feature]
        contributions[FEATURE_DISPLAY_NAMES[feature]] = float(
            transformed[0, feature_slice] @ ridge.coef_[feature_slice]
        )

    series = pd.Series(contributions, name="contribution")
    series["Prediction"] = float(series.sum())
    return series


def run_gam_regression_analysis(
    *,
    data_path: Path | str,
    test_size: float = 0.25,
    random_state: int = 7,
) -> dict[str, Any]:
    data = ensure_gam_demo_dataset(data_path, random_state=random_state)
    features = data.loc[:, GAM_FEATURE_COLUMNS].copy()
    target = data.loc[:, GAM_TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    linear_model = _make_linear_baseline()
    gam_model = _make_additive_spline_model()
    linear_model.fit(x_train, y_train)
    gam_model.fit(x_train, y_train)

    linear_pred = linear_model.predict(x_test)
    gam_pred = gam_model.predict(x_test)

    metrics = pd.DataFrame(
        [
            _metrics_row(y_test, linear_pred, label=MODEL_LABELS["linear"]),
            _metrics_row(y_test, gam_pred, label=MODEL_LABELS["gam"]),
        ]
    )

    test_results = x_test.copy()
    test_results["actual"] = y_test
    test_results["linear_pred"] = linear_pred
    test_results["gam_pred"] = gam_pred
    test_results = test_results.sort_values("actual").reset_index(drop=True)

    effect_curves = pd.concat(
        [
            _effect_curve_frame(linear_model, x_train, features, label=MODEL_LABELS["linear"]),
            _effect_curve_frame(gam_model, x_train, features, label=MODEL_LABELS["gam"]),
        ],
        ignore_index=True,
    )

    example_index = int(np.argmin(np.abs(gam_pred - y_test.to_numpy())))
    example_row = x_test.iloc[example_index]
    example_actual = float(y_test.iloc[example_index])
    example_prediction = float(gam_pred[example_index])
    breakdown = gam_prediction_breakdown(gam_model, example_row)
    breakdown_frame = breakdown.rename_axis("term").reset_index(name="contribution")
    breakdown_frame["actual"] = example_actual
    breakdown_frame["predicted"] = example_prediction

    figures = {
        "data_overview": plot_gam_data_overview(data=data),
        "model_comparison": plot_gam_model_comparison(
            y_test=y_test.reset_index(drop=True),
            linear_pred=np.asarray(linear_pred),
            gam_pred=np.asarray(gam_pred),
            metrics=metrics,
        ),
        "feature_effects": plot_gam_feature_effects(
            data=data,
            effect_curves=effect_curves,
        ),
        "prediction_breakdown": plot_gam_prediction_breakdown(
            breakdown=breakdown,
            actual=example_actual,
            predicted=example_prediction,
        ),
    }

    return {
        "data": data,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "test_results": test_results,
        "effect_curves": effect_curves,
        "breakdown": breakdown,
        "breakdown_frame": breakdown_frame,
        "figures": figures,
    }


def plot_gam_data_overview(*, data: pd.DataFrame):
    fig, axes = book_subplots(1, 3, size="three_panel")

    for axis, feature in zip(axes, GAM_FEATURE_COLUMNS):
        feature_data = data[[feature, GAM_TARGET_COLUMN]].sort_values(feature)
        bins = np.linspace(feature_data[feature].min(), feature_data[feature].max(), 12)
        feature_bins = pd.cut(feature_data[feature], bins=bins, include_lowest=True)
        grouped = (
            feature_data.assign(feature_bin=feature_bins)
            .groupby("feature_bin", observed=False)
            .agg(
                feature_value=(feature, "mean"),
                demand_mean=(GAM_TARGET_COLUMN, "mean"),
            )
            .dropna()
        )

        axis.scatter(
            feature_data[feature],
            feature_data[GAM_TARGET_COLUMN],
            color=CHAPTER_GRAY,
            alpha=0.18,
            s=16,
        )
        axis.plot(
            grouped["feature_value"],
            grouped["demand_mean"],
            color=BLUE,
            linewidth=2.4,
            marker="o",
            markersize=4,
        )
        axis.set_title(FEATURE_DISPLAY_NAMES[feature])
        axis.set_xlabel(FEATURE_DISPLAY_NAMES[feature])
        axis.grid(True, alpha=0.22)

    axes[0].set_ylabel("Observed demand index")
    fig.suptitle("Synthetic demand data used in the GAM notebook")
    return fig


def plot_gam_model_comparison(
    *,
    y_test: pd.Series,
    linear_pred: np.ndarray,
    gam_pred: np.ndarray,
    metrics: pd.DataFrame,
):
    fig, axes = book_subplots(1, 2, size="two_panel")
    actual = y_test.to_numpy()
    min_value = float(min(actual.min(), linear_pred.min(), gam_pred.min()))
    max_value = float(max(actual.max(), linear_pred.max(), gam_pred.max()))
    diagonal = np.linspace(min_value, max_value, 100)

    for axis, label, pred, color in (
        (axes[0], MODEL_LABELS["linear"], linear_pred, ORANGE),
        (axes[1], MODEL_LABELS["gam"], gam_pred, BLUE),
    ):
        axis.scatter(actual, pred, color=color, alpha=0.78, s=28)
        axis.plot(diagonal, diagonal, color=CHAPTER_GRAY, linewidth=1.2, linestyle="--")
        axis.set_title(label)
        axis.set_xlabel("True demand index")
        axis.set_ylabel("Predicted demand index")
        axis.grid(True, alpha=0.25)

        row = metrics.loc[metrics["model"] == label].iloc[0]
        axis.text(
            0.03,
            0.97,
            f"RMSE {row['RMSE']:.2f}\n$R^2$ {row['R2']:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": ANNOTATION_EDGE, "alpha": 0.95},
        )

    fig.suptitle("Linear baseline versus additive spline model")
    return fig


def plot_gam_feature_effects(
    *,
    data: pd.DataFrame,
    effect_curves: pd.DataFrame,
):
    fig, axes = book_subplots(1, 3, size="three_panel")

    for axis, feature in zip(axes, GAM_FEATURE_COLUMNS):
        feature_data = data[[feature, GAM_TARGET_COLUMN]].sort_values(feature)
        curve_data = effect_curves.loc[effect_curves["feature"] == feature]

        axis.scatter(
            feature_data[feature],
            feature_data[GAM_TARGET_COLUMN],
            color=CHAPTER_GRAY,
            alpha=0.18,
            s=16,
        )

        for label, color, linestyle in (
            (MODEL_LABELS["linear"], ORANGE, "--"),
            (MODEL_LABELS["gam"], BLUE, "-"),
        ):
            frame = curve_data.loc[curve_data["model"] == label]
            axis.plot(frame["feature_value"], frame["prediction"], color=color, linewidth=2.2, linestyle=linestyle, label=label)

        axis.set_title(FEATURE_DISPLAY_NAMES[feature])
        axis.set_xlabel(FEATURE_DISPLAY_NAMES[feature])
        axis.grid(True, alpha=0.22)

    axes[0].set_ylabel("Predicted demand index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    return fig


def plot_gam_prediction_breakdown(
    *,
    breakdown: pd.Series,
    actual: float,
    predicted: float,
):
    values = breakdown.drop("Prediction").sort_values()
    labels = [_wrap_label(label, width=18) for label in values.index]
    colors = [ORANGE if value < 0 else BLUE for value in values.values]

    fig, axis = book_subplots(size="breakdown")
    bars = axis.barh(labels, values.values, color=colors, alpha=0.94)
    axis.axvline(0.0, color=CHAPTER_GRAY, linewidth=1.0)
    axis.set_xlabel("Contribution to predicted demand")
    axis.set_title("Additive contribution breakdown for one example")
    axis.grid(True, axis="x", alpha=0.22)

    x_pad = max(0.5, 0.15 * max(abs(values.min()), abs(values.max())))
    axis.set_xlim(values.min() - x_pad, values.max() + x_pad)
    for bar, value in zip(bars, values.values):
        axis.text(
            value + (0.15 if value >= 0 else -0.15),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )

    axis.text(
        0.98,
        0.05,
        f"Predicted {predicted:.1f}\nActual {actual:.1f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": ANNOTATION_EDGE, "alpha": 0.95},
    )
    return fig
