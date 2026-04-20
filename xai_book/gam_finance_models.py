from __future__ import annotations

from pathlib import Path
import re
import textwrap
from typing import Any, Sequence

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from .plotting import ANNOTATION_EDGE, BLUE, CHAPTER_GRAY, ORANGE, SOFT_BLUE, book_subplots
from .var_models import (
    compute_log_returns,
    flat_price_baseline,
    load_finance_var_prices,
    metrics_table,
    rebuild_price_path,
    rolling_one_step_var_contributions,
    run_var_finance_analysis,
)


DEFAULT_FINANCE_GAM_TARGET = "AMZN"
DEFAULT_FINANCE_GAM_PEER = "NFLX"
DEFAULT_FINANCE_GAM_FEATURE_COLUMNS = (
    "target_lag1_return",
    "target_lag2_return",
    "peer_lag1_return",
    "peer_lag2_return",
)

FINANCE_GAM_FIGURE_FILENAMES = {
    "model_comparison": "gam_finance_model_comparison",
    "forecast": "gam_finance_forecast",
    "forecast_panels": "gam_finance_forecast_panels",
    "forecast_zoom": "gam_finance_forecast_zoom",
    "feature_effects": "gam_finance_feature_effects",
    "prediction_breakdown": "gam_finance_prediction_breakdown",
    "var_prediction_breakdown": "gam_finance_var_prediction_breakdown",
}

FINANCE_GAM_TABLE_FILENAMES = {
    "metrics": "gam_finance_metrics",
    "prediction_breakdown": "gam_finance_prediction_breakdown",
    "var_prediction_breakdown": "gam_finance_var_prediction_breakdown",
    "validation_search": "gam_finance_validation_search",
}

FINANCE_GAM_MODEL_LABELS = {
    "flat": "Flat baseline",
    "gam": "Spline GAM",
    "var": "VAR reference",
}

FINANCE_GAM_LEGEND_LABELS = {
    "gam": "Spline GAM",
    "var": "VAR",
}


def _wrap_label(text: str, *, width: int = 19) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def finance_display_series_labels(*, target: str, peer: str) -> dict[str, str]:
    return {
        target: "stock",
        peer: "mkt",
    }


def finance_return_symbol(label: str, *, lag: int | None = None) -> str:
    if lag is None:
        return rf"$r_t^{{\mathrm{{{label}}}}}$"
    return rf"$r_{{t-{lag}}}^{{\mathrm{{{label}}}}}$"


def finance_gam_feature_labels(
    *,
    target: str,
    peer: str,
    series_labels: dict[str, str] | None = None,
) -> dict[str, str]:
    series_labels = series_labels or finance_display_series_labels(target=target, peer=peer)
    return {
        "target_lag1_return": finance_return_symbol(series_labels[target], lag=1),
        "target_lag2_return": finance_return_symbol(series_labels[target], lag=2),
        "peer_lag1_return": finance_return_symbol(series_labels[peer], lag=1),
        "peer_lag2_return": finance_return_symbol(series_labels[peer], lag=2),
    }


def build_finance_gam_dataset(
    *,
    data_path: Path | str | None = None,
    target: str = DEFAULT_FINANCE_GAM_TARGET,
    peer: str = DEFAULT_FINANCE_GAM_PEER,
    feature_columns: Sequence[str] = DEFAULT_FINANCE_GAM_FEATURE_COLUMNS,
) -> dict[str, Any]:
    feature_columns = tuple(feature_columns)
    prices = load_finance_var_prices(data_path, tickers=(target, peer))
    log_returns = compute_log_returns(prices)
    series_labels = finance_display_series_labels(target=target, peer=peer)
    dataset = pd.DataFrame(
        {
            "target_lag1_return": log_returns[target].shift(1),
            "target_lag2_return": log_returns[target].shift(2),
            "peer_lag1_return": log_returns[peer].shift(1),
            "peer_lag2_return": log_returns[peer].shift(2),
            "target_return": log_returns[target],
        }
    ).dropna()
    dataset["target_price"] = prices.loc[dataset.index, target]

    return {
        "prices": prices,
        "log_returns": log_returns,
        "dataset": dataset,
        "feature_columns": feature_columns,
        "feature_labels": finance_gam_feature_labels(target=target, peer=peer, series_labels=series_labels),
        "series_labels": series_labels,
        "target": target,
        "peer": peer,
    }


def _split_time_blocks(
    dataset: pd.DataFrame,
    *,
    n_test: int,
    validation_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_full = dataset.iloc[:-n_test].copy()
    test = dataset.iloc[-n_test:].copy()
    validation_size = min(validation_size, max(4, len(train_full) // 4))
    train_core = train_full.iloc[:-validation_size].copy()
    validation = train_full.iloc[-validation_size:].copy()
    return train_core, validation, train_full, test


def _make_additive_spline_model(
    feature_columns: Sequence[str],
    *,
    n_knots: int,
    alpha: float,
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
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers,
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _target_price_frame(target: str, prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({target: prices}, index=prices.index)


def _evaluate_price_predictions(
    *,
    target: str,
    base_price: float,
    predicted_returns: pd.Series,
    actual_prices: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    pred_returns_frame = pd.DataFrame({target: predicted_returns}, index=predicted_returns.index)
    pred_prices = rebuild_price_path(pd.Series({target: base_price}), pred_returns_frame)[target]
    price_metrics = metrics_table(
        _target_price_frame(target, actual_prices),
        _target_price_frame(target, pred_prices),
    ).loc[target]
    return pred_prices, price_metrics


def _gam_validation_search_frame(
    *,
    target: str,
    train_core: pd.DataFrame,
    validation: pd.DataFrame,
    prices: pd.DataFrame,
    feature_columns: Sequence[str],
    gam_n_knots_grid: Sequence[int],
    gam_alpha_grid: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    x_core = train_core.loc[:, list(feature_columns)]
    y_core = train_core["target_return"]
    x_validation = validation.loc[:, list(feature_columns)]
    y_validation = validation["target_return"]
    validation_prices = prices.loc[validation.index, target]
    base_price = float(prices.loc[train_core.index[-1], target])
    rows: list[dict[str, float | str]] = []

    for n_knots in gam_n_knots_grid:
        for alpha in gam_alpha_grid:
            model = _make_additive_spline_model(feature_columns, n_knots=n_knots, alpha=alpha)
            model.fit(x_core, y_core)
            pred_returns = pd.Series(model.predict(x_validation), index=x_validation.index)
            pred_prices, price_metrics = _evaluate_price_predictions(
                target=target,
                base_price=base_price,
                predicted_returns=pred_returns,
                actual_prices=validation_prices,
            )
            rows.append(
                {
                    "model": FINANCE_GAM_MODEL_LABELS["gam"],
                    "alpha": float(alpha),
                    "n_knots": int(n_knots),
                    "validation_price_MAE": float(price_metrics["MAE"]),
                    "validation_price_RMSE": float(price_metrics["RMSE"]),
                    "validation_price_MAPE_pct": float(price_metrics["MAPE_pct"]),
                    "validation_return_RMSE_bp": float(mean_squared_error(y_validation, pred_returns) ** 0.5 * 1e4),
                    "n_validation_weeks": len(pred_prices),
                }
            )

    search = pd.DataFrame(rows).sort_values(
        ["validation_price_MAPE_pct", "validation_return_RMSE_bp", "alpha", "n_knots"],
    ).reset_index(drop=True)
    gam_best = search.iloc[0].to_dict()
    return search, gam_best


def finance_effect_curve_frame(
    model: Pipeline,
    *,
    train_features: pd.DataFrame,
    feature_columns: Sequence[str],
    model_label: str | None = None,
) -> pd.DataFrame:
    anchor_row = train_features.median().to_frame().T
    rows: list[pd.DataFrame] = []
    for feature in feature_columns:
        grid = np.linspace(train_features[feature].min(), train_features[feature].max(), 160)
        grid_frame = pd.concat([anchor_row] * len(grid), ignore_index=True)
        grid_frame[feature] = grid
        predictions = model.predict(grid_frame)
        rows.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "feature_value": grid,
                    "prediction": predictions,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    if model_label is not None:
        frame["model"] = model_label
    return frame


def finance_linear_prediction_breakdown(
    model: Pipeline,
    *,
    row: pd.Series,
    feature_columns: Sequence[str],
    feature_labels: dict[str, str],
) -> pd.Series:
    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["ridge"]
    row_frame = row.loc[list(feature_columns)].to_frame().T
    transformed = preprocessor.transform(row_frame)
    if not isinstance(transformed, np.ndarray):
        transformed = transformed.toarray()

    contributions: dict[str, float] = {"Intercept": float(ridge.intercept_)}
    for feature in feature_columns:
        feature_slice = preprocessor.output_indices_[feature]
        contributions[feature_labels[feature]] = float(
            transformed[0, feature_slice] @ ridge.coef_[feature_slice]
        )

    series = pd.Series(contributions, name="contribution")
    series["Prediction"] = float(series.sum())
    return series


def finance_gam_prediction_breakdown(
    model: Pipeline,
    *,
    row: pd.Series,
    feature_columns: Sequence[str],
    feature_labels: dict[str, str],
) -> pd.Series:
    preprocessor = model.named_steps["preprocessor"]
    ridge = model.named_steps["ridge"]
    row_frame = row.loc[list(feature_columns)].to_frame().T
    transformed = preprocessor.transform(row_frame)
    if not isinstance(transformed, np.ndarray):
        transformed = transformed.toarray()

    contributions: dict[str, float] = {"Intercept": float(ridge.intercept_)}
    for feature in feature_columns:
        feature_slice = preprocessor.output_indices_[feature]
        contributions[feature_labels[feature]] = float(transformed[0, feature_slice] @ ridge.coef_[feature_slice])

    series = pd.Series(contributions, name="contribution")
    series["Prediction"] = float(series.sum())
    return series


def finance_var_prediction_breakdown(
    *,
    train_returns: pd.DataFrame,
    future_returns: pd.DataFrame,
    lag: int,
    target: str,
    step_index: int,
    series_labels: dict[str, str] | None = None,
) -> pd.Series:
    contributions = rolling_one_step_var_contributions(
        train_returns,
        future_returns,
        lag=lag,
        target=target,
        step_index=step_index,
    )

    renamed: dict[str, float] = {}
    for term, value in contributions.items():
        if term == "intercept":
            label = "Intercept"
        elif term == "forecast_total":
            label = "Prediction"
        else:
            match = re.fullmatch(r"L(\d+)\.(.+)", term)
            if match is None:
                label = str(term)
            else:
                lag_number, source = match.groups()
                display_source = source if series_labels is None else series_labels.get(source, source)
                label = finance_return_symbol(display_source, lag=int(lag_number))
        renamed[label] = float(value)

    return pd.Series(renamed, name="contribution")


def _breakdown_frame(
    breakdown: pd.Series,
    *,
    forecast_date: pd.Timestamp,
    actual_return: float,
    predicted_return: float,
) -> pd.DataFrame:
    frame = breakdown.rename_axis("term").reset_index(name="contribution")
    frame["forecast_date"] = forecast_date
    frame["actual_return"] = actual_return
    frame["predicted_return"] = predicted_return
    return frame


def _metric_row(
    *,
    model: str,
    target: str,
    actual_prices: pd.Series,
    pred_prices: pd.Series,
    actual_returns: pd.Series,
    pred_returns: pd.Series,
) -> dict[str, float | str]:
    price_metrics = metrics_table(
        _target_price_frame(target, actual_prices),
        _target_price_frame(target, pred_prices),
    ).loc[target]
    return {
        "model": model,
        "price_MAE": float(price_metrics["MAE"]),
        "price_RMSE": float(price_metrics["RMSE"]),
        "price_MAPE_pct": float(price_metrics["MAPE_pct"]),
        "return_RMSE_bp": float(mean_squared_error(actual_returns, pred_returns) ** 0.5 * 1e4),
        "return_MAE_bp": float(mean_absolute_error(actual_returns, pred_returns) * 1e4),
    }


def run_finance_gam_analysis(
    *,
    data_path: Path | str | None = None,
    target: str = DEFAULT_FINANCE_GAM_TARGET,
    peer: str = DEFAULT_FINANCE_GAM_PEER,
    n_test: int = 12,
    validation_size: int = 8,
    gam_n_knots_grid: Sequence[int] = (3, 4, 5),
    gam_alpha_grid: Sequence[float] = (0.1, 0.3, 1.0, 4.0, 12.0),
    var_maxlags: int = 6,
    feature_columns: Sequence[str] = DEFAULT_FINANCE_GAM_FEATURE_COLUMNS,
) -> dict[str, Any]:
    prepared = build_finance_gam_dataset(
        data_path=data_path,
        target=target,
        peer=peer,
        feature_columns=feature_columns,
    )
    prices = prepared["prices"]
    log_returns = prepared["log_returns"]
    dataset = prepared["dataset"]
    feature_columns = tuple(prepared["feature_columns"])
    feature_labels = dict(prepared["feature_labels"])
    series_labels = dict(prepared["series_labels"])

    train_core, validation, train_full, test = _split_time_blocks(
        dataset,
        n_test=n_test,
        validation_size=validation_size,
    )
    search, gam_best = _gam_validation_search_frame(
        target=target,
        train_core=train_core,
        validation=validation,
        prices=prices,
        feature_columns=feature_columns,
        gam_n_knots_grid=gam_n_knots_grid,
        gam_alpha_grid=gam_alpha_grid,
    )

    x_train = train_full.loc[:, list(feature_columns)]
    y_train = train_full["target_return"]
    x_test = test.loc[:, list(feature_columns)]
    y_test = test["target_return"]
    actual_prices = prices.loc[test.index, target]
    history_prices = prices.loc[: train_full.index[-1], target]
    last_train_price = float(prices.loc[train_full.index[-1], target])

    gam_model = _make_additive_spline_model(
        feature_columns,
        n_knots=int(gam_best["n_knots"]),
        alpha=float(gam_best["alpha"]),
    )
    gam_model.fit(x_train, y_train)

    gam_pred_returns = pd.Series(gam_model.predict(x_test), index=x_test.index, name="predicted_return")

    gam_pred_prices, _ = _evaluate_price_predictions(
        target=target,
        base_price=last_train_price,
        predicted_returns=gam_pred_returns,
        actual_prices=actual_prices,
    )

    zero_returns = pd.Series(0.0, index=y_test.index, name="predicted_return")
    flat_prices = flat_price_baseline(pd.Series({target: last_train_price}), y_test.index)[target]

    var_analysis = run_var_finance_analysis(
        data_path=data_path,
        tickers=(target, peer),
        n_test=n_test,
        validation_size=validation_size,
        maxlags=var_maxlags,
        target_plot=target,
        target_local=target,
        impulse=peer,
        response=target,
        series_labels=series_labels,
    )
    var_pred_prices = var_analysis["pred_prices"][target].reindex(y_test.index)
    var_pred_returns = var_analysis["forecast_returns"][target].reindex(y_test.index)

    metrics = pd.DataFrame(
        [
            _metric_row(
                model=FINANCE_GAM_MODEL_LABELS["flat"],
                target=target,
                actual_prices=actual_prices,
                pred_prices=flat_prices,
                actual_returns=y_test,
                pred_returns=zero_returns,
            ),
            _metric_row(
                model=FINANCE_GAM_MODEL_LABELS["gam"],
                target=target,
                actual_prices=actual_prices,
                pred_prices=gam_pred_prices,
                actual_returns=y_test,
                pred_returns=gam_pred_returns,
            ),
            _metric_row(
                model=FINANCE_GAM_MODEL_LABELS["var"],
                target=target,
                actual_prices=actual_prices,
                pred_prices=var_pred_prices,
                actual_returns=y_test,
                pred_returns=var_pred_returns,
            ),
        ]
    )

    effect_curves = finance_effect_curve_frame(
        gam_model,
        train_features=x_train,
        feature_columns=feature_columns,
        model_label=FINANCE_GAM_MODEL_LABELS["gam"],
    )

    example_index = int(np.argmin(np.abs(gam_pred_returns.to_numpy() - y_test.to_numpy())))
    example_row = x_test.iloc[example_index]
    example_date = x_test.index[example_index]
    gam_breakdown = finance_gam_prediction_breakdown(
        gam_model,
        row=example_row,
        feature_columns=feature_columns,
        feature_labels=feature_labels,
    )
    gam_breakdown_frame = _breakdown_frame(
        gam_breakdown,
        forecast_date=example_date,
        actual_return=float(y_test.iloc[example_index]),
        predicted_return=float(gam_pred_returns.iloc[example_index]),
    )
    var_example_index = int(var_analysis["test"].index.get_loc(example_date))
    var_breakdown = finance_var_prediction_breakdown(
        train_returns=var_analysis["train"],
        future_returns=var_analysis["test"],
        lag=int(var_analysis["selected_lag"]),
        target=target,
        step_index=var_example_index,
        series_labels=series_labels,
    )
    var_breakdown_frame = _breakdown_frame(
        var_breakdown,
        forecast_date=example_date,
        actual_return=float(y_test.iloc[example_index]),
        predicted_return=float(var_pred_returns.loc[example_date]),
    )

    figures = {
        "model_comparison": plot_finance_gam_model_comparison(metrics),
        "forecast": plot_finance_gam_forecast(
            history_prices=history_prices,
            actual_prices=actual_prices,
            prediction_paths={
                FINANCE_GAM_MODEL_LABELS["gam"]: gam_pred_prices,
                FINANCE_GAM_MODEL_LABELS["var"]: var_pred_prices,
            },
            target_label=series_labels[target],
        ),
        "forecast_panels": plot_finance_gam_forecast_panels(
            history_prices=history_prices,
            actual_prices=actual_prices,
            prediction_paths={
                FINANCE_GAM_MODEL_LABELS["gam"]: gam_pred_prices,
                FINANCE_GAM_MODEL_LABELS["var"]: var_pred_prices,
            },
            target_label=series_labels[target],
        ),
        "forecast_zoom": plot_finance_gam_forecast(
            history_prices=history_prices,
            actual_prices=actual_prices,
            prediction_paths={
                FINANCE_GAM_MODEL_LABELS["gam"]: gam_pred_prices,
                FINANCE_GAM_MODEL_LABELS["var"]: var_pred_prices,
            },
            target_label=series_labels[target],
            history_tail_weeks=10,
            zoom_prediction_window=True,
        ),
        "feature_effects": plot_finance_gam_feature_effects(
            dataset=dataset,
            effect_curves=effect_curves,
            feature_columns=feature_columns,
            feature_labels=feature_labels,
            target_label=series_labels[target],
        ),
        "prediction_breakdown": plot_finance_gam_prediction_breakdown(
            breakdown=gam_breakdown,
            actual_return=float(y_test.iloc[example_index]),
            predicted_return=float(gam_pred_returns.iloc[example_index]),
            forecast_date=pd.Timestamp(example_date),
            model_label=FINANCE_GAM_LEGEND_LABELS["gam"],
        ),
        "var_prediction_breakdown": plot_finance_gam_prediction_breakdown(
            breakdown=var_breakdown,
            actual_return=float(y_test.iloc[example_index]),
            predicted_return=float(var_pred_returns.loc[example_date]),
            forecast_date=pd.Timestamp(example_date),
            model_label=FINANCE_GAM_LEGEND_LABELS["var"],
        ),
    }

    return {
        "prices": prices,
        "log_returns": log_returns,
        "dataset": dataset,
        "train_core": train_core,
        "validation": validation,
        "train_full": train_full,
        "test": test,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "search": search,
        "gam_best": gam_best,
        "gam_model": gam_model,
        "metrics": metrics,
        "history_prices": history_prices,
        "actual_prices": actual_prices,
        "flat_prices": flat_prices,
        "gam_pred_prices": gam_pred_prices,
        "gam_pred_returns": gam_pred_returns,
        "var_analysis": var_analysis,
        "feature_columns": feature_columns,
        "feature_labels": feature_labels,
        "series_labels": series_labels,
        "effect_curves": effect_curves,
        "breakdown": gam_breakdown,
        "breakdown_frame": gam_breakdown_frame,
        "gam_breakdown": gam_breakdown,
        "gam_breakdown_frame": gam_breakdown_frame,
        "var_breakdown": var_breakdown,
        "var_breakdown_frame": var_breakdown_frame,
        "example_index": example_index,
        "example_date": example_date,
        "target": target,
        "peer": peer,
        "figures": figures,
    }


def plot_finance_gam_model_comparison(metrics: pd.DataFrame):
    known_tail = [FINANCE_GAM_MODEL_LABELS["gam"], FINANCE_GAM_MODEL_LABELS["var"]]
    order = []
    if FINANCE_GAM_MODEL_LABELS["flat"] in set(metrics["model"]):
        order.append(FINANCE_GAM_MODEL_LABELS["flat"])
    order.extend(
        label
        for label in metrics["model"]
        if label not in order and label not in known_tail
    )
    order.extend(label for label in known_tail if label in set(metrics["model"]))
    color_map = {
        FINANCE_GAM_MODEL_LABELS["flat"]: CHAPTER_GRAY,
        FINANCE_GAM_MODEL_LABELS["gam"]: BLUE,
        FINANCE_GAM_MODEL_LABELS["var"]: SOFT_BLUE,
    }
    frame = metrics.set_index("model").loc[order].reset_index()
    labels = [_wrap_label(label, width=18) for label in frame["model"]]
    colors = [color_map.get(label, ORANGE) for label in frame["model"]]

    fig, axes = book_subplots(1, 2, size="two_panel", background="none")
    metric_specs = (
        ("price_MAPE_pct", r"Price MAPE (\%)"),
        ("return_RMSE_bp", "Return RMSE (bp)"),
    )
    for axis, (column, xlabel) in zip(axes, metric_specs):
        bars = axis.barh(labels, frame[column], color=colors, alpha=0.95)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(True, axis="x", alpha=0.25)
        axis.set_xlim(0.0, float(frame[column].max()) * 1.18)
        for bar, value in zip(bars, frame[column]):
            axis.text(
                bar.get_width() + 0.02 * max(1.0, frame[column].max()),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=10,
            )

    axes[0].set_title("Held-out price accuracy")
    axes[1].set_title("Held-out return accuracy")
    fig.suptitle("Forecast comparison on the weekly finance sample")
    return fig


def _plot_finance_gam_forecast_axis(
    axis,
    *,
    history_prices: pd.Series,
    actual_prices: pd.Series,
    prediction_paths: dict[str, pd.Series],
    target_label: str,
    history_tail_weeks: int | None = None,
    post_test_only: bool = False,
    title: str,
    show_legend: bool = True,
):
    plotted_history = history_prices.iloc[-history_tail_weeks:] if history_tail_weeks is not None else history_prices

    if not post_test_only:
        axis.plot(
            plotted_history.index,
            plotted_history,
            color=CHAPTER_GRAY,
            linewidth=1.9,
            alpha=0.38,
            label="_nolegend_",
        )

    axis.plot(
        actual_prices.index,
        actual_prices,
        color=CHAPTER_GRAY,
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        label="Observed",
    )
    style_map = {
        FINANCE_GAM_MODEL_LABELS["gam"]: {
            "color": BLUE,
            "linestyle": "-",
            "linewidth": 2.2,
            "legend_label": FINANCE_GAM_LEGEND_LABELS["gam"],
        },
        FINANCE_GAM_MODEL_LABELS["var"]: {
            "color": SOFT_BLUE,
            "linestyle": "-.",
            "linewidth": 2.1,
            "legend_label": FINANCE_GAM_LEGEND_LABELS["var"],
        },
    }
    for label, path in prediction_paths.items():
        style = style_map.get(
            label,
            {
                "color": ORANGE,
                "linestyle": "--",
                "linewidth": 2.0,
                "legend_label": label,
            },
        ).copy()
        legend_label = style.pop("legend_label", label)
        axis.plot(
            path.index,
            path,
            label=legend_label,
            marker="o",
            markersize=3.8,
            **style,
        )

    split_date = actual_prices.index[0]
    axis.axvline(split_date, color=ANNOTATION_EDGE, linewidth=1.1, linestyle=":")

    if post_test_only:
        left_index = split_date - pd.Timedelta(days=4)
        right_index = actual_prices.index[-1] + pd.Timedelta(days=4)
        axis.set_xlim(left_index, right_index)

        price_values = [actual_prices, *prediction_paths.values()]
        combined = pd.concat(price_values)
        y_span = float(combined.max() - combined.min())
        y_pad = max(0.015, 0.18 * y_span)
        axis.set_ylim(float(combined.min() - y_pad), float(combined.max() + y_pad))
    elif history_tail_weeks is not None:
        left_index = plotted_history.index[0]
        right_index = actual_prices.index[-1] + pd.Timedelta(days=4)
        axis.set_xlim(left_index, right_index)

        price_values = [plotted_history, actual_prices, *prediction_paths.values()]
        combined = pd.concat(price_values)
        y_span = float(combined.max() - combined.min())
        y_pad = max(0.015, 0.18 * y_span)
        axis.set_ylim(float(combined.min() - y_pad), float(combined.max() + y_pad))

    axis.text(
        split_date,
        axis.get_ylim()[1],
        " test starts",
        ha="left",
        va="top",
        fontsize=10,
        color=CHAPTER_GRAY,
    )

    if post_test_only:
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    axis.set_title(title)
    axis.set_ylabel(f"Adjusted {target_label} price")
    axis.grid(True, alpha=0.26)
    if show_legend:
        legend_ncol = min(4, len(prediction_paths) + 1)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=legend_ncol,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.2,
            handlelength=2.6,
            handletextpad=0.6,
        )
    return axis


def plot_finance_gam_forecast(
    *,
    history_prices: pd.Series,
    actual_prices: pd.Series,
    prediction_paths: dict[str, pd.Series],
    target_label: str = "stock",
    history_tail_weeks: int | None = None,
    zoom_prediction_window: bool = False,
):
    fig, axis = book_subplots(size="wide", background="none")
    title = f"One-step price forecast for {target_label}"
    if zoom_prediction_window:
        title += " (zoomed)"
    _plot_finance_gam_forecast_axis(
        axis,
        history_prices=history_prices,
        actual_prices=actual_prices,
        prediction_paths=prediction_paths,
        target_label=target_label,
        history_tail_weeks=history_tail_weeks,
        post_test_only=False,
        title=title,
        show_legend=True,
    )
    return fig


def plot_finance_gam_forecast_panels(
    *,
    history_prices: pd.Series,
    actual_prices: pd.Series,
    prediction_paths: dict[str, pd.Series],
    target_label: str = "stock",
):
    fig, axes = book_subplots(
        1,
        2,
        size="two_panel",
        background="none",
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.55, 0.85)},
    )
    _plot_finance_gam_forecast_axis(
        axes[0],
        history_prices=history_prices,
        actual_prices=actual_prices,
        prediction_paths=prediction_paths,
        target_label=target_label,
        title=f"{target_label}: full forecast",
        show_legend=False,
    )
    _plot_finance_gam_forecast_axis(
        axes[1],
        history_prices=history_prices,
        actual_prices=actual_prices,
        prediction_paths=prediction_paths,
        target_label=target_label,
        post_test_only=True,
        title="Post-test window",
        show_legend=False,
    )
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.28, wspace=0.20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=min(4, len(labels)),
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.2,
        handlelength=2.6,
        handletextpad=0.6,
    )
    return fig


def plot_finance_gam_feature_effects(
    *,
    dataset: pd.DataFrame,
    effect_curves: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_labels: dict[str, str],
    target_label: str = "stock",
):
    fig, axes = book_subplots(2, 2, size="grid_2x2", background="none")
    axes = axes.ravel()
    percent_formatter = FuncFormatter(lambda value, _: f"{100 * value:.1f}\\%")
    curve_styles = {
        FINANCE_GAM_MODEL_LABELS["gam"]: {"color": BLUE, "linewidth": 2.3, "linestyle": "-"},
    }

    for axis, feature in zip(axes, feature_columns):
        feature_data = dataset[[feature, "target_return"]].sort_values(feature)
        curve_data = effect_curves.loc[effect_curves["feature"] == feature]

        axis.scatter(
            feature_data[feature],
            feature_data["target_return"],
            color=CHAPTER_GRAY,
            alpha=0.15,
            s=16,
        )
        if "model" in curve_data.columns and curve_data["model"].nunique() > 1:
            for model_label, model_curve in curve_data.groupby("model", sort=False):
                style = curve_styles.get(
                    model_label,
                    {"color": ORANGE, "linewidth": 2.0, "linestyle": "--"},
                )
                axis.plot(
                    model_curve["feature_value"],
                    model_curve["prediction"],
                    label=model_label,
                    **style,
                )
        else:
            axis.plot(
                curve_data["feature_value"],
                curve_data["prediction"],
                color=BLUE,
                linewidth=2.3,
            )

        axis.axhline(0.0, color=ANNOTATION_EDGE, linewidth=1.0)
        axis.set_title(feature_labels[feature])
        axis.yaxis.set_major_formatter(percent_formatter)
        axis.grid(True, alpha=0.22)

    for axis in axes[2:]:
        axis.set_xlabel("Lagged return value")
    axes[0].set_ylabel(f"Predicted next {finance_return_symbol(target_label)}")
    axes[2].set_ylabel(f"Predicted next {finance_return_symbol(target_label)}")
    if "model" in effect_curves.columns and effect_curves["model"].nunique() > 1:
        axes[0].legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
        )

    fig.suptitle("Spline GAM effect curves on lagged stock and mkt returns")
    return fig


def plot_finance_gam_prediction_breakdown(
    *,
    breakdown: pd.Series,
    actual_return: float,
    predicted_return: float,
    forecast_date: pd.Timestamp,
    model_label: str,
):
    values = breakdown.drop("Prediction").sort_values()
    labels = [_wrap_label(label, width=18) for label in values.index]
    colors = [ORANGE if value < 0 else BLUE for value in values.values]

    fig, axis = book_subplots(size="breakdown", background="none")
    bars = axis.barh(labels, values.values * 1e4, color=colors, alpha=0.95)
    axis.axvline(0.0, color=CHAPTER_GRAY, linewidth=1.0)
    axis.set_xlabel("Contribution to predicted next return (bp)")
    axis.set_title(f"{model_label} contribution breakdown for {forecast_date:%Y-%m-%d}")
    axis.grid(True, axis="x", alpha=0.22)

    x_pad = max(6.0, 0.14 * max(abs(values.min() * 1e4), abs(values.max() * 1e4)))
    axis.set_xlim(values.min() * 1e4 - x_pad, values.max() * 1e4 + x_pad)
    for bar, value in zip(bars, values.values * 1e4):
        axis.text(
            value + (1.5 if value >= 0 else -1.5),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )

    axis.text(
        0.98,
        0.05,
        f"Predicted {predicted_return * 100:.2f}\\%\nActual {actual_return * 100:.2f}\\%",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    return fig
