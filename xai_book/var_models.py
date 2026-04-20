from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Sequence

from . import mpl_config as _mpl_config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .plotting import (
    ANNOTATION_EDGE,
    BLUE,
    CHAPTER_GRAY,
    ORANGE,
    book_subplots,
    blue_orange_cmap,
    extended_book_palette,
)

DEFAULT_FINANCE_TICKERS = ("AAPL", "GOOG", "MSFT", "AMZN")
LECTURE_VAR_CANDIDATES = (
    {
        "label": "AAPL + AMZN",
        "tickers": ("AAPL", "AMZN"),
        "target_plot": "AAPL",
        "target_local": "AAPL",
        "impulse": "AMZN",
        "response": "AAPL",
    },
    {
        "label": "AAPL + MSFT + AMZN",
        "tickers": ("AAPL", "MSFT", "AMZN"),
        "target_plot": "AAPL",
        "target_local": "AAPL",
        "impulse": "AMZN",
        "response": "AAPL",
    },
    {
        "label": "AAPL + GOOG + MSFT + AMZN",
        "tickers": ("AAPL", "GOOG", "MSFT", "AMZN"),
        "target_plot": "AAPL",
        "target_local": "AAPL",
        "impulse": "AMZN",
        "response": "AAPL",
    },
    {
        "label": "AMZN + NFLX",
        "tickers": ("AMZN", "NFLX"),
        "target_plot": "AMZN",
        "target_local": "AMZN",
        "impulse": "NFLX",
        "response": "AMZN",
    },
)
DEFAULT_LECTURE_VAR_LABEL = "AAPL + AMZN"


def _default_series_labels(tickers: Sequence[str]) -> dict[str, str]:
    tickers = list(tickers)
    if len(tickers) == 2:
        return {
            tickers[0]: "stock",
            tickers[1]: "mkt",
        }
    return {ticker: ticker for ticker in tickers}


def _display_series_label(name: str, *, series_labels: dict[str, str] | None = None) -> str:
    if series_labels is None:
        return name
    return series_labels.get(name, name)


def _display_equation_label(label: str, *, series_labels: dict[str, str] | None = None) -> str:
    if label.endswith(" equation"):
        base_name = label.removesuffix(" equation")
        return f"{_display_series_label(base_name, series_labels=series_labels)} equation"
    return _display_series_label(label, series_labels=series_labels)


def _return_symbol(label: str, *, lag: int | None = None) -> str:
    if lag is None:
        return rf"$r_t^{{\mathrm{{{label}}}}}$"
    return rf"$r_{{t-{lag}}}^{{\mathrm{{{label}}}}}$"


def _relabel_contribution_terms(
    contributions: pd.Series,
    *,
    series_labels: dict[str, str] | None = None,
) -> pd.Series:
    if not series_labels:
        return contributions

    renamed: dict[str, float] = {}
    for term, value in contributions.items():
        if term == "intercept":
            renamed["Intercept"] = float(value)
            continue
        if term == "forecast_total":
            renamed["Prediction"] = float(value)
            continue
        if term.startswith("L") and "." in term:
            lag_prefix, _, source = term.partition(".")
            lag_number = int(lag_prefix.removeprefix("L"))
            renamed[_return_symbol(_display_series_label(source, series_labels=series_labels), lag=lag_number)] = float(value)
            continue
        renamed[_display_series_label(str(term), series_labels=series_labels)] = float(value)
    return pd.Series(renamed, name=contributions.name)


def _wrap_var_label(text: str, *, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def _with_inferred_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame

    freq = frame.index.freqstr or frame.index.inferred_freq
    if freq is None:
        return frame

    if frame.index.freq is not None and frame.index.freqstr == freq:
        return frame

    restored = frame.copy()
    restored.index = pd.DatetimeIndex(restored.index, freq=freq, name=restored.index.name)
    return restored


def _var_modules():
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    return VAR, adfuller, coint_johansen


def load_finance_var_prices(
    data_path: Path | str | None = None,
    *,
    tickers: Sequence[str] = DEFAULT_FINANCE_TICKERS,
) -> pd.DataFrame:
    tickers = list(tickers)
    csv_path = Path(data_path) if data_path is not None else None

    if csv_path is not None and csv_path.exists():
        raw = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        import plotly.express as px

        raw = px.data.stocks(datetimes=True)

    prices = (
        raw.rename(columns={"date": "Date"})
        .set_index("Date")[tickers]
        .sort_index()
    )

    return _with_inferred_frequency(prices)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff().dropna()
    return _with_inferred_frequency(returns)


def adf_table(frame: pd.DataFrame) -> pd.DataFrame:
    _, adfuller, _ = _var_modules()

    rows = []
    for column in frame.columns:
        statistic, pvalue, *_ = adfuller(frame[column].dropna(), autolag="AIC")
        rows.append(
            {
                "series": column,
                "adf_stat": statistic,
                "adf_pvalue": pvalue,
                "stationary_at_5pct": pvalue < 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("series")


def johansen_trace_table(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    _, _, coint_johansen = _var_modules()

    johansen = coint_johansen(np.log(prices), det_order=0, k_ar_diff=1)
    table = pd.DataFrame(
        {
            "trace_stat": johansen.lr1,
            "crit_90pct": johansen.cvt[:, 0],
            "crit_95pct": johansen.cvt[:, 1],
            "crit_99pct": johansen.cvt[:, 2],
            "reject_rank_at_95pct": johansen.lr1 > johansen.cvt[:, 1],
        },
        index=[f"r <= {index}" for index in range(len(prices.columns))],
    )
    estimated_rank_95 = int((johansen.lr1 > johansen.cvt[:, 1]).sum())
    return table, estimated_rank_95


def split_train_test(
    log_returns: pd.DataFrame,
    *,
    n_test: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = log_returns.iloc[:-n_test].copy()
    test = log_returns.iloc[-n_test:].copy()
    return _with_inferred_frequency(train), _with_inferred_frequency(test)


def rebuild_price_path(
    last_observed_price: pd.Series,
    forecast_returns: pd.DataFrame,
) -> pd.DataFrame:
    return last_observed_price * np.exp(forecast_returns.cumsum())


def flat_price_baseline(
    last_observed_price: pd.Series,
    index: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(last_observed_price.to_numpy().reshape(1, -1), len(index), axis=0),
        index=index,
        columns=last_observed_price.index,
    )


def metrics_table(actual: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in actual.columns:
        mae = mean_absolute_error(actual[column], pred[column])
        rmse = mean_squared_error(actual[column], pred[column]) ** 0.5
        mape = (np.abs(actual[column] - pred[column]) / actual[column]).mean() * 100
        rows.append({"series": column, "MAE": mae, "RMSE": rmse, "MAPE_pct": mape})
    return pd.DataFrame(rows).set_index("series")


def mean_metric(forecast_metrics: pd.DataFrame, model_name: str, metric_name: str) -> float:
    return float(forecast_metrics[(model_name, metric_name)].mean())


def rolling_one_step_var_forecast(
    train_returns: pd.DataFrame,
    future_returns: pd.DataFrame,
    *,
    lag: int,
) -> pd.DataFrame:
    VAR, _, _ = _var_modules()

    history = _with_inferred_frequency(train_returns.copy())
    future_returns = _with_inferred_frequency(future_returns)
    predictions = []
    for timestamp in future_returns.index:
        history = _with_inferred_frequency(history.sort_index())
        fitted = VAR(history).fit(lag)
        prediction = fitted.forecast(history.values[-lag:], steps=1)[0]
        predictions.append(prediction)
        history.loc[timestamp] = future_returns.loc[timestamp]

    forecast = pd.DataFrame(predictions, index=future_returns.index, columns=train_returns.columns)
    return _with_inferred_frequency(forecast)


def rolling_one_step_var_contributions(
    train_returns: pd.DataFrame,
    future_returns: pd.DataFrame,
    *,
    lag: int,
    target: str,
    step_index: int,
) -> pd.Series:
    VAR, _, _ = _var_modules()

    future_returns = _with_inferred_frequency(future_returns)
    if not 0 <= step_index < len(future_returns):
        raise IndexError(f"step_index={step_index} is out of range for {len(future_returns)} forecast steps")

    history = _with_inferred_frequency(train_returns.copy())
    for offset, timestamp in enumerate(future_returns.index):
        history = _with_inferred_frequency(history.sort_index())
        fitted = VAR(history).fit(lag)
        if offset == step_index:
            return one_step_contributions(fitted, history, target)
        history.loc[timestamp] = future_returns.loc[timestamp]

    raise RuntimeError("Failed to compute rolling VAR contributions for the requested forecast step")


def lag_backtest_table(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    n_test: int = 12,
    validation_size: int = 8,
    maxlags: int = 6,
) -> tuple[pd.DataFrame, int]:
    VAR, _, _ = _var_modules()

    train_full, _ = split_train_test(log_returns, n_test=n_test)
    validation_size = min(validation_size, max(4, len(train_full) // 4))
    train_core = _with_inferred_frequency(train_full.iloc[:-validation_size].copy())
    validation = _with_inferred_frequency(train_full.iloc[-validation_size:].copy())
    last_core_price = prices.loc[train_core.index[-1]]
    actual_validation_prices = prices.loc[validation.index]

    candidate_maxlag = min(maxlags, max(1, len(train_core) // 5))
    rows = []
    for lag in range(1, candidate_maxlag + 1):
        fitted = VAR(train_core).fit(lag)
        validation_returns = rolling_one_step_var_forecast(train_core, validation, lag=lag)
        validation_prices = rebuild_price_path(last_core_price, validation_returns)
        validation_metrics = metrics_table(actual_validation_prices, validation_prices)
        rows.append(
            {
                "lag": lag,
                "mean_mae": validation_metrics["MAE"].mean(),
                "mean_rmse": validation_metrics["RMSE"].mean(),
                "mean_mape_pct": validation_metrics["MAPE_pct"].mean(),
                "aic": fitted.aic,
                "bic": fitted.bic,
                "is_stable": bool(fitted.is_stable()),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["mean_rmse", "bic", "lag"]).reset_index(drop=True)
    selected_lag = int(frame.iloc[0]["lag"])
    return frame, selected_lag


def coefficient_frames(fitted_var) -> dict[int, pd.DataFrame]:
    frames = {}
    for lag in range(1, fitted_var.k_ar + 1):
        frames[lag] = pd.DataFrame(
            fitted_var.coefs[lag - 1],
            index=[f"{name} equation" for name in fitted_var.names],
            columns=fitted_var.names,
        )
    return frames


def one_step_contributions(fitted_var, history: pd.DataFrame, target: str) -> pd.Series:
    target_index = fitted_var.names.index(target)
    contributions = {"intercept": fitted_var.intercept[target_index]}

    for lag in range(1, fitted_var.k_ar + 1):
        observation = history.iloc[-lag]
        for source in fitted_var.names:
            source_index = fitted_var.names.index(source)
            coefficient = fitted_var.coefs[lag - 1, target_index, source_index]
            contributions[f"L{lag}.{source}"] = coefficient * observation[source]

    series = pd.Series(contributions, name=f"{target} one-step contribution")
    series["forecast_total"] = series.sum()
    return series


def granger_pvalue_matrix(fitted_var, columns: Sequence[str]) -> pd.DataFrame:
    columns = list(columns)
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    for caused in columns:
        for causing in columns:
            if caused == causing:
                continue
            matrix.loc[caused, causing] = fitted_var.test_causality(
                caused,
                [causing],
                kind="f",
            ).pvalue
    return matrix


def irf_response_frame(
    fitted_var,
    *,
    impulse: str,
    response: str,
    horizon: int = 10,
) -> pd.DataFrame:
    irf = fitted_var.irf(horizon)
    impulse_index = fitted_var.names.index(impulse)
    response_index = fitted_var.names.index(response)
    values = irf.orth_irfs[:, response_index, impulse_index]
    return pd.DataFrame({"horizon": np.arange(horizon + 1), "response": values})


def fevd_table(
    fitted_var,
    *,
    horizon: int = 10,
) -> pd.DataFrame:
    fevd = fitted_var.fevd(horizon)
    table = pd.DataFrame(index=fitted_var.names, columns=fitted_var.names, dtype=float)
    for index, target in enumerate(fitted_var.names):
        table.loc[target] = fevd.decomp[index, horizon - 1]
    return table


def run_var_finance_analysis(
    *,
    data_path: Path | str | None = None,
    tickers: Sequence[str] = DEFAULT_FINANCE_TICKERS,
    n_test: int = 12,
    validation_size: int = 8,
    maxlags: int = 6,
    target_plot: str = "AAPL",
    target_local: str = "GOOG",
    impulse: str = "AAPL",
    response: str = "GOOG",
    series_labels: dict[str, str] | None = None,
    irf_horizon: int = 10,
    fevd_horizon: int = 10,
) -> dict[str, object]:
    VAR, _, _ = _var_modules()

    prices = load_finance_var_prices(data_path, tickers=tickers)
    tickers = tuple(prices.columns)
    target_plot = target_plot if target_plot in tickers else tickers[0]
    target_local = target_local if target_local in tickers else tickers[0]
    impulse = impulse if impulse in tickers else tickers[-1]
    response = response if response in tickers else tickers[0]
    if series_labels is None:
        series_labels = _default_series_labels(tickers)
    else:
        series_labels = {ticker: series_labels.get(ticker, ticker) for ticker in tickers}
    log_returns = compute_log_returns(prices)
    levels_adf = adf_table(prices)
    returns_adf = adf_table(log_returns)
    johansen_table, estimated_rank_95 = johansen_trace_table(prices)

    train, test = split_train_test(log_returns, n_test=n_test)
    lag_selection, selected_lag = lag_backtest_table(
        prices,
        log_returns,
        n_test=n_test,
        validation_size=validation_size,
        maxlags=maxlags,
    )

    fitted = VAR(train).fit(selected_lag)
    whiteness = fitted.test_whiteness(nlags=max(selected_lag + 1, 6))
    normality = fitted.test_normality()
    diagnostics = pd.DataFrame(
        {"value": [
            selected_lag,
            bool(fitted.is_stable()),
            fitted.aic,
            fitted.bic,
            whiteness.pvalue,
            normality.pvalue,
        ]},
        index=[
            "selected_lag",
            "is_stable",
            "aic",
            "bic",
            "whiteness_pvalue",
            "normality_pvalue",
        ],
    )

    forecast_returns = rolling_one_step_var_forecast(train, test, lag=selected_lag)
    last_train_price = prices.loc[train.index[-1]]
    train_prices = prices.loc[: train.index[-1]].copy()
    pred_prices = rebuild_price_path(last_train_price, forecast_returns)
    actual_prices = prices.loc[test.index]
    flat_baseline = flat_price_baseline(last_train_price, actual_prices.index)

    var_metrics = metrics_table(actual_prices, pred_prices)
    baseline_metrics = metrics_table(actual_prices, flat_baseline)
    forecast_metrics = pd.concat(
        {"VAR rolling one-step": var_metrics, "Flat baseline": baseline_metrics},
        axis=1,
    )
    forecast_summary = pd.Series(
        {
            "mean_var_mae": mean_metric(forecast_metrics, "VAR rolling one-step", "MAE"),
            "mean_var_rmse": mean_metric(forecast_metrics, "VAR rolling one-step", "RMSE"),
            "mean_var_mape_pct": mean_metric(forecast_metrics, "VAR rolling one-step", "MAPE_pct"),
            "mean_baseline_mape_pct": mean_metric(forecast_metrics, "Flat baseline", "MAPE_pct"),
            "mean_mape_gain_pct": (
                mean_metric(forecast_metrics, "Flat baseline", "MAPE_pct")
                - mean_metric(forecast_metrics, "VAR rolling one-step", "MAPE_pct")
            ),
            "validation_mape_pct": float(lag_selection.iloc[0]["mean_mape_pct"]),
        },
        name="forecast_summary",
    )

    coefficients = coefficient_frames(fitted)
    local_contributions = _relabel_contribution_terms(
        one_step_contributions(fitted, train, target_local),
        series_labels=series_labels,
    )
    granger_pvalues = granger_pvalue_matrix(fitted, train.columns)
    irf_frame = irf_response_frame(
        fitted,
        impulse=impulse,
        response=response,
        horizon=irf_horizon,
    )
    fevd_frame = fevd_table(fitted, horizon=fevd_horizon)

    return {
        "prices": prices,
        "log_returns": log_returns,
        "levels_adf": levels_adf,
        "returns_adf": returns_adf,
        "johansen_table": johansen_table,
        "estimated_rank_95": estimated_rank_95,
        "train": train,
        "test": test,
        "lag_selection": lag_selection,
        "selected_lag": selected_lag,
        "fitted": fitted,
        "diagnostics": diagnostics,
        "forecast_returns": forecast_returns,
        "train_prices": train_prices,
        "pred_prices": pred_prices,
        "actual_prices": actual_prices,
        "flat_baseline": flat_baseline,
        "forecast_metrics": forecast_metrics,
        "forecast_summary": forecast_summary,
        "coefficients": coefficients,
        "local_contributions": local_contributions,
        "granger_pvalues": granger_pvalues,
        "irf_frame": irf_frame,
        "fevd_table": fevd_frame,
        "tickers": list(tickers),
        "series_labels": series_labels,
        "target_plot": target_plot,
        "target_local": target_local,
        "impulse": impulse,
        "response": response,
    }


def compare_var_candidate_models(
    *,
    data_path: Path | str | None = None,
    candidates: Sequence[dict[str, object]] = LECTURE_VAR_CANDIDATES,
    n_test: int = 12,
    validation_size: int = 8,
    maxlags: int = 6,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    analyses: dict[str, dict[str, object]] = {}
    rows = []

    for spec in candidates:
        label = str(spec["label"])
        analysis = run_var_finance_analysis(
            data_path=data_path,
            tickers=tuple(spec["tickers"]),
            n_test=n_test,
            validation_size=validation_size,
            maxlags=maxlags,
            target_plot=str(spec.get("target_plot", spec["tickers"][0])),
            target_local=str(spec.get("target_local", spec["tickers"][0])),
            impulse=str(spec.get("impulse", spec["tickers"][-1])),
            response=str(spec.get("response", spec["tickers"][0])),
        )
        analyses[label] = analysis
        rows.append(
            {
                "candidate": label,
                "tickers": ", ".join(spec["tickers"]),
                "n_series": len(spec["tickers"]),
                "selected_lag": analysis["selected_lag"],
                "validation_mape_pct": analysis["forecast_summary"]["validation_mape_pct"],
                "test_var_mape_pct": analysis["forecast_summary"]["mean_var_mape_pct"],
                "test_baseline_mape_pct": analysis["forecast_summary"]["mean_baseline_mape_pct"],
                "mape_gain_pct": analysis["forecast_summary"]["mean_mape_gain_pct"],
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["test_var_mape_pct", "selected_lag", "n_series", "candidate"]
    ).reset_index(drop=True)
    return frame, analyses


def plot_price_indices(
    prices: pd.DataFrame,
    *,
    series_labels: dict[str, str] | None = None,
):
    fig, axis = book_subplots(size="wide")
    colors = extended_book_palette()
    display_prices = prices.rename(columns=lambda name: _display_series_label(name, series_labels=series_labels))
    display_prices.plot(ax=axis, color=colors[: len(display_prices.columns)], linewidth=2.0)
    axis.set_title("Normalized price indices")
    axis.set_ylabel("Normalized level")
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=min(3, len(prices.columns)),
        frameon=False,
        borderaxespad=0.0,
    )
    return fig


def plot_lag_backtest(lag_selection: pd.DataFrame):
    fig, axis = book_subplots(size="single")
    axis.plot(lag_selection["lag"], lag_selection["mean_rmse"], marker="o", color=BLUE)
    axis.set_title("Validation RMSE by lag")
    axis.set_xlabel("Lag order")
    axis.set_ylabel("Mean RMSE on validation prices")
    axis.set_xticks(lag_selection["lag"])
    axis.grid(True, alpha=0.3)
    return fig


def _wrapped_labels(values: Sequence[str], *, width: int = 20) -> list[str]:
    return [_wrap_var_label(value.replace(" + ", " + "), width=width) for value in values]


def plot_candidate_comparison(candidate_frame: pd.DataFrame):
    fig, axis = book_subplots(size="wide")
    y_positions = np.arange(len(candidate_frame))
    labels = _wrapped_labels(candidate_frame["candidate"], width=20)
    bar_height = 0.34

    axis.barh(
        y_positions + bar_height / 2,
        candidate_frame["test_baseline_mape_pct"],
        height=bar_height,
        color=CHAPTER_GRAY,
        alpha=0.28,
        label="Flat baseline",
    )
    axis.barh(
        y_positions - bar_height / 2,
        candidate_frame["test_var_mape_pct"],
        height=bar_height,
        color=BLUE,
        alpha=0.95,
        label="VAR",
    )
    axis.scatter(
        candidate_frame["validation_mape_pct"],
        y_positions,
        color=ORANGE,
        s=44,
        zorder=3,
        label="Validation MAPE",
    )

    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(r"Mean MAPE (\%)")
    axis.set_title("Candidate VAR systems on the weekly stock sample")
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    x_max = float(candidate_frame[["test_var_mape_pct", "test_baseline_mape_pct"]].to_numpy().max())
    axis.set_xlim(0.0, x_max + 2.7)
    for y_position, (_, row) in zip(y_positions, candidate_frame.iterrows()):
        axis.text(
            x_max + 0.35,
            y_position,
            f"lag {int(row['selected_lag'])}",
            va="center",
            ha="left",
            fontsize=10,
            color=CHAPTER_GRAY,
        )
    return fig


def plot_forecast_vs_actual(
    history_prices: pd.DataFrame,
    actual_prices: pd.DataFrame,
    pred_prices: pd.DataFrame,
    *,
    target: str,
    series_labels: dict[str, str] | None = None,
):
    fig, axis = book_subplots(size="wide")
    axis.plot(
        history_prices.index,
        history_prices[target],
        color=CHAPTER_GRAY,
        linewidth=1.9,
        alpha=0.65,
        label="Observed history",
    )
    axis.plot(
        actual_prices.index,
        actual_prices[target],
        color=ORANGE,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="True next price",
    )
    axis.plot(
        pred_prices.index,
        pred_prices[target],
        color=BLUE,
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=4.5,
        label="Predicted next price",
    )
    axis.axvline(actual_prices.index[0], linewidth=1.1, linestyle=":", color=CHAPTER_GRAY, alpha=0.9)
    target_label = _display_series_label(target, series_labels=series_labels)
    axis.set_title(_wrap_var_label(f"History plus one-step-ahead price forecasts for {target_label}", width=46))
    axis.set_ylabel("Normalized level")
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )
    return fig


def plot_forecast_summary(
    history_prices: pd.DataFrame,
    actual_prices: pd.DataFrame,
    pred_prices: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    *,
    title: str = "Rolling one-step VAR forecasts",
    series_labels: dict[str, str] | None = None,
):
    columns = list(actual_prices.columns)
    fig, axes = book_subplots(len(columns), 1, size="grid", sharex=True, extra_height=0.3)
    if len(columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, columns):
        axis.plot(
            history_prices.index,
            history_prices[column],
            color=CHAPTER_GRAY,
            linewidth=1.8,
            alpha=0.65,
            label="Observed history",
        )
        axis.plot(
            actual_prices.index,
            actual_prices[column],
            color=ORANGE,
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            label="True next price",
        )
        axis.plot(
            pred_prices.index,
            pred_prices[column],
            color=BLUE,
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=4.0,
            label="Predicted next price",
        )
        axis.set_ylabel(_display_series_label(column, series_labels=series_labels))
        axis.grid(True, alpha=0.25)
        axis.axvline(actual_prices.index[0], linewidth=1.0, linestyle=":", color=CHAPTER_GRAY, alpha=0.9)

        var_mape = forecast_metrics.loc[column, ("VAR rolling one-step", "MAPE_pct")]
        baseline_mape = forecast_metrics.loc[column, ("Flat baseline", "MAPE_pct")]
        axis.text(
            0.02,
            0.96,
            f"MAPE {var_mape:.1f}\\% vs {baseline_mape:.1f}\\%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "none", "edgecolor": ANNOTATION_EDGE, "alpha": 1.0},
        )

    axes[0].set_title(title)
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )
    axes[-1].set_xlabel("Date")
    return fig


def plot_coefficient_heatmaps(
    coefficients: dict[int, pd.DataFrame],
    *,
    series_labels: dict[str, str] | None = None,
):
    lags = sorted(coefficients)
    fig, axes = book_subplots(1, len(lags), size="grid", squeeze=False, extra_height=1.0)
    vmax = max(frame.abs().to_numpy().max() for frame in coefficients.values())
    cmap = blue_orange_cmap()

    for axis, lag in zip(axes[0], lags):
        frame = coefficients[lag]
        image = axis.imshow(frame.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_title(f"Lag {lag} coefficients")
        axis.set_xticks(range(len(frame.columns)))
        axis.set_xticklabels(
            [_wrap_var_label(_display_series_label(label, series_labels=series_labels), width=12) for label in frame.columns],
            rotation=0,
            ha="center",
        )
        axis.set_yticks(range(len(frame.index)))
        axis.set_yticklabels(
            [_wrap_var_label(_display_equation_label(label, series_labels=series_labels), width=14) for label in frame.index]
        )
        for row in range(frame.shape[0]):
            for column in range(frame.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{frame.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    return fig


def plot_one_step_contributions(
    contributions: pd.Series,
    *,
    target: str,
    series_labels: dict[str, str] | None = None,
):
    fig, axis = book_subplots(size="breakdown", background="none")
    values = contributions.drop(labels=["forecast_total", "Prediction"], errors="ignore")
    colors = [BLUE if value >= 0 else ORANGE for value in values]
    labels = [_wrap_var_label(label, width=16) for label in values.index]
    bars = axis.barh(labels, values.values, color=colors)
    axis.axvline(0.0, linewidth=1.0, color=CHAPTER_GRAY)
    target_label = _display_series_label(target, series_labels=series_labels)
    axis.set_title(f"One-step forecast contribution for {target_label}")
    axis.set_xlabel("Contribution to forecasted log return")
    axis.grid(True, axis="x", alpha=0.3)
    x_pad = max(0.01, 0.12 * max(abs(values.min()), abs(values.max())))
    axis.set_xlim(values.min() - x_pad, values.max() + x_pad)
    for bar, value in zip(bars, values.values):
        axis.text(
            value + (0.01 if value >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9.5,
        )
    return fig


def plot_irf_response(
    irf_frame: pd.DataFrame,
    *,
    impulse: str,
    response: str,
    series_labels: dict[str, str] | None = None,
):
    fig, axis = book_subplots(size="single")
    axis.plot(irf_frame["horizon"], irf_frame["response"], marker="o", color=BLUE)
    axis.axhline(0, linewidth=1, color=CHAPTER_GRAY)
    impulse_label = _display_series_label(impulse, series_labels=series_labels)
    response_label = _display_series_label(response, series_labels=series_labels)
    axis.set_title(_wrap_var_label(f"Orthogonalized IRF: {response_label} response to {impulse_label}", width=42))
    axis.set_xlabel("Horizon")
    axis.set_ylabel("Response in log-return units")
    axis.grid(True, alpha=0.3)
    return fig
