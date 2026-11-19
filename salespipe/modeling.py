"""Regresi tren sales bulanan per produk (materi: Supervised Learning & Regression).

Target : sales bulanan per produk.
Fitur  : t = urutan bulan.
Split  : kronologis per bulan (train/validasi/test), bukan acak, agar data
         masa depan tidak bocor ke training (data leakage).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import ModelConfig, Schema

log = logging.getLogger(__name__)

FEATURE = ["t"]
TARGET = "Sales"
SET_NAMES = ("train", "validasi", "test")
NAIVE = "Naive (rata-rata train)"


# Modeling functions
def build_monthly(
    df: pd.DataFrame,
    s: Schema,
    cfg: ModelConfig,
) -> pd.DataFrame:
    df = df.copy()
    df[s.order_date] = pd.to_datetime(df[s.order_date])

    # Preserve the historical calendar before filtering sales.
    months = pd.date_range(
        df[s.order_date].min().to_period("M").to_timestamp(),
        df[s.order_date].max().to_period("M").to_timestamp(),
        freq="MS",
    )

    # Preserve all products present in the cleaned dataset.
    products = cast(
        pd.DataFrame,
        df.loc[:, [s.product_id, s.product_name, s.category]],
    ).drop_duplicates()

    # Only completed transactions contribute to sales.
    if cfg.sales_status is not None:
        df = cast(
            pd.DataFrame,
            df.loc[df[s.status] == cfg.sales_status].copy(),
        )

    df["Month"] = df[s.order_date].dt.to_period("M").dt.to_timestamp()

    sales = df.groupby([s.product_id, "Month"], as_index=False).agg(
        **{s.sales: (s.sales, "sum")}
    )

    # Create every product-month combination.
    grid = products.merge(
        pd.DataFrame({"Month": months}),
        how="cross",
    )

    monthly = grid.merge(
        sales,
        how="left",
        on=[s.product_id, "Month"],
        validate="one_to_one",
    )

    monthly = monthly.rename(columns={s.sales: TARGET})
    monthly[TARGET] = monthly[TARGET].fillna(0.0)

    monthly["t"] = (
        (monthly["Month"].dt.year - months[0].year) * 12
        + monthly["Month"].dt.month
        - months[0].month
    )

    # Encode month of year as cyclical features.
    month_number = monthly["Month"].dt.month

    monthly["month_sin"] = np.sin(2 * np.pi * (month_number - 1) / 12)

    monthly["month_cos"] = np.cos(2 * np.pi * (month_number - 1) / 12)

    return monthly.sort_values(["Month", s.product_id]).reset_index(drop=True)


def split_months(
    months: pd.DatetimeIndex, cfg: ModelConfig
) -> dict[str, pd.DatetimeIndex]:
    n = len(months)
    n_train, n_val = round(n * cfg.train_ratio), round(n * cfg.val_ratio)

    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError(
            f"{n} bulan terlalu sedikit untuk split {cfg.train_ratio}/{cfg.val_ratio}"
        )

    return {
        "train": months[:n_train],
        "validasi": months[n_train : n_train + n_val],
        "test": months[n_train + n_val :],
    }


def _poly(degree: int):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def build_models(cfg: ModelConfig) -> dict:
    models = {
        NAIVE: lambda: DummyRegressor(strategy="mean"),
        "Linear (baseline)": lambda: LinearRegression(),
    }
    for d in cfg.poly_degrees:
        models[f"Polynomial {d}"] = lambda d=d: _poly(d)
    for a in cfg.alphas:
        models[f"Ridge (a={a})"] = lambda a=a: make_pipeline(
            StandardScaler(), Ridge(alpha=a)
        )
        models[f"Lasso (a={a})"] = lambda a=a: make_pipeline(
            StandardScaler(), Lasso(alpha=a, max_iter=10000)
        )
    return models


def predict_per_product(
    make_model,
    train: pd.DataFrame,
    sets: dict[str, pd.DataFrame],
    product_col: str,
):
    """Train and predict independently for each product."""
    predictions = {
        name: pd.Series(index=data.index, dtype=float) for name, data in sets.items()
    }

    for pid, group in train.groupby(product_col):
        model = make_model().fit(
            group[FEATURE],
            group[TARGET],
        )

        for split_name, data in sets.items():
            rows = data.loc[data[product_col] == pid]

            # Skip prediction when the split contains no observations.
            if rows.empty:
                continue

            predictions[split_name].loc[rows.index] = model.predict(rows[FEATURE])

    return predictions


def predict_seasonal_naive(
    sets: dict[str, pd.DataFrame],
    product_col: str,
    seasonal_period: int = 12,
) -> dict[str, pd.Series]:
    """Predict sales using the corresponding month from the previous year."""
    if seasonal_period < 1:
        raise ValueError("Seasonal period harus minimal 1.")

    historical = pd.concat([sets["train"], sets["validasi"], sets["test"]])

    historical = historical.set_index([product_col, "t"])[TARGET]

    predictions = {}

    for split_name in SET_NAMES:
        current = sets[split_name]

        previous_index = pd.MultiIndex.from_arrays(
            [
                current[product_col],
                current["t"] - seasonal_period,
            ],
            names=[product_col, "t"],
        )

        predictions[split_name] = (
            historical.reindex(previous_index).set_axis(current.index).astype(float)
        )

    return predictions


def predict_moving_average(
    sets: dict[str, pd.DataFrame],
    product_col: str,
    window: int = 3,
) -> dict[str, pd.Series]:
    """Predict sales using a recursive moving average."""
    if window < 1:
        raise ValueError("Window harus minimal 1.")

    predictions = {
        name: pd.Series(index=data.index, dtype=float) for name, data in sets.items()
    }

    for pid, train_group in sets["train"].groupby(product_col):
        history = train_group.sort_values("t")[TARGET].tolist()

        # Training predictions use the previous actual observations.
        for index in range(window, len(history)):
            predictions["train"].loc[train_group.sort_values("t").index[index]] = float(
                np.mean(history[index - window : index])
            )

        # Forecast validation and test recursively from the training period.
        for split_name in ("validasi", "test"):
            split_data = sets[split_name]
            product_data = cast(
                pd.DataFrame,
                split_data.loc[split_data[product_col] == pid].sort_values("t"),
            )

            for index in product_data.index:
                if len(history) < window:
                    raise ValueError(f"Insufficient history for product {pid}.")

                prediction = float(np.mean(history[-window:]))

                predictions[split_name].loc[index] = prediction
                history.append(prediction)

    return predictions


def calculate_wape(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    denominator = np.abs(y_true).sum()

    if denominator == 0:
        return float("nan")

    numerator = np.abs(y_true - y_pred).sum()

    return float(numerator / denominator * 100)


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "WAPE": calculate_wape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(
    sets: dict[str, pd.DataFrame],
    s: Schema,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """Compare regression models and forecasting baselines."""
    rows: list[dict[str, str | float]] = []

    # 1. Evaluate regression models.
    for name, make_model in build_models(cfg).items():
        preds = predict_per_product(
            make_model,
            sets["train"],
            sets,
            s.product_id,
        )

        row: dict[str, str | float] = {"model": name}

        for split_name in SET_NAMES:
            y_true = cast(pd.Series, sets[split_name][TARGET])
            y_pred = cast(pd.Series, preds[split_name])

            metrics = regression_metrics(y_true, y_pred)

            for metric, value in metrics.items():
                row[f"{split_name}_{metric}"] = value

        rows.append(row)

    # 2. Evaluate forecasting baselines separately.
    baseline_predictions = {
        "Seasonal Naive": predict_seasonal_naive(
            sets,
            s.product_id,
        ),
        "Moving Average": predict_moving_average(
            sets,
            s.product_id,
            window=3,
        ),
    }

    for model_name, preds in baseline_predictions.items():
        row = {"model": model_name}

        for split_name in SET_NAMES:
            y_pred = preds[split_name].dropna()

            if y_pred.empty:
                for metric in ("MAE", "MSE", "RMSE", "WAPE", "R2"):
                    row[f"{split_name}_{metric}"] = float("nan")

                continue

            y_true = cast(
                pd.Series,
                sets[split_name].loc[y_pred.index, TARGET],
            )

            metrics = regression_metrics(y_true, y_pred)

            for metric, value in metrics.items():
                row[f"{split_name}_{metric}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def compare_models_per_product(
    sets: dict[str, pd.DataFrame],
    s: Schema,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """Evaluate MAE, RMSE, and WAPE per product and model."""
    rows: list[dict] = []

    # Evaluate regression models.
    for name, make_model in build_models(cfg).items():
        preds = predict_per_product(
            make_model,
            sets["train"],
            sets,
            s.product_id,
        )

        for pid, train_group in sets["train"].groupby(s.product_id):
            row = {
                s.product_id: pid,
                s.product_name: train_group[s.product_name].iloc[0],
                s.category: train_group[s.category].iloc[0],
                "model": name,
            }

            for split_name in SET_NAMES:
                product_data = sets[split_name].loc[
                    sets[split_name][s.product_id] == pid
                ]

                if product_data.empty:
                    continue

                y_true = cast(pd.Series, product_data[TARGET])
                y_pred = cast(
                    pd.Series,
                    preds[split_name].loc[product_data.index],
                )

                mse = mean_squared_error(y_true, y_pred)

                row[f"{split_name}_MAE"] = mean_absolute_error(
                    y_true,
                    y_pred,
                )
                row[f"{split_name}_RMSE"] = np.sqrt(mse)
                row[f"{split_name}_WAPE"] = calculate_wape(
                    y_true,
                    y_pred,
                )

            rows.append(row)

    # Evaluate forecasting baselines.
    baseline_predictions = {
        "Seasonal Naive": predict_seasonal_naive(
            sets,
            s.product_id,
        ),
        "Moving Average": predict_moving_average(
            sets,
            s.product_id,
            window=3,
        ),
    }

    for model_name, preds in baseline_predictions.items():
        for pid, train_group in sets["train"].groupby(s.product_id):
            row = {
                s.product_id: pid,
                s.product_name: train_group[s.product_name].iloc[0],
                s.category: train_group[s.category].iloc[0],
                "model": model_name,
            }

            for split_name in SET_NAMES:
                product_data = sets[split_name].loc[
                    sets[split_name][s.product_id] == pid
                ]

                if product_data.empty:
                    continue

                y_pred = preds[split_name].reindex(product_data.index).dropna()

                if y_pred.empty:
                    continue

                y_true = cast(
                    pd.Series,
                    product_data.loc[y_pred.index, TARGET],
                )

                mse = mean_squared_error(y_true, y_pred)

                row[f"{split_name}_MAE"] = mean_absolute_error(
                    y_true,
                    y_pred,
                )
                row[f"{split_name}_RMSE"] = np.sqrt(mse)
                row[f"{split_name}_WAPE"] = calculate_wape(
                    y_true,
                    y_pred,
                )

            rows.append(row)

    return (
        pd.DataFrame(rows).sort_values(["model", s.product_id]).reset_index(drop=True)
    )


def best_model(results: pd.DataFrame) -> str:
    """Select the best overall model using validation RMSE."""
    if results.empty:
        raise ValueError("Tidak ada hasil evaluasi model.")

    validation_rmse = cast(
        pd.Series,
        results["validasi_RMSE"],
    )

    valid_scores = validation_rmse.dropna()

    if valid_scores.empty:
        raise ValueError("Tidak ada validation RMSE yang valid.")

    best_idx = valid_scores.idxmin()

    return str(results.at[best_idx, "model"])


def best_ml_model(results: pd.DataFrame) -> str:
    """Select the best machine learning model, excluding Naive."""
    candidates = cast(
        pd.DataFrame,
        results.loc[
            ~results["model"].isin([NAIVE, "Seasonal Naive", "Moving Average"])
        ],
    )

    if candidates.empty:
        raise ValueError("Tidak ada model ML yang dapat dipilih.")

    validation_rmse = cast(
        pd.Series,
        candidates["validasi_RMSE"],
    )

    valid_scores = validation_rmse.dropna()

    if valid_scores.empty:
        raise ValueError("Tidak ada validation RMSE model ML yang valid.")

    best_idx = valid_scores.idxmin()

    return str(candidates.at[best_idx, "model"])


def best_model_per_product(
    product_metrics: pd.DataFrame,
    product_col: str,
) -> pd.DataFrame:
    """Select the best model for each product using validation RMSE."""
    required_columns = {
        product_col,
        "model",
        "validasi_RMSE",
    }

    missing_columns = required_columns.difference(product_metrics.columns)

    if missing_columns:
        raise ValueError(f"Kolom evaluasi tidak ditemukan: {sorted(missing_columns)}")

    valid_metrics = product_metrics.dropna(subset=["validasi_RMSE"])

    if valid_metrics.empty:
        raise ValueError("Tidak ada validation RMSE yang valid.")

    best_indices = valid_metrics.groupby(product_col)["validasi_RMSE"].idxmin()

    return (
        valid_metrics.loc[best_indices].sort_values(product_col).reset_index(drop=True)
    )


def forecast_future(
    monthly: pd.DataFrame,
    product_metrics: pd.DataFrame,
    s: Schema,
    cfg: ModelConfig,
    horizon: int = 3,
) -> pd.DataFrame:
    """Forecast future monthly sales using the best model per product."""
    if horizon < 1:
        raise ValueError("Horizon harus minimal 1 bulan.")

    if monthly.empty:
        raise ValueError("Dataset monthly tidak boleh kosong.")

    selected_models = best_model_per_product(
        product_metrics,
        s.product_id,
    )

    model_factories = build_models(cfg)

    month_column = cast(pd.Series, monthly["Month"])
    time_column = cast(pd.Series, monthly["t"])

    last_month = pd.Timestamp(month_column.max())
    last_t = int(time_column.max())

    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    future_t = np.arange(
        last_t + 1,
        last_t + horizon + 1,
    )

    future_features = pd.DataFrame({"t": future_t})

    rows: list[dict[str, object]] = []

    for _, selected in selected_models.iterrows():
        pid = selected[s.product_id]
        model_name = str(selected["model"])

        history = cast(
            pd.DataFrame,
            monthly.loc[monthly[s.product_id] == pid].sort_values("t"),
        )

        if history.empty:
            raise ValueError(f"Historical data unavailable for product {pid}.")

        if model_name == "Seasonal Naive":
            historical_sales = cast(
                pd.Series,
                history.set_index("t")[TARGET],
            )

            predictions = historical_sales.reindex(future_t - 12).to_numpy(dtype=float)

            if np.isnan(predictions).any():
                raise ValueError(
                    f"Historical seasonal values unavailable for product {pid}."
                )

        elif model_name == "Moving Average":
            window = 3
            ma_history = history[TARGET].astype(float).tolist()
            predictions = []

            if len(ma_history) < window:
                raise ValueError(
                    f"Insufficient history for moving average, product {pid}."
                )

            for _ in range(horizon):
                prediction = max(
                    0.0,
                    float(np.mean(ma_history[-window:])),
                )

                predictions.append(prediction)
                ma_history.append(prediction)

            predictions = np.array(predictions)

        else:
            if model_name not in model_factories:
                raise ValueError(f"Unknown model: {model_name}")

            model = model_factories[model_name]()

            model.fit(
                history[FEATURE],
                history[TARGET],
            )

            predictions = model.predict(future_features)

        for month, prediction in zip(
            future_months,
            predictions,
            strict=True,
        ):
            rows.append(
                {
                    s.product_id: pid,
                    s.product_name: selected[s.product_name],
                    s.category: selected[s.category],
                    "Month": month,
                    "model": model_name,
                    "PredictedSales": max(
                        0.0,
                        float(prediction),
                    ),
                }
            )

    return pd.DataFrame(rows)


def product_trends(
    monthly: pd.DataFrame, train_months: pd.DatetimeIndex, s: Schema
) -> pd.DataFrame:
    """Slope Linear Regression per produk (negatif = turun), dari data train dan seluruh data."""
    rows = []
    for key, g in monthly.groupby([s.product_id, s.product_name, s.category]):
        pid, name, cat = cast(tuple[object, object, object], key)
        month_column = cast(pd.Series, g["Month"])
        train_mask = month_column.isin(train_months.tolist())
        tr = cast(pd.DataFrame, g.loc[train_mask])
        lr_all = LinearRegression().fit(g[FEATURE], g[TARGET])
        lr_train = LinearRegression().fit(tr[FEATURE], tr[TARGET])
        rows.append(
            {
                s.product_id: pid,
                s.product_name: name,
                s.category: cat,
                "avg_monthly_sales": g[TARGET].mean(),
                "slope_train": lr_train.coef_[0],
                "slope_all": lr_all.coef_[0],
                "R2_all": lr_all.score(g[FEATURE], g[TARGET]),
            }
        )
    out = pd.DataFrame(rows).sort_values("slope_train")
    out["trend_train"] = np.where(out["slope_train"] < 0, "turun", "naik/stabil")
    return out


def build_walk_forward_folds(
    monthly: pd.DataFrame,
    initial_train_months: int = 12,
    validation_months: int = 3,
    step_months: int = 3,
    test_months: int = 3,
) -> list[dict[str, pd.DataFrame]]:
    """Build expanding-window folds while preserving the final test period."""
    if monthly.empty:
        raise ValueError("Dataset monthly tidak boleh kosong.")

    if (
        min(
            initial_train_months,
            validation_months,
            step_months,
        )
        < 1
    ):
        raise ValueError("Jumlah bulan harus minimal 1.")

    if test_months < 1:
        raise ValueError("Test period harus minimal 1 bulan.")

    months = sorted(monthly["Month"].drop_duplicates())

    development_months = months[:-test_months]

    if len(development_months) < (initial_train_months + validation_months):
        raise ValueError("Insufficient months for walk-forward validation.")

    folds: list[dict[str, pd.DataFrame]] = []
    train_end = initial_train_months

    while train_end + validation_months <= len(development_months):
        train_months = development_months[:train_end]

        valid_months = development_months[train_end : train_end + validation_months]

        train_data = cast(
            pd.DataFrame,
            monthly.loc[monthly["Month"].isin(train_months)].copy(),
        )

        valid_data = cast(
            pd.DataFrame,
            monthly.loc[monthly["Month"].isin(valid_months)].copy(),
        )

        folds.append(
            {
                "train": train_data,
                "validasi": valid_data,
            }
        )

        train_end += step_months

    return folds


def evaluate_walk_forward(
    folds: list[dict[str, pd.DataFrame]],
    s: Schema,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """
    Evaluate regression models and forecasting baselines
    across walk-forward validation folds.
    """
    rows: list[dict[str, str | int | float]] = []

    for fold_number, fold in enumerate(folds, start=1):
        # Track the number of results before evaluating this fold.
        fold_rows_before = len(rows)

        # Add an empty test DataFrame because the existing prediction
        # functions expect train, validation, and test keys.
        sets: dict[str, pd.DataFrame] = {
            "train": fold["train"],
            "validasi": fold["validasi"],
            "test": fold["validasi"].iloc[0:0].copy(),
        }

        # 1. Evaluate regression models.
        for model_name, make_model in build_models(cfg).items():
            predictions = predict_per_product(
                make_model,
                sets["train"],
                sets,
                s.product_id,
            )

            y_pred = cast(
                pd.Series,
                predictions["validasi"],
            ).dropna()

            if y_pred.empty:
                continue

            y_true = cast(
                pd.Series,
                sets["validasi"].loc[y_pred.index, TARGET],
            )

            metrics = regression_metrics(y_true, y_pred)

            rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    **metrics,
                }
            )

        # 2. Evaluate forecasting baselines.
        baseline_predictions = {
            "Seasonal Naive": predict_seasonal_naive(
                sets,
                s.product_id,
            ),
            "Moving Average": predict_moving_average(
                sets,
                s.product_id,
                window=3,
            ),
        }

        for model_name, predictions in baseline_predictions.items():
            y_pred = predictions["validasi"].dropna()

            if y_pred.empty:
                continue

            y_true = cast(
                pd.Series,
                sets["validasi"].loc[y_pred.index, TARGET],
            )

            metrics = regression_metrics(y_true, y_pred)

            rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    **metrics,
                }
            )

        # Calculate the number of models evaluated in this fold.
        fold_rows_evaluated = len(rows) - fold_rows_before

        log.info(
            "Walk-forward fold %d selesai: %d model dievaluasi.",
            fold_number,
            fold_rows_evaluated,
        )

    if not rows:
        raise ValueError("Tidak ada hasil walk-forward evaluation.")

    results = pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)

    log.info(
        "Walk-forward evaluation selesai: %d fold, %d total evaluasi model.",
        len(folds),
        len(results),
    )

    return results


def evaluate_walk_forward_per_product(
    folds: list[dict[str, pd.DataFrame]],
    s: Schema,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """Evaluate every model separately for each product across walk-forward folds."""
    rows: list[dict[str, object]] = []

    for fold_number, fold in enumerate(folds, start=1):
        sets: dict[str, pd.DataFrame] = {
            "train": fold["train"],
            "validasi": fold["validasi"],
            "test": fold["validasi"].iloc[0:0].copy(),
        }

        # Evaluate regression models.
        model_predictions = {
            model_name: predict_per_product(
                make_model,
                sets["train"],
                sets,
                s.product_id,
            )["validasi"]
            for model_name, make_model in build_models(cfg).items()
        }

        # Evaluate forecasting baselines.
        baseline_predictions = {
            "Seasonal Naive": predict_seasonal_naive(
                sets,
                s.product_id,
            )["validasi"],
            "Moving Average": predict_moving_average(
                sets,
                s.product_id,
                window=3,
            )["validasi"],
        }

        all_predictions = {
            **model_predictions,
            **baseline_predictions,
        }

        # Calculate metrics independently for each product.
        for pid, product_data in sets["validasi"].groupby(s.product_id):
            product_data = product_data.sort_values("t")

            for model_name, predictions in all_predictions.items():
                y_pred = cast(
                    pd.Series,
                    predictions.reindex(product_data.index).dropna(),
                )

                if y_pred.empty:
                    continue

                y_true_aligned = cast(
                    pd.Series,
                    product_data.loc[y_pred.index, TARGET],
                )

                mse = mean_squared_error(y_true_aligned, y_pred)

                metrics = {
                    "MAE": mean_absolute_error(y_true_aligned, y_pred),
                    "RMSE": np.sqrt(mse),
                    "WAPE": calculate_wape(y_true_aligned, y_pred),
                }

                rows.append(
                    {
                        "fold": fold_number,
                        s.product_id: pid,
                        s.product_name: product_data[s.product_name].iloc[0],
                        s.category: product_data[s.category].iloc[0],
                        "model": model_name,
                        **metrics,
                    }
                )

        log.info(
            "Per-product walk-forward fold %d selesai.",
            fold_number,
        )

    if not rows:
        raise ValueError("Tidak ada hasil per-product walk-forward evaluation.")

    return (
        pd.DataFrame(rows)
        .sort_values(["fold", s.product_id, "model"])
        .reset_index(drop=True)
    )


def select_models_per_product(
    walk_forward_product_metrics: pd.DataFrame,
    s: Schema,
) -> pd.DataFrame:
    """Select the lowest mean-RMSE model for each product across all folds."""
    required_columns = {
        s.product_id,
        s.product_name,
        s.category,
        "model",
        "fold",
        "RMSE",
    }

    missing_columns = required_columns.difference(walk_forward_product_metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom walk-forward tidak ditemukan: {sorted(missing_columns)}"
        )

    summary = walk_forward_product_metrics.groupby(
        [s.product_id, "model"],
        as_index=False,
    ).agg(
        mean_RMSE=("RMSE", "mean"),
        folds=("fold", "nunique"),
    )

    # Every product must have results from all three folds.
    expected_folds = walk_forward_product_metrics["fold"].nunique()

    summary = summary.loc[summary["folds"] == expected_folds].copy()

    if summary.empty:
        raise ValueError("Tidak ada model dengan evaluasi lengkap.")

    # Select the model with the lowest mean RMSE for each product.
    selected = (
        summary.sort_values(
            [s.product_id, "mean_RMSE", "model"],
        )
        .groupby(s.product_id, as_index=False)
        .first()
    )

    # Attach product metadata.
    products = walk_forward_product_metrics[
        [s.product_id, s.product_name, s.category]
    ].drop_duplicates(subset=[s.product_id])

    selected = selected.merge(
        products,
        on=s.product_id,
        how="left",
        validate="one_to_one",
    )

    return (
        selected[
            [
                s.product_id,
                s.product_name,
                s.category,
                "model",
                "mean_RMSE",
                "folds",
            ]
        ]
        .sort_values(s.product_id)
        .reset_index(drop=True)
    )


def _predict_product_model(
    model_name: str,
    history: pd.DataFrame,
    future: pd.DataFrame,
    model_factories: dict,
) -> np.ndarray:
    """Generate a fixed-origin forecast for one product."""
    history = history.sort_values("t")
    future = future.sort_values("t")

    if model_name == "Seasonal Naive":
        historical_sales = history.set_index("t")[TARGET]

        predictions = historical_sales.reindex(future["t"] - 12).to_numpy(dtype=float)

        if np.isnan(predictions).any():
            raise ValueError("Historical seasonal values unavailable.")

    elif model_name == "Moving Average":
        window = 3
        values = history[TARGET].astype(float).tolist()

        if len(values) < window:
            raise ValueError("Insufficient history for moving average.")

        predictions = []

        for _ in range(len(future)):
            prediction = max(
                0.0,
                float(np.mean(values[-window:])),
            )

            predictions.append(prediction)
            values.append(prediction)

        predictions = np.array(predictions)

    else:
        if model_name not in model_factories:
            raise ValueError(f"Unknown model: {model_name}")

        model = model_factories[model_name]()

        model.fit(
            history[FEATURE],
            history[TARGET],
        )

        predictions = model.predict(future[FEATURE])

    return np.maximum(predictions, 0.0)


def evaluate_final_test(
    monthly: pd.DataFrame,
    walk_forward_metrics: pd.DataFrame,
    walk_forward_product_metrics: pd.DataFrame,
    s: Schema,
    cfg: ModelConfig,
    test_months: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Select models using walk-forward validation, retrain on all development
    data, and evaluate once on the untouched final test period.
    """
    if monthly.empty:
        raise ValueError("Dataset monthly tidak boleh kosong.")

    if test_months < 1:
        raise ValueError("Test period harus minimal 1 bulan.")

    months = pd.DatetimeIndex(pd.to_datetime(monthly["Month"].unique())).sort_values()

    if len(months) <= test_months:
        raise ValueError("Insufficient months for final test evaluation.")

    development_months = months[:-test_months]
    final_test_months = months[-test_months:]

    development = monthly.loc[monthly["Month"].isin(development_months)].copy()

    test = monthly.loc[monthly["Month"].isin(final_test_months)].copy()

    if development.empty or test.empty:
        raise ValueError("Development atau final test dataset kosong.")

    log.info(
        "Final evaluation: Development %s .. %s (%d bulan), "
        "Test %s .. %s (%d bulan).",
        development_months[0].strftime("%Y-%m"),
        development_months[-1].strftime("%Y-%m"),
        len(development_months),
        final_test_months[0].strftime("%Y-%m"),
        final_test_months[-1].strftime("%Y-%m"),
        len(final_test_months),
    )

    # 1. Select the best model independently for each product.
    selected_models = select_models_per_product(
        walk_forward_product_metrics,
        s,
    )

    # 2. Select the best overall ML model using walk-forward validation.
    excluded_models = [
        NAIVE,
        "Seasonal Naive",
        "Moving Average",
    ]

    ml_candidates = walk_forward_metrics.loc[
        ~walk_forward_metrics["model"].isin(excluded_models)
    ]

    best_ml = ml_candidates.groupby("model")["RMSE"].mean().dropna().sort_values()

    if best_ml.empty:
        raise ValueError("Tidak ada model ML yang valid.")

    best_ml_model_name = str(best_ml.index[0])

    log.info("Best overall ML model: %s", best_ml_model_name)

    model_factories = build_models(cfg)

    # 3. Define the three strategies to compare.
    strategies = {
        "Global Naive": NAIVE,
        "Best Global ML": best_ml_model_name,
        "Per-Product Selection": None,
    }

    selected_model_map = selected_models.set_index(s.product_id)["model"].to_dict()

    prediction_rows: list[dict[str, object]] = []

    # 4. Generate fixed-origin forecasts.
    for strategy, global_model_name in strategies.items():
        for pid, history in development.groupby(s.product_id):
            history = history.sort_values("t")

            future = cast(
                pd.DataFrame,
                test.loc[test[s.product_id] == pid].sort_values("t"),
            )

            if future.empty:
                continue

            if strategy == "Per-Product Selection":
                model_name = selected_model_map.get(pid)

                if model_name is None:
                    raise ValueError(f"No selected model for product {pid}.")
            else:
                model_name = global_model_name

            predictions = _predict_product_model(
                model_name=model_name,
                history=history,
                future=future,
                model_factories=model_factories,
            )

            for (_, actual), prediction in zip(
                future.iterrows(),
                predictions,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "strategy": strategy,
                        s.product_id: pid,
                        s.product_name: actual[s.product_name],
                        s.category: actual[s.category],
                        "Month": actual["Month"],
                        "model": model_name,
                        "ActualSales": float(actual[TARGET]),
                        "PredictedSales": float(prediction),
                    }
                )

    predictions_df = pd.DataFrame(prediction_rows)

    if predictions_df.empty:
        raise ValueError("Final test tidak menghasilkan prediksi.")

    # 5. Calculate overall metrics for each strategy.
    overall_rows = []

    for strategy, group in predictions_df.groupby("strategy"):
        metrics = regression_metrics(
            group["ActualSales"],
            group["PredictedSales"],
        )

        overall_rows.append(
            {
                "strategy": strategy,
                "models_used": (group["model"].nunique()),
                **metrics,
            }
        )

    overall_metrics = (
        pd.DataFrame(overall_rows).sort_values("RMSE").reset_index(drop=True)
    )

    # 6. Calculate metrics independently for each product.
    product_rows = []

    for (strategy, pid), group in predictions_df.groupby(["strategy", s.product_id]):
        metrics = regression_metrics(
            group["ActualSales"],
            group["PredictedSales"],
        )

        product_rows.append(
            {
                "strategy": strategy,
                s.product_id: pid,
                s.product_name: group[s.product_name].iloc[0],
                "model": group["model"].iloc[0],
                **metrics,
            }
        )

    product_metrics = (
        pd.DataFrame(product_rows)
        .sort_values(["strategy", s.product_id])
        .reset_index(drop=True)
    )

    log.info(
        "Final test evaluation completed: %d predictions, %d products.",
        len(predictions_df),
        predictions_df[s.product_id].nunique(),
    )

    return {
        "final_selected_models": selected_models,
        "final_test_predictions": predictions_df,
        "final_test_metrics": overall_metrics,
        "final_test_product_metrics": product_metrics,
    }


def run_modeling(
    df: pd.DataFrame,
    s: Schema,
    cfg: ModelConfig,
) -> dict[str, pd.DataFrame]:
    """Run the complete sales modeling pipeline."""

    # 1. Build monthly sales dataset.
    monthly = build_monthly(df, s, cfg)

    # 2. Generate chronological train/validation/test splits.
    months = pd.DatetimeIndex(pd.to_datetime(monthly["Month"].unique())).sort_values()

    month_splits = split_months(months, cfg)

    sets: dict[str, pd.DataFrame] = {
        name: monthly.loc[monthly["Month"].isin(split_dates.tolist())].copy()
        for name, split_dates in month_splits.items()
    }

    for name, data in sets.items():
        split_dates = month_splits[name]

        log.info(
            "%s %3d bulan (%s .. %s), %d baris",
            name,
            len(split_dates),
            split_dates[0].strftime("%Y-%m"),
            split_dates[-1].strftime("%Y-%m"),
            len(data),
        )

    # 3. Generate expanding-window walk-forward folds.
    folds = build_walk_forward_folds(
        monthly=monthly,
        initial_train_months=12,
        validation_months=3,
        step_months=3,
        test_months=3,
    )

    for index, fold in enumerate(folds, start=1):
        train_months = pd.DatetimeIndex(
            pd.to_datetime(fold["train"]["Month"].unique())
        ).sort_values()

        valid_months = pd.DatetimeIndex(
            pd.to_datetime(fold["validasi"]["Month"].unique())
        ).sort_values()

        log.info(
            "Fold %d: Train %s .. %s (%d bulan), " "Validation %s .. %s (%d bulan)",
            index,
            train_months[0].strftime("%Y-%m"),
            train_months[-1].strftime("%Y-%m"),
            len(train_months),
            valid_months[0].strftime("%Y-%m"),
            valid_months[-1].strftime("%Y-%m"),
            len(valid_months),
        )

    # 4. Evaluate models using walk-forward validation.
    walk_forward_metrics = evaluate_walk_forward(
        folds=folds,
        s=s,
        cfg=cfg,
    )

    walk_forward_product_metrics = evaluate_walk_forward_per_product(
        folds=folds,
        s=s,
        cfg=cfg,
    )

    # 6. Select models using walk-forward validation
    # and evaluate them on the untouched final test period.
    final_evaluation = evaluate_final_test(
        monthly=monthly,
        walk_forward_metrics=walk_forward_metrics,
        walk_forward_product_metrics=walk_forward_product_metrics,
        s=s,
        cfg=cfg,
        test_months=3,
    )

    log.info(
        "Best models selected for %d products.",
        len(final_evaluation["final_selected_models"]),
    )

    log.info(
        "Final test results:\n%s",
        final_evaluation["final_test_metrics"].to_string(index=False),
    )

    # 7. Aggregate walk-forward evaluation results.
    walk_forward_summary = (
        walk_forward_metrics.groupby("model")
        .agg(
            folds=("fold", "count"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_WAPE=("WAPE", "mean"),
            std_WAPE=("WAPE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
        )
        .reset_index()
        .sort_values("mean_RMSE")
        .reset_index(drop=True)
    )

    if walk_forward_summary.empty:
        raise ValueError("Walk-forward summary tidak menghasilkan evaluasi.")

    # 8. Identify the best model using average walk-forward RMSE.
    valid_walk_forward = walk_forward_summary.dropna(subset=["mean_RMSE"])

    if valid_walk_forward.empty:
        raise ValueError("Tidak ada mean walk-forward RMSE yang valid.")

    best_walk_forward = str(valid_walk_forward.iloc[0]["model"])

    # Exclude forecasting baselines when selecting the best ML model.
    ml_candidates = valid_walk_forward.loc[
        ~valid_walk_forward["model"].isin([NAIVE, "Seasonal Naive", "Moving Average"])
    ]

    if ml_candidates.empty:
        raise ValueError("Tidak ada model ML yang valid dalam walk-forward evaluation.")

    best_walk_forward_ml = str(ml_candidates.iloc[0]["model"])

    log.info(
        "Model terbaik menurut Walk-Forward RMSE: %s",
        best_walk_forward,
    )

    log.info(
        "Model ML terbaik menurut Walk-Forward RMSE: %s",
        best_walk_forward_ml,
    )

    # 9. Evaluate models using the original chronological split.
    results = compare_models(
        sets=sets,
        s=s,
        cfg=cfg,
    )

    product_metrics = compare_models_per_product(
        sets=sets,
        s=s,
        cfg=cfg,
    )

    # 10. Analyze product sales trends.
    product_trend = product_trends(
        monthly=monthly,
        train_months=month_splits["train"],
        s=s,
    )

    # 11. Generate future forecasts using the best model per product.
    future_forecast = forecast_future(
        monthly=monthly,
        product_metrics=product_metrics,
        s=s,
        cfg=cfg,
        horizon=3,
    )

    log.info(
        "Berhasil menghasilkan %d prediksi untuk %d produk.",
        len(future_forecast),
        future_forecast[s.product_id].nunique(),
    )

    # 12. Identify the best models using the original validation set.
    overall_best = best_model(results)
    ml_best = best_ml_model(results)

    log.info(
        "Model terbaik keseluruhan menurut original validation: %s",
        overall_best,
    )

    log.info(
        "Model ML terbaik menurut original validation: %s",
        ml_best,
    )

    # 13. Return all modeling artifacts.
    return {
        "monthly": monthly,
        "train": sets["train"],
        "validasi": sets["validasi"],
        "test": sets["test"],
        "model_metrics": results,
        "product_model_metrics": product_metrics,
        "product_trend": product_trend,
        "future_forecast": future_forecast,
        "walk_forward_metrics": walk_forward_metrics,
        "walk_forward_summary": walk_forward_summary,
        "walk_forward_product_metrics": walk_forward_product_metrics,
        "final_selected_models": final_evaluation["final_selected_models"],
        "final_test_predictions": final_evaluation["final_test_predictions"],
        "final_test_metrics": final_evaluation["final_test_metrics"],
        "final_test_product_metrics": final_evaluation["final_test_product_metrics"],
    }
