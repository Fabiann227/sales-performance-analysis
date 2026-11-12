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
    make_model, train: pd.DataFrame, sets: dict[str, pd.DataFrame], product_col: str
):
    """Satu model per produk: tiap produk punya pola tren sendiri."""
    preds = {k: pd.Series(index=d.index, dtype=float) for k, d in sets.items()}
    for pid, g in train.groupby(product_col):
        model = make_model().fit(g[FEATURE], g[TARGET])
        for k, d in sets.items():
            rows = d[d[product_col] == pid]
            preds[k].loc[rows.index] = model.predict(rows[FEATURE])
    return preds


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
        results.loc[~results["model"].isin([NAIVE, "Seasonal Naive"])],
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


def run_modeling(
    df: pd.DataFrame, s: Schema, cfg: ModelConfig
) -> dict[str, pd.DataFrame]:
    """Kembalikan semua artefak: split, metrik, tren per produk."""
    monthly = build_monthly(df, s, cfg)
    month_sets = split_months(pd.DatetimeIndex(sorted(monthly["Month"].unique())), cfg)
    sets: dict[str, pd.DataFrame] = {}

    month_column = cast(pd.Series, monthly["Month"])

    for name, months in month_sets.items():
        mask = month_column.isin(months.tolist())

        sets[name] = cast(
            pd.DataFrame,
            monthly.loc[mask].copy(),
        )

    for k, ms in month_sets.items():
        log.info(
            "%-9s %2d bulan (%s .. %s), %d baris",
            k,
            len(ms),
            f"{ms[0]:%Y-%m}",
            f"{ms[-1]:%Y-%m}",
            len(sets[k]),
        )

    results = compare_models(sets, s, cfg)
    product_metrics = compare_models_per_product(
        sets,
        s,
        cfg,
    )
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

    log.info(
        "model terbaik keseluruhan menurut validasi: %s",
        best_model(results),
    )

    log.info(
        "model ML terbaik menurut validasi: %s",
        best_ml_model(results),
    )

    return {
        **{f"{k}_monthly": d for k, d in sets.items()},
        "model_metrics": results,
        "product_model_metrics": product_metrics,
        "product_trend": product_trends(monthly, month_sets["train"], s),
        "future_forecast": future_forecast,
    }
