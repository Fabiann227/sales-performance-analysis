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


def regression_metrics(y, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y, y_pred)
    return {
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y, y_pred),
    }


def compare_models(
    sets: dict[str, pd.DataFrame], s: Schema, cfg: ModelConfig
) -> pd.DataFrame:
    rows = []
    for name, make_model in build_models(cfg).items():
        preds = predict_per_product(make_model, sets["train"], sets, s.product_id)
        row = {"model": name}
        for k in SET_NAMES:
            row.update(
                {
                    f"{k}_{m}": v
                    for m, v in regression_metrics(sets[k][TARGET], preds[k]).items()
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def best_model(results: pd.DataFrame) -> str:
    """Select the best ML model using validation RMSE."""
    candidates = cast(
        pd.DataFrame,
        results.loc[results["model"] != NAIVE],
    )

    if candidates.empty:
        raise ValueError("Tidak ada model ML yang dapat dipilih.")

    validation_rmse = cast(
        pd.Series,
        candidates["validasi_RMSE"],
    )

    best_idx = validation_rmse.idxmin()

    return str(candidates.at[best_idx, "model"])


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
    log.info("model terbaik menurut validasi: %s", best_model(results))

    return {
        **{f"{k}_monthly": d for k, d in sets.items()},
        "model_metrics": results,
        "product_trend": product_trends(monthly, month_sets["train"], s),
    }
