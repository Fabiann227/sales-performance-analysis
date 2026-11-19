"""Deterministic, artifact-based business insights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def _read(temp_dir: Path, filename: str) -> pd.DataFrame:
    path = temp_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required insight artifact not found: {path}")
    return cast(pd.DataFrame, pd.read_csv(path))


def _error_artifacts(temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = _read(temp_dir, "final_test_predictions.csv")
    predictions["Error"] = predictions["ActualSales"] - predictions["PredictedSales"]
    predictions["AbsoluteError"] = predictions["Error"].abs()
    predictions["SquaredError"] = predictions["Error"] ** 2
    grouped = cast(pd.DataFrame, predictions.groupby(["strategy", "ProductID", "ProductName", "Category"], as_index=False).agg(
        TotalAbsoluteError=("AbsoluteError", "sum"),
        MeanAbsoluteError=("AbsoluteError", "mean"),
        RMSE=("SquaredError", lambda values: float(np.sqrt(values.mean()))),
        ActualSales=("ActualSales", "sum"),
    ))
    totals = grouped.groupby("strategy")["TotalAbsoluteError"].transform("sum")
    grouped["ErrorContribution"] = np.where(totals != 0, grouped["TotalAbsoluteError"] / totals * 100, 0.0)
    grouped["RelativeAbsoluteErrorPct"] = np.where(
        grouped["ActualSales"] != 0,
        grouped["TotalAbsoluteError"] / grouped["ActualSales"] * 100,
        np.nan,
    )
    summary = cast(pd.DataFrame, predictions.groupby("strategy", as_index=False).agg(
        TotalAbsoluteError=("AbsoluteError", "sum"),
        MAE=("AbsoluteError", "mean"),
        RMSE=("SquaredError", lambda values: float(np.sqrt(values.mean()))),
        TotalActualSales=("ActualSales", "sum"),
    ))
    summary["WAPE"] = np.where(summary["TotalActualSales"] != 0, summary["TotalAbsoluteError"] / summary["TotalActualSales"] * 100, np.nan)
    grouped.to_csv(temp_dir / "final_test_error_analysis.csv", index=False)
    return grouped, summary


def _forecast_product_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast levels and changes between forecast months."""
    data = forecast.copy()
    data["Month"] = pd.to_datetime(data["Month"])
    data = data.sort_values(["ProductID", "Month"])
    rows: list[dict[str, object]] = []
    for key, group in data.groupby(
        ["ProductID", "ProductName", "model"], sort=False
    ):
        product_id, product_name, model = cast(tuple[object, object, object], key)
        values = group["PredictedSales"].astype(float)
        rows.append(
            {
                "ProductID": product_id,
                "ProductName": product_name,
                "model": model,
                "ForecastTotal": float(values.sum()),
                "FirstForecastSales": float(values.iloc[0]),
                "LastForecastSales": float(values.iloc[-1]),
                "ForecastChange": float(values.iloc[-1] - values.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("ProductID").reset_index(drop=True)


def generate_insights(temp_dir: Path) -> dict[str, Any]:
    """Create derived artifacts and structured findings from generated results."""
    metrics = _read(temp_dir, "final_test_metrics.csv")
    trends = _read(temp_dir, "product_trend.csv")
    forecast = _read(temp_dir, "future_forecast.csv")
    walk_forward = _read(temp_dir, "walk_forward_summary.csv")
    errors, error_summary = _error_artifacts(temp_dir)

    best_strategy = str(metrics.sort_values("RMSE").iloc[0]["strategy"])
    best_walk_forward = str(walk_forward.sort_values("mean_RMSE").iloc[0]["model"])
    increasing = trends.loc[
        trends["slope_train"] > 0, "ProductName"
    ].astype(str).tolist()
    decreasing = trends.loc[
        trends["slope_train"] < 0, "ProductName"
    ].astype(str).tolist()
    top_errors = errors.loc[errors["strategy"] == best_strategy].nlargest(5, "TotalAbsoluteError")["ProductName"].astype(str).tolist()
    forecast["Month"] = pd.to_datetime(forecast["Month"])
    forecast_summary = cast(
        pd.DataFrame,
        forecast.groupby("Month")["PredictedSales"].sum().reset_index(),
    ).sort_values(by="Month")
    forecast_change = float(forecast_summary["PredictedSales"].iloc[-1] - forecast_summary["PredictedSales"].iloc[0]) if len(forecast_summary) > 1 else 0.0
    forecast_products = _forecast_product_summary(forecast)

    return {
        "best_strategy": best_strategy,
        "best_walk_forward_model": best_walk_forward,
        "increasing_products": increasing,
        "decreasing_products": decreasing,
        "top_error_products": top_errors,
        "forecast_change_first_to_last": forecast_change,
        "forecast_total": float(forecast["PredictedSales"].sum()),
        "error_summary": error_summary,
        "error_details": errors,
        "forecast_products": forecast_products,
        "selected_model_distribution": forecast["model"].value_counts().to_dict(),
    }
