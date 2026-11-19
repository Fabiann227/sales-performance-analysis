"""Generate the project's eight analytical visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def _read(directory: Path, filename: str, dates: list[str] | None = None) -> pd.DataFrame:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Required visualization artifact not found: {path}")
    return cast(pd.DataFrame, pd.read_csv(path, parse_dates=dates or []))


def _save(fig: Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overall_monthly_sales(df: pd.DataFrame, output_dir: Path) -> Path:
    monthly = cast(pd.DataFrame, df.groupby("Month")["Sales"].sum().reset_index()).sort_values(by="Month")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly["Month"], monthly["Sales"], marker="o", linewidth=2)
    ax.set(title="Overall Monthly Sales (2024–2025)", xlabel="Month", ylabel="Total Sales")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, output_dir, "overall_monthly_sales.png")


def plot_product_monthly_sales(df: pd.DataFrame, output_dir: Path) -> Path:
    products = sorted(df["ProductName"].unique())
    rows = max(1, (len(products) + 4) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 3.5 * rows), sharex=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, product in zip(axes_flat, products):
        product_data = df.loc[df["ProductName"] == product].sort_values("Month")
        ax.plot(product_data["Month"], product_data["Sales"], marker="o", linewidth=1.5)
        ax.set_title(product, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes_flat[len(products) :]:
        ax.set_visible(False)
    fig.suptitle("Monthly Sales Trends by Product (2024–2025)", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "product_monthly_sales.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_product_sales_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    product_sales = cast(pd.DataFrame, df.groupby("ProductName")["Sales"].sum().reset_index()).sort_values(by="Sales")
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(product_sales["ProductName"], product_sales["Sales"])
    ax.set(title="Total Sales by Product (2024–2025)", xlabel="Total Sales", ylabel="Product")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in product_sales["Sales"]], padding=4, fontsize=8)
    ax.margins(x=0.15)
    return _save(fig, output_dir, "product_sales_comparison.png")


def plot_model_performance_comparison(temp_dir: Path, output_dir: Path) -> Path:
    df = _read(temp_dir, "walk_forward_summary.csv").sort_values("mean_RMSE")
    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(df["model"], df["mean_RMSE"], xerr=df["std_RMSE"], capsize=4)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set(title="Walk-Forward Validation: Model Performance Comparison", xlabel="Mean RMSE (logarithmic scale)", ylabel="Forecasting Model")
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, df["mean_RMSE"], strict=True):
        ax.annotate(f"{value:,.2f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(12, 0), textcoords="offset points", va="center", fontsize=9)
    ax.margins(x=0.4)
    return _save(fig, output_dir, "model_performance_comparison.png")


def plot_actual_vs_predicted_sales(temp_dir: Path, output_dir: Path) -> Path:
    df = _read(temp_dir, "final_test_predictions.csv", ["Month"])
    actual = df.drop_duplicates(["ProductID", "Month"]).groupby("Month", as_index=False)["ActualSales"].sum()
    predicted = df.groupby(["strategy", "Month"], as_index=False)["PredictedSales"].sum()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(actual["Month"], actual["ActualSales"], marker="o", linewidth=2.5, label="Actual Sales")
    for strategy, group in predicted.groupby("strategy"):
        ax.plot(group["Month"], group["PredictedSales"], marker="o", linestyle="--", label=strategy)
    ax.set(title="Actual vs Predicted Sales (October–December 2025)", xlabel="Month", ylabel="Total Sales")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    return _save(fig, output_dir, "actual_vs_predicted_sales.png")


def plot_forecasting_error_contribution(temp_dir: Path, output_dir: Path) -> Path:
    df = _read(temp_dir, "final_test_error_analysis.csv")
    errors = df.loc[
        df["strategy"] == "Global Training Mean Baseline"
    ].nlargest(10, "ErrorContribution").sort_values("ErrorContribution")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(errors["ProductName"], errors["ErrorContribution"])
    ax.set(title="Top 10 Products Contributing to Forecasting Error (Global Training Mean Baseline)", xlabel="Contribution to Total Absolute Error (%)", ylabel="Product")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in errors["ErrorContribution"]], padding=4, fontsize=9)
    ax.margins(x=0.15)
    return _save(fig, output_dir, "forecasting_error_contribution.png")


def plot_overall_future_forecast(temp_dir: Path, output_dir: Path) -> Path:
    df = _read(temp_dir, "future_forecast.csv", ["Month"])
    monthly = df.groupby("Month", as_index=False)["PredictedSales"].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly["Month"], monthly["PredictedSales"], marker="o", linewidth=2.5)
    ax.set(title="Overall Future Sales Forecast (January–March 2026)", xlabel="Month", ylabel="Predicted Total Sales")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, output_dir, "overall_future_forecast.png")


def plot_future_forecast_by_product(temp_dir: Path, output_dir: Path) -> Path:
    df = _read(temp_dir, "future_forecast.csv", ["Month"])
    products = sorted(df["ProductName"].unique())
    rows = max(1, (len(products) + 4) // 5)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 3.5 * rows), sharex=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, product in zip(axes_flat, products):
        data = df.loc[df["ProductName"] == product].sort_values("Month")
        ax.plot(data["Month"], data["PredictedSales"], marker="o", linewidth=1.8)
        ax.set_title(f"{product}\nModel: {data['model'].iloc[0]}", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    for ax in axes_flat[len(products) :]:
        ax.set_visible(False)
    fig.suptitle("Future Sales Forecast by Product (January–March 2026)", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = output_dir / "future_forecast_by_product.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_visualizations(temp_dir: Path, output_dir: Path) -> list[Path]:
    """Generate and return paths to all eight charts."""
    monthly = _read(temp_dir, "monthly.csv", ["Month"])
    return [
        plot_overall_monthly_sales(monthly, output_dir),
        plot_product_monthly_sales(monthly, output_dir),
        plot_product_sales_comparison(monthly, output_dir),
        plot_model_performance_comparison(temp_dir, output_dir),
        plot_actual_vs_predicted_sales(temp_dir, output_dir),
        plot_forecasting_error_contribution(temp_dir, output_dir),
        plot_overall_future_forecast(temp_dir, output_dir),
        plot_future_forecast_by_product(temp_dir, output_dir),
    ]
