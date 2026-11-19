from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

INPUT_FILE = Path("temp/monthly.csv")
OUTPUT_DIR = Path("visualizations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and prepare monthly sales data."""
    df = pd.read_csv(INPUT_FILE)
    df["Month"] = pd.to_datetime(df["Month"])
    return df


def plot_overall_monthly_sales(df: pd.DataFrame) -> None:
    """Plot total monthly sales across all products."""
    monthly_sales = (
        df.groupby("Month", as_index=False)
        .agg({"Sales": "sum"})
        .sort_values(by="Month")
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        "Overall Monthly Sales (2024–2025)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Total Sales", fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "overall_monthly_sales.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_product_monthly_sales(df: pd.DataFrame) -> None:
    """Plot monthly sales trends for each product."""
    products = sorted(df["ProductName"].unique())

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(20, 14),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, product in zip(axes, products, strict=True):
        product_data = df.loc[df["ProductName"] == product].sort_values(by="Month")

        ax.plot(
            product_data["Month"],
            product_data["Sales"],
            marker="o",
            linewidth=1.5,
        )

        ax.set_title(
            product,
            fontsize=10,
            fontweight="bold",
        )

        ax.grid(axis="y", alpha=0.3)

        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=8,
        )

    fig.suptitle(
        "Monthly Sales Trends by Product (2024–2025)",
        fontsize=17,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        OUTPUT_DIR / "product_monthly_sales.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_product_sales_comparison(df: pd.DataFrame) -> None:
    """Compare cumulative sales across all products."""
    product_sales = (
        df.groupby("ProductName", as_index=False)
        .agg({"Sales": "sum"})
        .sort_values(by="Sales", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    bars = ax.barh(
        product_sales["ProductName"],
        product_sales["Sales"],
    )

    ax.set_title(
        "Total Sales by Product (2024–2025)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Total Sales", fontsize=11)
    ax.set_ylabel("Product", fontsize=11)

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    ax.bar_label(
        bars,
        labels=[f"{value:,.0f}" for value in product_sales["Sales"]],
        padding=4,
        fontsize=8,
    )

    ax.margins(x=0.15)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "product_sales_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_model_performance_comparison() -> None:
    """Compare forecasting models using walk-forward mean RMSE."""
    input_file = Path("temp/walk_forward_summary.csv")

    df = pd.read_csv(input_file)

    df = df.sort_values(by="mean_RMSE", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 8))

    bars = ax.barh(
        df["model"],
        df["mean_RMSE"],
        xerr=df["std_RMSE"],
        capsize=4,
        error_kw={"elinewidth": 1},
    )

    ax.invert_yaxis()
    ax.set_xscale("log")

    ax.set_title(
        "Walk-Forward Validation: Model Performance Comparison",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Mean RMSE (logarithmic scale)",
        fontsize=11,
    )
    ax.set_ylabel(
        "Forecasting Model",
        fontsize=11,
    )

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(
        bars,
        df["mean_RMSE"],
        strict=True,
    ):
        ax.annotate(
            f"{value:,.2f}",
            xy=(
                value,
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 2,
            },
        )

    ax.margins(x=0.40)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "model_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_actual_vs_predicted_sales() -> None:
    """Compare actual and predicted sales across forecasting strategies."""
    input_file = Path("temp/final_test_predictions.csv")
    df = pd.read_csv(input_file)

    df["Month"] = pd.to_datetime(df["Month"])

    # Calculate actual sales only once per product and month.
    actual_sales = (
        df.drop_duplicates(subset=["ProductID", "Month"])
        .groupby("Month", as_index=False)
        .agg({"ActualSales": "sum"})
        .sort_values(by="Month")
    )

    # Aggregate predictions separately for each strategy.
    predicted_sales = (
        df.groupby(["strategy", "Month"], as_index=False)
        .agg({"PredictedSales": "sum"})
        .sort_values(by="Month")
    )

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(
        actual_sales["Month"],
        actual_sales["ActualSales"],
        marker="o",
        linewidth=2.5,
        label="Actual Sales",
    )

    for strategy in predicted_sales["strategy"].unique():
        strategy_data = predicted_sales.loc[predicted_sales["strategy"] == strategy]

        ax.plot(
            strategy_data["Month"],
            strategy_data["PredictedSales"],
            marker="o",
            linewidth=2,
            linestyle="--",
            label=strategy,
        )

    ax.set_title(
        "Actual vs Predicted Sales (October–December 2025)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Total Sales", fontsize=11)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "actual_vs_predicted_sales.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_forecasting_error_contribution() -> None:
    """Visualize the top 10 contributors to Global Naive forecasting error."""
    input_file = Path("temp/final_test_error_analysis.csv")

    df = pd.read_csv(input_file)

    naive_errors = df.loc[df["strategy"] == "Global Naive"].copy()

    top_errors = (
        naive_errors.sort_values(
            by="ErrorContribution",
            ascending=False,
        )
        .head(10)
        .sort_values(
            by="ErrorContribution",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        top_errors["ProductName"],
        top_errors["ErrorContribution"],
    )

    ax.set_title(
        "Top 10 Products Contributing to Forecasting Error",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Contribution to Total Absolute Error (%)",
        fontsize=11,
    )
    ax.set_ylabel("Product", fontsize=11)

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in top_errors["ErrorContribution"]],
        padding=4,
        fontsize=9,
    )

    ax.margins(x=0.15)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "forecasting_error_contribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_overall_future_forecast() -> None:
    """Plot aggregate predicted sales for January–March 2026."""
    input_file = Path("temp/future_forecast.csv")

    df = pd.read_csv(input_file)
    df["Month"] = pd.to_datetime(df["Month"])

    monthly_forecast = (
        df.groupby("Month", as_index=False)
        .agg({"PredictedSales": "sum"})
        .sort_values(by="Month")
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        monthly_forecast["Month"],
        monthly_forecast["PredictedSales"],
        marker="o",
        linewidth=2.5,
    )

    ax.set_title(
        "Overall Future Sales Forecast (January–March 2026)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Predicted Total Sales", fontsize=11)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "overall_future_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_future_forecast_by_product() -> None:
    """Plot future sales forecasts and selected models for each product."""
    input_file = Path("temp/future_forecast.csv")

    df = pd.read_csv(input_file)
    df["Month"] = pd.to_datetime(df["Month"])

    products = sorted(df["ProductName"].unique())

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(20, 14),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, product in zip(axes, products, strict=True):
        product_data = df.loc[df["ProductName"] == product].sort_values(by="Month")

        ax.plot(
            product_data["Month"],
            product_data["PredictedSales"],
            marker="o",
            linewidth=1.8,
        )

        selected_model = product_data["model"].iloc[0]

        ax.set_title(
            f"{product}\nModel: {selected_model}",
            fontsize=9,
            fontweight="bold",
        )

        ax.grid(axis="y", alpha=0.3)

        ax.tick_params(
            axis="x",
            rotation=30,
            labelsize=8,
        )

    fig.suptitle(
        "Future Sales Forecast by Product (January–March 2026)",
        fontsize=17,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        OUTPUT_DIR / "future_forecast_by_product.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Generate all sales analysis visualizations."""
    df = load_data()

    plot_overall_monthly_sales(df)
    plot_product_monthly_sales(df)
    plot_product_sales_comparison(df)
    plot_model_performance_comparison()
    plot_actual_vs_predicted_sales()
    plot_forecasting_error_contribution()
    plot_overall_future_forecast()
    plot_future_forecast_by_product()

    print("\nVisualization completed successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Generated 8 visualization files.")


if __name__ == "__main__":
    main()
