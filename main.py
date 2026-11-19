"""Entry point for the modular sales analysis pipeline."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from salespipe.cleaning import clean_and_merge
from salespipe.config import load_config
from salespipe.insights import generate_insights
from salespipe.modeling import run_modeling
from salespipe.reporting import generate_report
from salespipe.visualization import generate_visualizations

log = logging.getLogger(__name__)

CLEAN_FILE = "clean_merged.csv"


def cmd_clean(args: argparse.Namespace, cfg) -> None:
    """Clean and merge the raw datasets."""
    df = clean_and_merge(args.data_dir, cfg.schema, cfg.period)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.out_dir / CLEAN_FILE
    df.to_csv(output_path, index=False)

    log.info("Cleaned dataset saved to: %s", output_path)


def save_artifacts(
    artifacts: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save all modeling artifacts as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in artifacts.items():
        output_table = table.copy()

        numeric_columns = output_table.select_dtypes(include="number").columns

        output_table[numeric_columns] = output_table[numeric_columns].round(4)

        output_path = output_dir / f"{name}.csv"
        output_table.to_csv(output_path, index=False)

        log.info("Saved: %s", output_path)


def print_model_metrics(artifacts: dict[str, pd.DataFrame]) -> None:
    """Print the original train, validation, and test model comparison."""
    metrics = artifacts["model_metrics"]

    columns = [
        "model",
        "train_RMSE",
        "validation_RMSE",
        "test_MAE",
        "test_RMSE",
        "test_R2",
        "train_WAPE",
        "validation_WAPE",
        "test_WAPE",
    ]

    print("\n=== Model Comparison ===")
    print(metrics[columns].round(3).to_string(index=False))


def print_walk_forward_summary(
    artifacts: dict[str, pd.DataFrame],
) -> None:
    """Print the aggregated walk-forward evaluation results."""
    summary = artifacts["walk_forward_summary"]

    columns = [
        "model",
        "folds",
        "mean_MAE",
        "std_MAE",
        "mean_RMSE",
        "std_RMSE",
        "mean_WAPE",
        "std_WAPE",
        "mean_R2",
        "std_R2",
    ]

    print("\n=== Walk-Forward Validation Summary ===")
    print(summary[columns].round(3).to_string(index=False))


def print_product_trends(artifacts: dict[str, pd.DataFrame]) -> None:
    """Print the sales trend analysis for each product."""
    print("\n=== Product Sales Trends ===")
    print(artifacts["product_trend"].round(2).to_string(index=False))


def cmd_model(args: argparse.Namespace, cfg) -> None:
    """Run the modeling pipeline and export its results."""
    input_path = args.out_dir / CLEAN_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} does not exist. Run 'python main.py clean' first."
        )

    df = pd.read_csv(
        input_path,
        parse_dates=[
            cfg.schema.order_date,
            cfg.schema.signup_date,
        ],
    )

    log.info("Starting modeling pipeline...")

    artifacts = run_modeling(
        df=df,
        s=cfg.schema,
        cfg=cfg.model,
    )

    # Export every artifact returned by run_modeling().
    # This includes walk_forward_metrics.csv and walk_forward_summary.csv.
    save_artifacts(
        artifacts=artifacts,
        output_dir=args.out_dir,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print_model_metrics(artifacts)
    print_walk_forward_summary(artifacts)
    print_product_trends(artifacts)

    log.info(
        "\nModeling complete. All results saved to: %s",
        args.out_dir,
    )


def cmd_visualize(args: argparse.Namespace) -> list[Path]:
    """Generate all eight charts from the modeling artifacts."""
    paths = generate_visualizations(args.out_dir, args.visualizations_dir)
    log.info("Generated %d visualizations in %s", len(paths), args.visualizations_dir)
    return paths


def cmd_insights(args: argparse.Namespace) -> dict:
    """Generate deterministic derived analysis artifacts and findings."""
    insights = generate_insights(args.out_dir)
    log.info("Business insights generated from %s", args.out_dir)
    return insights


def cmd_report(args: argparse.Namespace, insights: dict | None = None) -> Path:
    """Generate the local HTML report."""
    if insights is None:
        insights = cmd_insights(args)
    report_path = generate_report(
        temp_dir=args.out_dir,
        visualization_dir=args.visualizations_dir,
        report_dir=args.report_dir,
        insights=insights,
    )
    log.info("Report generated: %s", report_path)
    return report_path


def run_all(args: argparse.Namespace, cfg) -> Path:
    """Run the complete workflow in dependency order."""
    print("[1/5] Data Preparation ...............", end=" ", flush=True)
    cmd_clean(args, cfg)
    print("DONE")

    print("[2/5] Model Training & Evaluation ...", end=" ", flush=True)
    cmd_model(args, cfg)
    print("DONE")

    print("[3/5] Visualization ..................", end=" ", flush=True)
    cmd_visualize(args)
    print("DONE")

    print("[4/5] Business Insights ..............", end=" ", flush=True)
    insights = cmd_insights(args)
    print("DONE")

    print("[5/5] Report Generation ..............", end=" ", flush=True)
    report_path = cmd_report(args, insights)
    print("DONE")
    print("\nPipeline completed successfully.")
    print(f"\nReport: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "command",
        choices=["clean", "model", "visualize", "insights", "report", "all"],
        help="Pipeline command to execute.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets"),
        help="Folder containing raw datasets.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temp"),
        help="Folder for generated CSV files.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON configuration override.",
    )

    parser.add_argument(
        "--visualizations-dir",
        type=Path,
        default=Path("visualizations"),
        help="Folder for generated PNG charts.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/latest"),
        help="Folder for the generated HTML report.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    cfg = load_config(args.config)

    if args.command == "all":
        run_all(args, cfg)
        return

    if args.command == "clean":
        cmd_clean(args, cfg)

    if args.command == "model":
        cmd_model(args, cfg)

    if args.command == "visualize":
        cmd_visualize(args)

    if args.command == "insights":
        cmd_insights(args)

    if args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
