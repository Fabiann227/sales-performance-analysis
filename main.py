"""Entry point: python main.py {clean,model,all} [--data-dir DIR] [--out-dir DIR] [--config FILE]"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from salespipe.cleaning import clean_and_merge
from salespipe.config import load_config
from salespipe.modeling import run_modeling

log = logging.getLogger(__name__)

CLEAN_FILE = "clean_merged.csv"


def cmd_clean(args: argparse.Namespace, cfg) -> None:
    """Clean and merge the raw datasets."""
    df = clean_and_merge(args.data_dir, cfg.schema, cfg.period)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.out_dir / CLEAN_FILE
    df.to_csv(output_path, index=False)

    log.info("Dataset temizlendi dan tersimpan: %s", output_path)


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

        log.info("Tersimpan: %s", output_path)


def print_model_metrics(artifacts: dict[str, pd.DataFrame]) -> None:
    """Print the original train, validation, and test model comparison."""
    metrics = artifacts["model_metrics"]

    columns = [
        "model",
        "train_RMSE",
        "validasi_RMSE",
        "test_MAE",
        "test_RMSE",
        "test_R2",
        "train_WAPE",
        "validasi_WAPE",
        "test_WAPE",
    ]

    print("\n=== Perbandingan Model ===")
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
    print("\n=== Tren Penjualan per Produk ===")
    print(artifacts["product_trend"].round(2).to_string(index=False))


def cmd_model(args: argparse.Namespace, cfg) -> None:
    """Run the modeling pipeline and export its results."""
    input_path = args.out_dir / CLEAN_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} belum ada. Jalankan 'python main.py clean' dulu."
        )

    df = pd.read_csv(
        input_path,
        parse_dates=[
            cfg.schema.order_date,
            cfg.schema.signup_date,
        ],
    )

    log.info("Memulai modeling pipeline...")

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
        "\nModeling selesai. Seluruh hasil tersimpan di: %s",
        args.out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "command",
        choices=["clean", "model", "all"],
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    cfg = load_config(args.config)

    if args.command in ("clean", "all"):
        cmd_clean(args, cfg)

    if args.command in ("model", "all"):
        cmd_model(args, cfg)


if __name__ == "__main__":
    main()
