"""Entry point: python main.py {clean,model,all} [--data-dir DIR] [--out-dir DIR] [--config FILE]"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from salespipe.cleaning import clean_and_merge
from salespipe.config import load_config
from salespipe.modeling import run_modeling

CLEAN_FILE = "clean_merged.csv"


def cmd_clean(args, cfg) -> None:
    df = clean_and_merge(args.data_dir, cfg.schema)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / CLEAN_FILE, index=False)
    logging.info("tersimpan: %s", args.out_dir / CLEAN_FILE)


def cmd_model(args, cfg) -> None:
    path = args.out_dir / CLEAN_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} belum ada. Jalankan 'clean' dulu.")
    df = pd.read_csv(path, parse_dates=[cfg.schema.order_date, cfg.schema.signup_date])

    artifacts = run_modeling(df, cfg.schema, cfg.model)
    for name, table in artifacts.items():
        table.round(4).to_csv(args.out_dir / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    metrics = artifacts["model_metrics"]
    cols = ["model", "train_RMSE", "validasi_RMSE", "test_MAE", "test_RMSE", "test_R2"]
    print("\n=== Perbandingan model ===")
    print(metrics[cols].round(3).to_string(index=False))
    print("\n=== Tren per produk ===")
    print(artifacts["product_trend"].round(2).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["clean", "model", "all"])
    p.add_argument("--data-dir", type=Path, default=Path("datasets"), help="folder dataset raw")
    p.add_argument("--out-dir", type=Path, default=Path("temp"), help="folder output")
    p.add_argument("--config", type=Path, default=None, help="JSON override nama file/kolom/parameter model")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.command in ("clean", "all"):
        cmd_clean(args, cfg)
    if args.command in ("model", "all"):
        cmd_model(args, cfg)


if __name__ == "__main__":
    main()
