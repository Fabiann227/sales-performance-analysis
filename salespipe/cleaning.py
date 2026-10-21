"""Cleaning + merge tabel raw (orders, customers, payments, products).

Aturan mengikuti materi Cleaning.pdf:
- duplikat penuh dihapus
- kolom yang menentukan waktu/nominal (tanggal, quantity, discount) tidak diimputasi:
  baris missing dihapus (missing kecil & acak)
- numerik non-nominal -> median; kategorikal -> "Unknown"
- tipe data tanggal & angka diperbaiki
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Schema

log = logging.getLogger(__name__)

UNKNOWN = "Unknown"


def read_table(path: Path) -> pd.DataFrame:
    readers = {".csv": pd.read_csv, ".xlsx": pd.read_excel, ".parquet": pd.read_parquet}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Format file tidak didukung: {path.suffix} (didukung: {sorted(readers)})")
    return reader(path)


def required_columns(s: Schema) -> dict[str, list[str]]:
    return {
        "orders": [s.order_id, s.customer_id, s.product_id, s.order_date, s.quantity,
                   s.discount, s.payment_method, s.status],
        "customers": [s.customer_id, s.age, s.city, s.signup_date],
        "payments": [s.order_id, s.payment_status],
        "products": [s.product_id, s.product_name, s.category, s.unit_price],
    }


def load_tables(data_dir: Path, s: Schema) -> dict[str, pd.DataFrame]:
    files = {"orders": s.orders_file, "customers": s.customers_file,
             "payments": s.payments_file, "products": s.products_file}
    tables = {}
    for name, filename in files.items():
        path = Path(data_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"File '{filename}' untuk tabel {name} tidak ditemukan di {data_dir}")
        tables[name] = read_table(path)
    _validate_columns(tables, s)
    return tables


def _validate_columns(tables: dict[str, pd.DataFrame], s: Schema) -> None:
    for name, cols in required_columns(s).items():
        missing = [c for c in cols if c not in tables[name].columns]
        if missing:
            raise ValueError(f"Tabel {name}: kolom {missing} tidak ada. "
                             f"Sesuaikan lewat --config. Kolom tersedia: {list(tables[name].columns)}")


def _report_missing(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        miss = df.isnull().sum()
        log.info("RAW %-9s %s missing: %s", name, df.shape, miss[miss > 0].to_dict() or "-")


def clean_customers(df: pd.DataFrame, s: Schema) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df[s.age] = df[s.age].fillna(df[s.age].median()).astype(int)  # numerik -> median
    df[s.city] = (df[s.city].str.strip().str.title().replace(s.city_aliases)
                  .fillna(UNKNOWN))  # kategorikal -> Unknown (modus akan membiaskan)
    df[s.signup_date] = pd.to_datetime(df[s.signup_date], errors="coerce")
    return df


def clean_orders(df: pd.DataFrame, s: Schema) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates().copy()
    log.info("orders: duplikat dihapus %d", n - len(df))

    df[s.order_date] = pd.to_datetime(df[s.order_date], errors="coerce")  # format tak valid -> NaT

    # kolom penentu waktu/nominal: tidak diimputasi
    n = len(df)
    df = df.dropna(subset=[s.order_date, s.quantity, s.discount])
    log.info("orders: baris missing/invalid tanggal-quantity-discount dihapus %d", n - len(df))

    n = len(df)
    df = df[df[s.quantity] > 0]
    log.info("orders: baris quantity <= 0 dihapus %d", n - len(df))

    df[s.payment_method] = df[s.payment_method].fillna(UNKNOWN)
    df[s.quantity] = df[s.quantity].astype(int)
    return df


def merge_tables(orders, payments, products, customers, s: Schema) -> pd.DataFrame:
    """validate= mencegah baris menggandakan diam-diam bila kunci tidak unik."""
    payments = payments[[s.order_id, s.payment_status]]  # tanggal bayar redundan dengan order_date
    df = (orders
          .merge(payments, on=s.order_id, how="left", validate="one_to_one")
          .merge(products, on=s.product_id, how="left", validate="many_to_one"))
    n = len(df)
    df = df.merge(customers, on=s.customer_id, how="inner", validate="many_to_one")
    log.info("merge: baris dengan customer tidak dikenal (orphan) dihapus %d", n - len(df))
    return df


def add_sales(df: pd.DataFrame, s: Schema) -> pd.DataFrame:
    df[s.sales] = (df[s.quantity] * df[s.unit_price] * (1 - df[s.discount])).round(2)
    return df


def clean_and_merge(data_dir: Path, s: Schema) -> pd.DataFrame:
    tables = load_tables(data_dir, s)
    _report_missing(tables)

    df = merge_tables(
        clean_orders(tables["orders"], s),
        tables["payments"],
        tables["products"],
        clean_customers(tables["customers"], s),
        s,
    )
    df = add_sales(df, s).sort_values(s.order_id).reset_index(drop=True)

    remaining = df.isnull().sum()
    if remaining.any():
        raise ValueError(f"Masih ada missing value setelah cleaning: {remaining[remaining > 0].to_dict()}")
    log.info("hasil cleaning: %s", df.shape)
    return df
