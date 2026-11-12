"""Cleaning and merging raw marketplace tables."""

from __future__ import annotations

import logging
from math import isnan
from pathlib import Path
from typing import cast

import pandas as pd

from .config import PeriodConfig, Schema

log = logging.getLogger(__name__)

UNKNOWN = "Unknown"


def read_table(path: Path) -> pd.DataFrame:
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".parquet": pd.read_parquet,
    }

    reader = readers.get(path.suffix.lower())

    if reader is None:
        raise ValueError(
            f"Format file tidak didukung: {path.suffix} "
            f"(didukung: {sorted(readers)})"
        )

    return reader(path)


def required_columns(s: Schema) -> dict[str, list[str]]:
    return {
        "orders": [
            s.order_id,
            s.customer_id,
            s.product_id,
            s.order_date,
            s.quantity,
            s.discount,
            s.payment_method,
            s.status,
        ],
        "customers": [
            s.customer_id,
            s.age,
            s.city,
            s.signup_date,
        ],
        "payments": [
            s.order_id,
            s.payment_status,
        ],
        "products": [
            s.product_id,
            s.product_name,
            s.category,
            s.unit_price,
        ],
    }


def load_tables(
    data_dir: Path,
    s: Schema,
) -> dict[str, pd.DataFrame]:
    files = {
        "orders": s.orders_file,
        "customers": s.customers_file,
        "payments": s.payments_file,
        "products": s.products_file,
    }

    tables: dict[str, pd.DataFrame] = {}

    for name, filename in files.items():
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"File '{filename}' untuk tabel {name} "
                f"tidak ditemukan di {data_dir}"
            )

        tables[name] = read_table(path)

    _validate_columns(tables, s)

    return tables


def _validate_columns(
    tables: dict[str, pd.DataFrame],
    s: Schema,
) -> None:
    for name, columns in required_columns(s).items():
        missing = [column for column in columns if column not in tables[name].columns]

        if missing:
            raise ValueError(
                f"Tabel {name}: kolom {missing} tidak ada. "
                f"Sesuaikan lewat --config. "
                f"Kolom tersedia: {list(tables[name].columns)}"
            )


def _report_missing(
    tables: dict[str, pd.DataFrame],
) -> None:
    for name, df in tables.items():
        missing: dict[str, int] = {}

        for column in df.columns:
            series = cast(pd.Series, df[column])
            count = int(cast(int, series.isna().sum()))

            if count > 0:
                missing[column] = count

        log.info(
            "RAW %-9s %s missing: %s",
            name,
            df.shape,
            missing or "-",
        )


def clean_customers(
    df: pd.DataFrame,
    s: Schema,
) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    age_column = cast(pd.Series, df[s.age])
    age = cast(
        pd.Series,
        pd.to_numeric(age_column, errors="coerce"),
    )

    age_median = float(age.median())

    if isnan(age_median):
        raise ValueError("Kolom Age tidak memiliki nilai valid untuk imputasi.")

    df[s.age] = age.fillna(age_median).astype("int64")

    city = cast(pd.Series, df[s.city]).astype("string")

    df[s.city] = city.str.strip().str.title().replace(s.city_aliases).fillna(UNKNOWN)

    df[s.signup_date] = pd.to_datetime(
        df[s.signup_date],
        errors="coerce",
    )

    return df


def clean_orders(
    df: pd.DataFrame,
    s: Schema,
) -> pd.DataFrame:
    initial_count = len(df)

    # 1. Remove duplicate records.
    df = df.drop_duplicates().copy()

    log.info(
        "orders: duplikat dihapus %d",
        initial_count - len(df),
    )

    # 2. Convert columns to their expected data types.
    df[s.order_date] = pd.to_datetime(
        df[s.order_date],
        errors="coerce",
    )

    quantity = cast(pd.Series, df[s.quantity])
    discount = cast(pd.Series, df[s.discount])

    df[s.quantity] = pd.to_numeric(
        quantity,
        errors="coerce",
    )

    df[s.discount] = pd.to_numeric(
        discount,
        errors="coerce",
    )

    # 3. Remove rows with missing or invalid critical values.
    previous_count = len(df)

    df = df.dropna(
        subset=[
            s.order_date,
            s.quantity,
            s.discount,
        ]
    )

    log.info(
        "orders: missing/invalid values removed %d",
        previous_count - len(df),
    )

    # 4. Validate quantity and discount.
    previous_count = len(df)

    quantity = cast(pd.Series, df[s.quantity])
    discount = cast(pd.Series, df[s.discount])

    valid = (quantity > 0) & discount.between(0, 1)

    df = cast(pd.DataFrame, df.loc[valid].copy())

    log.info(
        "orders: invalid quantity/discount removed %d",
        previous_count - len(df),
    )

    # 5. Convert quantity to integer.
    quantity = cast(pd.Series, df[s.quantity])

    df[s.quantity] = quantity.astype("int64")

    # 6. Handle missing payment methods.
    payment_method = cast(pd.Series, df[s.payment_method])

    df[s.payment_method] = payment_method.fillna(UNKNOWN)

    return df


def filter_period(
    df: pd.DataFrame,
    s: Schema,
    p: PeriodConfig,
) -> pd.DataFrame:
    # 1. Convert the order date column to datetime.
    dates = cast(
        pd.Series,
        pd.to_datetime(
            df[s.order_date],
            errors="coerce",
        ),
    )

    # 2. Initialize a boolean mask.
    keep = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    # 3. Apply optional start date.
    if p.start is not None:
        keep &= dates >= pd.Timestamp(p.start)

    # 4. Apply optional end date.
    if p.end is not None:
        keep &= dates <= pd.Timestamp(p.end)

    # 5. Remove incomplete years when required.
    if p.full_years_only:
        filtered_dates = dates.loc[keep]

        months_per_year = (
            filtered_dates.dt.to_period("M").groupby(filtered_dates.dt.year).nunique()
        )

        partial_years = months_per_year.loc[months_per_year < 12]

        if not partial_years.empty:
            log.info(
                "periode: tahun tidak lengkap dibuang %s",
                partial_years.to_dict(),
            )

        keep &= ~dates.dt.year.isin(partial_years.index)

    # 6. Apply the final filter.
    out = cast(
        pd.DataFrame,
        df.loc[keep].copy(),
    )

    # 7. Ensure the result is not empty.
    if out.empty:
        raise ValueError(
            "Tidak ada order tersisa setelah filter periode; "
            "periksa konfigurasi 'period'."
        )

    # 8. Log the filtering result.
    log.info(
        "periode: %s .. %s, baris di luar periode dihapus %d",
        f"{out[s.order_date].min():%Y-%m-%d}",
        f"{out[s.order_date].max():%Y-%m-%d}",
        len(df) - len(out),
    )

    return out


def merge_tables(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    s: Schema,
) -> pd.DataFrame:
    payments = cast(
        pd.DataFrame,
        payments.loc[:, [s.order_id, s.payment_status]],
    )

    df = orders.merge(
        payments,
        on=s.order_id,
        validate="one_to_one",
    ).merge(
        products,
        on=s.product_id,
        how="left",
        validate="many_to_one",
    )

    previous_count = len(df)

    df = df.merge(
        customers,
        on=s.customer_id,
        how="inner",
        validate="many_to_one",
    )

    log.info(
        "merge: baris dengan customer tidak dikenal (orphan) dihapus %d",
        previous_count - len(df),
    )

    return df


def add_sales(
    df: pd.DataFrame,
    s: Schema,
) -> pd.DataFrame:
    df = df.copy()

    df[s.unit_price] = pd.to_numeric(
        df[s.unit_price],
        errors="coerce",
    )

    invalid_price = df[s.unit_price].isna() | (df[s.unit_price] <= 0)

    if bool(invalid_price.any()):
        raise ValueError("UnitPrice mengandung nilai kosong atau tidak valid.")

    df[s.sales] = (df[s.quantity] * df[s.unit_price] * (1 - df[s.discount])).round(2)

    return df


def clean_and_merge(
    data_dir: Path,
    s: Schema,
    period: PeriodConfig,
) -> pd.DataFrame:
    tables = load_tables(data_dir, s)

    _report_missing(tables)

    df = merge_tables(
        filter_period(
            clean_orders(tables["orders"], s),
            s,
            period,
        ),
        tables["payments"],
        tables["products"],
        clean_customers(tables["customers"], s),
        s,
    )

    df = add_sales(df, s).sort_values(s.order_id).reset_index(drop=True)

    missing: dict[str, int] = {}

    for column in df.columns:
        series = cast(pd.Series, df[column])
        count = int(cast(int, series.isna().sum()))

        if count > 0:
            missing[column] = count

    if missing:
        raise ValueError(f"Masih ada missing value setelah cleaning: {missing}")

    log.info("hasil cleaning: %s", df.shape)

    return df
