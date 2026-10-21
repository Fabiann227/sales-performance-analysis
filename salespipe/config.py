"""Konfigurasi pipeline: nama file, nama kolom (schema), dan parameter model.

Default cocok dengan dataset di folder datasets/. Untuk dataset lain dengan
struktur serupa tapi nama file/kolom berbeda, cukup buat file JSON override
(lihat config.example.json) dan berikan lewat --config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Schema:
    # nama file tiap tabel di folder data
    orders_file: str = "orders.csv"
    customers_file: str = "customers.csv"
    payments_file: str = "payments.csv"
    products_file: str = "products.csv"

    # kunci
    order_id: str = "OrderID"
    customer_id: str = "CustomerID"
    product_id: str = "ProductID"

    # kolom orders
    order_date: str = "OrderDate"
    quantity: str = "Quantity"
    discount: str = "Discount"  # pecahan, mis. 0.1 = diskon 10%
    payment_method: str = "PaymentMethod"
    status: str = "Status"

    # kolom customers
    age: str = "Age"
    city: str = "City"
    signup_date: str = "SignupDate"

    # kolom payments
    payment_status: str = "PaymentStatus"

    # kolom products
    product_name: str = "ProductName"
    category: str = "Category"
    unit_price: str = "UnitPrice"

    # kolom turunan hasil pipeline
    sales: str = "Sales"

    # koreksi penulisan kota yang diketahui salah (selain title-case otomatis)
    city_aliases: dict = field(default_factory=lambda: {"Mashad": "Mashhad"})


@dataclass(frozen=True)
class ModelConfig:
    sales_status: str | None = "Completed"  # hanya status ini yang dihitung sebagai penjualan; None = semua
    train_ratio: float = 0.70
    val_ratio: float = 0.15  # sisanya menjadi test
    poly_degrees: tuple = (2, 5)
    alphas: tuple = (0.1, 1, 10)


@dataclass(frozen=True)
class Config:
    schema: Schema = field(default_factory=Schema)
    model: ModelConfig = field(default_factory=ModelConfig)


def _override(obj, values: dict):
    valid = {f.name for f in fields(obj)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Kunci konfigurasi tidak dikenal: {sorted(unknown)}. Yang valid: {sorted(valid)}")
    return replace(obj, **values)


def load_config(path: str | Path | None) -> Config:
    """Baca override dari JSON: {"schema": {...}, "model": {...}}. None = default."""
    cfg = Config()
    if path is None:
        return cfg
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"schema", "model"}
    if unknown:
        raise ValueError(f"Bagian konfigurasi tidak dikenal: {sorted(unknown)}")
    return Config(
        schema=_override(cfg.schema, raw.get("schema", {})),
        model=_override(cfg.model, raw.get("model", {})),
    )
