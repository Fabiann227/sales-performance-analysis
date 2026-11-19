"""Pipeline configuration: file names, schema columns, and model parameters.

The defaults match the datasets in the datasets/ folder. For another dataset
with a similar structure, create a JSON override and pass it with --config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Schema:
    # File name for each table in the data directory.
    orders_file: str = "orders.csv"
    customers_file: str = "customers.csv"
    payments_file: str = "payments.csv"
    products_file: str = "products.csv"

    # Keys.
    order_id: str = "OrderID"
    customer_id: str = "CustomerID"
    product_id: str = "ProductID"

    # Order columns.
    order_date: str = "OrderDate"
    quantity: str = "Quantity"
    discount: str = "Discount"  # Fraction, for example 0.1 = 10% discount.
    payment_method: str = "PaymentMethod"
    status: str = "Status"

    # Customer columns.
    age: str = "Age"
    city: str = "City"
    signup_date: str = "SignupDate"

    # Payment columns.
    payment_status: str = "PaymentStatus"

    # Product columns.
    product_name: str = "ProductName"
    category: str = "Category"
    unit_price: str = "UnitPrice"

    # Derived pipeline column.
    sales: str = "Sales"

    # Known city spelling corrections in addition to title casing.
    city_aliases: dict[str, str] = field(default_factory=lambda: {"Mashad": "Mashhad"})


@dataclass(frozen=True)
class ModelConfig:
    sales_status: str | None = (
        "Completed"  # Only this status contributes to sales; None means all statuses.
    )
    train_ratio: float = 0.70
    val_ratio: float = 0.15  # The remainder becomes the test set.
    poly_degrees: tuple[int, ...] = (2, 5)
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class PeriodConfig:
    full_years_only: bool = (
        True  # Drop years without all 12 months, such as a partial 2026.
    )
    start: str | None = None  # Optional lower bound, format "YYYY-MM-DD".
    end: str | None = None  # Optional upper bound, format "YYYY-MM-DD".


@dataclass(frozen=True)
class Config:
    schema: Schema = field(default_factory=Schema)
    model: ModelConfig = field(default_factory=ModelConfig)
    period: PeriodConfig = field(default_factory=PeriodConfig)


def _override[T: Schema | ModelConfig | PeriodConfig](
    obj: T,
    values: dict[str, object],
) -> T:
    valid = {f.name for f in fields(obj)}
    unknown = set(values) - valid

    if unknown:
        raise ValueError(
            f"Unknown configuration keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )

    return replace(obj, **values)


def load_config(path: str | Path | None) -> Config:
    """Load a JSON override with schema, model, and period sections."""
    cfg = Config()
    if path is None:
        return cfg
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"schema", "model", "period"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return Config(
        schema=_override(cfg.schema, raw.get("schema", {})),
        model=_override(cfg.model, raw.get("model", {})),
        period=_override(cfg.period, raw.get("period", {})),
    )
