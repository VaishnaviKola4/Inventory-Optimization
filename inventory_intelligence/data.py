from __future__ import annotations
from pathlib import Path
import pandas as pd

SALES_REQUIRED = {"date", "product_id", "units_sold"}
INVENTORY_REQUIRED = {"product_id", "on_hand"}
PRODUCTS_REQUIRED = {"product_id"}
SUPPLIERS_REQUIRED = {"product_id", "lead_time_days"}

class DataValidationError(ValueError):
    """Raised when input data does not satisfy the expected schema."""

def _read_csv(source):
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (list, tuple)):
        return pd.DataFrame(source)
    if isinstance(source, dict):
        return pd.DataFrame(source)
    return pd.read_csv(Path(source))

def _require(df: pd.DataFrame, required: set[str], name: str):
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {sorted(missing)}")

def load_inputs(sales, inventory, products, suppliers):
    sales_df, inventory_df = _read_csv(sales), _read_csv(inventory)
    products_df, suppliers_df = _read_csv(products), _read_csv(suppliers)
    _require(sales_df, SALES_REQUIRED, "sales")
    _require(inventory_df, INVENTORY_REQUIRED, "inventory")
    _require(products_df, PRODUCTS_REQUIRED, "products")
    _require(suppliers_df, SUPPLIERS_REQUIRED, "suppliers")

    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    if sales_df["date"].isna().any():
        raise DataValidationError("sales.date contains invalid or missing dates")
    for col in ["units_sold"]:
        sales_df[col] = pd.to_numeric(sales_df[col], errors="coerce")
        if sales_df[col].isna().any() or (sales_df[col] < 0).any():
            raise DataValidationError(f"sales.{col} must contain non-negative numeric values")
    for col in ["promotion_flag", "holiday_flag"]:
        if col not in sales_df:
            sales_df[col] = 0
        sales_df[col] = pd.to_numeric(sales_df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    for col in ["on_hand"]:
        inventory_df[col] = pd.to_numeric(inventory_df[col], errors="coerce")
        if inventory_df[col].isna().any() or (inventory_df[col] < 0).any():
            raise DataValidationError(f"inventory.{col} must contain non-negative numeric values")
    for col in ["on_order", "reserved"]:
        if col not in inventory_df:
            inventory_df[col] = 0
        inventory_df[col] = pd.to_numeric(inventory_df[col], errors="coerce").fillna(0)
        if (inventory_df[col] < 0).any():
            raise DataValidationError(f"inventory.{col} must be non-negative")

    suppliers_df["lead_time_days"] = pd.to_numeric(suppliers_df["lead_time_days"], errors="coerce")
    if suppliers_df["lead_time_days"].isna().any() or (suppliers_df["lead_time_days"] < 0).any():
        raise DataValidationError("suppliers.lead_time_days must be non-negative numeric values")
    if "review_period_days" not in suppliers_df:
        suppliers_df["review_period_days"] = 7
    suppliers_df["review_period_days"] = pd.to_numeric(
        suppliers_df["review_period_days"], errors="coerce"
    ).fillna(7).clip(lower=1)

    defaults = {"category": "Uncategorized", "unit_cost": 0.0, "unit_price": 0.0,
                "shelf_life_days": 0, "pack_size": 1, "minimum_order_quantity": 1}
    for col, default in defaults.items():
        if col not in products_df:
            products_df[col] = default
        products_df[col] = products_df[col].fillna(default)
    for col in ["unit_cost", "unit_price", "shelf_life_days", "pack_size", "minimum_order_quantity"]:
        products_df[col] = pd.to_numeric(products_df[col], errors="coerce").fillna(defaults[col])
    products_df["pack_size"] = products_df["pack_size"].clip(lower=1).astype(int)
    products_df["minimum_order_quantity"] = products_df["minimum_order_quantity"].clip(lower=1).astype(int)

    if sales_df.duplicated(["date", "product_id"] + (["warehouse_id"] if "warehouse_id" in sales_df else [])).any():
        # Aggregate duplicate daily rows instead of silently discarding sales.
        keys = ["date", "product_id"] + (["warehouse_id"] if "warehouse_id" in sales_df else [])
        aggregations = {"units_sold": "sum", "promotion_flag": "max", "holiday_flag": "max"}
        sales_df = sales_df.groupby(keys, as_index=False).agg(aggregations)
    if inventory_df.duplicated(["product_id"] + (["warehouse_id"] if "warehouse_id" in inventory_df else [])).any():
        raise DataValidationError("inventory has duplicate product/warehouse keys")
    if products_df["product_id"].duplicated().any():
        raise DataValidationError("products.product_id must be unique")
    if suppliers_df["product_id"].duplicated().any():
        raise DataValidationError("suppliers.product_id must be unique")

    return sales_df, inventory_df, products_df, suppliers_df
