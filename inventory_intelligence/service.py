from __future__ import annotations
from .data import load_inputs
from .forecasting import forecast_all, forecast_product
from .recommendations import build_recommendations

def generate_recommendations(sales, inventory, products, suppliers, horizon=30,
                             service_level=0.95, minimum_history_days=45):
    sales_df, inventory_df, products_df, suppliers_df = load_inputs(sales, inventory, products, suppliers)
    forecasts = forecast_all(sales_df, horizon=horizon, minimum_history_days=minimum_history_days)
    # Create warehouse-level forecasts when both sales and inventory have warehouse IDs.
    if "warehouse_id" in sales_df.columns and "warehouse_id" in inventory_df.columns:
        for (pid, wh), group in sales_df.groupby(["product_id", "warehouse_id"]):
            forecasts[f"{str(pid)}::{str(wh)}"] = forecast_product(
                group[["date", "units_sold"]], horizon, minimum_history_days
            )
    recommendations = build_recommendations(
        sales_df, inventory_df, products_df, suppliers_df, forecasts,
        horizon=horizon, service_level=service_level
    )
    return {
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "horizon_days": int(horizon),
        "service_level": float(service_level),
        "inventory_recommendations": recommendations,
        "summary": {
            "products_evaluated": len({r["product_id"] for r in recommendations}),
            "reorder_count": sum(r["recommendation_type"] == "Reorder" for r in recommendations),
            "hold_count": sum(r["recommendation_type"] == "Hold" for r in recommendations),
            "promote_count": sum(r["recommendation_type"] == "Promote" for r in recommendations),
        }
    }
