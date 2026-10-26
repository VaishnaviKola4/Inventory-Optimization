from __future__ import annotations
import math
import pandas as pd
from .inventory import inventory_metrics

def build_recommendations(sales: pd.DataFrame, inventory: pd.DataFrame,
                          products: pd.DataFrame, suppliers: pd.DataFrame,
                          forecasts: dict, horizon: int = 30,
                          service_level: float = 0.95) -> list[dict]:
    recs = []
    inv_has_wh = "warehouse_id" in inventory.columns
    sales_has_wh = "warehouse_id" in sales.columns
    # Aggregate at product level unless warehouse-specific inventory is supplied.
    inv_rows = inventory.to_dict("records")
    product_map = products.set_index(products["product_id"].astype(str), drop=False).to_dict("index")
    supplier_map = suppliers.set_index(suppliers["product_id"].astype(str), drop=False).to_dict("index")
    sales_ids = set(sales["product_id"].astype(str))
    inv_ids = set(inventory["product_id"].astype(str))
    all_ids = sorted(sales_ids | inv_ids | set(products["product_id"].astype(str)))

    for pid in all_ids:
        p = product_map.get(pid, {})
        s = supplier_map.get(pid, {"lead_time_days": 7, "review_period_days": 7})
        f = forecasts.get(pid, {"daily_forecast": [0.0] * horizon, "method": "no_history",
                                "daily_mean": 0.0, "daily_std": 0.0})
        rows = [r for r in inv_rows if str(r["product_id"]) == pid]
        if not rows:
            rows = [{"product_id": pid, "on_hand": 0, "on_order": 0, "reserved": 0}]
        # If there are multiple warehouses, provide one recommendation per warehouse.
        for row in rows:
            wh = row.get("warehouse_id")
            local_sales = sales[sales["product_id"].astype(str) == pid]
            if wh is not None and sales_has_wh:
                local_sales = local_sales[local_sales["warehouse_id"].astype(str) == str(wh)]
                local_forecast = forecasts.get(f"{pid}::{wh}", f)
            else:
                local_forecast = f
            daily = local_forecast.get("daily_forecast", [0.0] * horizon)
            metrics = inventory_metrics(
                row.get("on_hand", 0), row.get("on_order", 0), row.get("reserved", 0),
                daily, int(s.get("lead_time_days", 7)), int(s.get("review_period_days", 7)),
                float(local_forecast.get("daily_std", 0.0)), service_level
            )
            pack = int(p.get("pack_size", 1) or 1)
            moq = int(p.get("minimum_order_quantity", 1) or 1)
            position = metrics["inventory_position"]
            target = metrics["target_stock"]
            reorder_point = metrics["reorder_point"]
            if position <= reorder_point and (metrics["daily_forecast_mean"] > 0 or metrics["available_stock"] <= 0):
                raw_qty = max(0.0, target - position)
                qty = max(moq, math.ceil(raw_qty / pack) * pack) if raw_qty > 0 else moq
                kind = "Reorder"
                timing = "Order immediately" if metrics["available_stock"] <= metrics["lead_time_demand"] else "Order within 3 days"
                reason = (f"Inventory position ({position:.1f}) is at/below reorder point "
                          f"({reorder_point:.1f}); forecast lead-time demand is "
                          f"{metrics['lead_time_demand']:.1f} units and supplier lead time is "
                          f"{int(s.get('lead_time_days', 7))} days.")
            elif metrics["projected_excess"] >= max(10, target * 0.25) and metrics["available_stock"] > 0:
                qty = -int(math.floor(metrics["projected_excess"]))
                kind = "Promote"
                timing = "Review markdown/promotion this week"
                reason = (f"Available stock ({metrics['available_stock']:.1f}) exceeds target stock "
                          f"({target:.1f}); projected excess is {metrics['projected_excess']:.1f} units.")
            else:
                qty = 0
                kind = "Hold"
                timing = "Monitor weekly"
                reason = (f"Inventory position ({position:.1f}) is within the current target range; "
                          f"forecast average is {metrics['daily_forecast_mean']:.2f} units/day.")
            rec = {
                "product_id": pid,
                "recommendation_type": kind,
                "quantity": int(qty),
                "timing": timing,
                "reasoning": reason,
                "warehouse_id": str(wh) if wh is not None else None,
                "forecast_horizon_days": int(horizon),
                "forecast_demand_units": round(float(sum(daily)), 2),
                "forecast_method": local_forecast.get("method", "unknown"),
                "supplier_lead_time_days": int(s.get("lead_time_days", 7)),
                "metrics": metrics,
            }
            recs.append(rec)
    return recs
