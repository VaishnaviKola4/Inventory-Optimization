from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
import math
import pandas as pd
from .demo_model import forecast_all_products, load_model_bundle
from .inventory import inventory_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "retail_demand_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"


def load_demo_tables(data_dir: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    return {
        "sales": pd.read_csv(data_dir / "sample_sales.csv", parse_dates=["date"]),
        "inventory": pd.read_csv(data_dir / "sample_inventory.csv"),
        "products": pd.read_csv(data_dir / "sample_products.csv"),
        "suppliers": pd.read_csv(data_dir / "sample_suppliers.csv"),
    }


def build_demo_result(horizon: int = 30, service_level: float = 0.95,
                      data_dir: str | Path = DATA_DIR, model_path: str | Path = MODEL_PATH) -> dict:
    tables = load_demo_tables(data_dir)
    sales, inventory, products, suppliers = (tables[k] for k in ["sales", "inventory", "products", "suppliers"])
    bundle = load_model_bundle(model_path)
    forecasts = forecast_all_products(sales, products, bundle, horizon=horizon)
    product_map = products.set_index(products.product_id.astype(str)).to_dict("index")
    supplier_map = suppliers.set_index(suppliers.product_id.astype(str)).to_dict("index")
    recs = []
    for row in inventory.to_dict("records"):
        pid = str(row["product_id"])
        p = product_map[pid]
        sup = supplier_map[pid]
        f = forecasts[pid]
        daily = f["daily_forecast"]
        metrics = inventory_metrics(
            row["on_hand"], row.get("on_order", 0), row.get("reserved", 0), daily,
            int(sup["lead_time_days"]), int(sup.get("review_period_days", 7)),
            float(f["daily_std"]), service_level
        )
        position, target, available = metrics["inventory_position"], metrics["target_stock"], metrics["available_stock"]
        pack = int(p.get("pack_size", 1) or 1)
        moq = int(p.get("minimum_order_quantity", 1) or 1)
        if position <= metrics["reorder_point"]:
            raw_qty = max(0.0, target - position)
            qty = max(moq, math.ceil(raw_qty / pack) * pack) if raw_qty > 0 else moq
            kind = "Reorder"
            timing = "Order today" if available <= metrics["lead_time_demand"] else "Order within 3 days"
            reason = (f"Inventory position ({position:.1f}) is at or below reorder point "
                      f"({metrics['reorder_point']:.1f}); estimated lead-time demand is "
                      f"{metrics['lead_time_demand']:.1f} units over {int(sup['lead_time_days'])} days.")
        elif metrics["projected_excess"] >= max(10, target * 0.25):
            qty = -int(metrics["projected_excess"])
            kind = "Promote"
            timing = "Review markdown this week"
            reason = (f"Available stock ({available:.1f}) exceeds target stock ({target:.1f}); "
                      f"estimated excess is {metrics['projected_excess']:.1f} units.")
        else:
            qty = 0
            kind = "Hold"
            timing = "Monitor weekly"
            reason = (f"Inventory position ({position:.1f}) is within the policy range; "
                      f"model forecasts {f['forecast_demand_units']:.1f} units over the next {horizon} days.")
        recs.append({
            "product_id": pid, "product_name": p.get("product_name", pid),
            "category": p.get("category", "Uncategorized"),
            "warehouse_id": row.get("warehouse_id", p.get("warehouse_id")),
            "recommendation_type": kind, "quantity": int(qty), "timing": timing,
            "reasoning": reason, "forecast_horizon_days": int(horizon),
            "forecast_demand_units": f["forecast_demand_units"],
            "forecast_method": f["forecast_method"],
            "supplier_lead_time_days": int(sup["lead_time_days"]),
            "metrics": metrics,
        })
    actions = {a: sum(r["recommendation_type"] == a for r in recs) for a in ["Reorder", "Hold", "Promote"]}
    metrics_file = Path(METRICS_PATH)
    model_metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": "Synthetic retail demo dataset — 10,000 daily SKU observations",
        "synthetic_data": True,
        "source_data_start_date": str(sales.date.min().date()),
        "source_data_end_date": str(sales.date.max().date()),
        "sales_rows": int(len(sales)), "sku_count": int(sales.product_id.nunique()),
        "warehouse_count": int(sales.warehouse_id.nunique()),
        "horizon_days": int(horizon), "service_level": float(service_level),
        "inventory_recommendations": recs,
        "forecast_series": forecasts,
        "summary": {
            "products_evaluated": len(recs), **{f"{a.lower()}_count": n for a, n in actions.items()},
            "forecast_units_total": round(sum(r["forecast_demand_units"] for r in recs), 2),
            "inventory_cost_value": round(sum(float(r["on_hand"]) * float(product_map[str(r["product_id"])].get("unit_cost", 0)) for r in inventory.to_dict("records")), 2),
            "projected_excess_units": round(sum(r["metrics"]["projected_excess"] for r in recs), 2),
            "data_points": int(len(sales)),
        },
        "model_metrics": model_metrics,
        "assumptions": {
            "dataset": "Generated synthetic retail data; not a real retailer dataset.",
            "inventory": "On-hand, on-order, reserved stock and supplier lead times are synthetic scenario values.",
            "future_calendar": "Promotion schedule and holiday markers are planned synthetic features.",
            "model": "Pre-trained Random Forest model shipped with this demo; retrain with train_demo_model.py.",
        },
    }
