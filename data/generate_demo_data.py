from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261023
N_PRODUCTS = 20
N_DAYS = 500

CATEGORIES = {
    "Grocery": ["Snack", "Beverage", "Pantry"],
    "Home": ["Storage", "Kitchen", "Decor"],
    "Personal Care": ["Skincare", "Haircare", "Wellness"],
    "Apparel": ["Basics", "Seasonal", "Accessories"],
    "Electronics": ["Audio", "Accessories", "Smart Home"],
}


def is_holiday(dt: pd.Timestamp) -> int:
    # Synthetic retail event calendar: US-style major shopping events.
    thanksgiving = pd.Timestamp(year=dt.year, month=11, day=1)
    thanksgiving = thanksgiving + pd.Timedelta(days=(3 - thanksgiving.dayofweek) % 7 + 21)
    black_friday = thanksgiving + pd.Timedelta(days=1)
    cyber_monday = thanksgiving + pd.Timedelta(days=4)
    ranges = [
        (pd.Timestamp(dt.year, 1, 1), pd.Timestamp(dt.year, 1, 3)),
        (pd.Timestamp(dt.year, 2, 12), pd.Timestamp(dt.year, 2, 15)),
        (pd.Timestamp(dt.year, 7, 1), pd.Timestamp(dt.year, 7, 5)),
        (pd.Timestamp(dt.year, 8, 20), pd.Timestamp(dt.year, 9, 10)),
        (pd.Timestamp(dt.year, 11, 20), pd.Timestamp(dt.year, 11, 30)),
        (pd.Timestamp(dt.year, 12, 15), pd.Timestamp(dt.year, 12, 31)),
        (thanksgiving, thanksgiving), (black_friday, black_friday),
        (cyber_monday, cyber_monday),
    ]
    return int(any(start <= dt <= end for start, end in ranges))


def planned_promotion(day_index: int, sku_index: int) -> int:
    # Deterministic campaign calendar, known to the synthetic retailer.
    cycle = 47 + (sku_index % 5) * 3
    offset = (sku_index * 7 + 11) % cycle
    return int((day_index + offset) % cycle in {0, 1, 2, 3})


def generate(output_dir: str | Path = "data", n_days: int = N_DAYS, seed: int = SEED):
    if n_days < 100:
        raise ValueError("n_days must be at least 100 so lag features and holdout evaluation are meaningful")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")
    store_ids = ["WH-NORTH", "WH-SOUTH", "WH-EAST", "WH-WEST"]
    base_products = [
        ("SKU-1001", "Granola Snack Box", "Grocery", 8.0, 2.49, 0.11, 8, 45, 5, 7, "fast"),
        ("SKU-1002", "Sparkling Water 12-pack", "Grocery", 11.0, 5.99, 0.08, 12, 48, 4, 7, "fast"),
        ("SKU-1003", "Premium Coffee Beans", "Grocery", 5.5, 12.99, 0.06, 6, 24, 9, 7, "steady"),
        ("SKU-1004", "Protein Bar Variety", "Grocery", 7.5, 3.49, 0.09, 10, 50, 6, 7, "promo"),
        ("SKU-1005", "Olive Oil 500ml", "Grocery", 2.7, 10.99, 0.04, 4, 20, 12, 7, "steady"),
        ("SKU-2001", "Modular Storage Bin", "Home", 4.5, 18.99, 0.04, 6, 30, 10, 7, "steady"),
        ("SKU-2002", "Ceramic Mug Set", "Home", 2.5, 24.99, 0.05, 4, 20, 14, 7, "seasonal"),
        ("SKU-2003", "LED Desk Lamp", "Home", 1.8, 34.99, 0.03, 3, 15, 16, 7, "slow"),
        ("SKU-2004", "Kitchen Organizer", "Home", 3.7, 16.50, 0.04, 5, 25, 9, 7, "steady"),
        ("SKU-3001", "Hydrating Face Serum", "Personal Care", 6.5, 19.99, 0.07, 8, 40, 6, 7, "fast"),
        ("SKU-3002", "Daily Shampoo 400ml", "Personal Care", 9.0, 8.49, 0.03, 12, 60, 5, 7, "steady"),
        ("SKU-3003", "Vitamin Wellness Pack", "Personal Care", 3.2, 14.99, 0.04, 4, 20, 8, 7, "intermittent"),
        ("SKU-3004", "Hair Styling Kit", "Personal Care", 1.1, 22.99, 0.06, 3, 15, 12, 7, "slow"),
        ("SKU-4001", "Everyday Cotton Tee", "Apparel", 5.0, 14.99, 0.05, 6, 30, 10, 7, "steady"),
        ("SKU-4002", "Lightweight Jacket", "Apparel", 1.8, 59.99, 0.08, 2, 12, 21, 7, "seasonal"),
        ("SKU-4003", "Everyday Socks 5-pack", "Apparel", 6.0, 9.99, 0.04, 8, 40, 7, 7, "fast"),
        ("SKU-4004", "Canvas Tote Bag", "Apparel", 2.0, 12.50, 0.05, 4, 20, 11, 7, "slow"),
        ("SKU-5001", "Wireless Earbuds", "Electronics", 2.2, 49.99, 0.04, 3, 18, 14, 7, "promo"),
        ("SKU-5002", "USB-C Charging Cable", "Electronics", 7.0, 11.99, 0.03, 10, 50, 8, 7, "fast"),
        ("SKU-5003", "Smart Plug Duo", "Electronics", 1.5, 29.99, 0.04, 3, 15, 13, 7, "intermittent"),
    ]
    product_rows = []
    sku_to_idx = {}
    for i, product in enumerate(base_products):
        pid, name, category, base, price, cv, pack, moq, lead, review, pattern = product
        sku_to_idx[pid] = i
        product_rows.append({
            "product_id": pid, "product_name": name, "category": category,
            "warehouse_id": store_ids[i % len(store_ids)], "unit_cost": round(price * (0.46 + 0.06 * (i % 3)), 2),
            "unit_price": price, "pack_size": pack, "minimum_order_quantity": moq,
            "shelf_life_days": 180 if category == "Grocery" else 0,
            "base_daily_demand": base, "demand_pattern": pattern,
        })
    products = pd.DataFrame(product_rows)

    sales_rows = []
    start = dates[0]
    for i, product in enumerate(base_products):
        pid, name, category, base, price, cv, pack, moq, lead, review, pattern = product
        store_id = store_ids[i % len(store_ids)]
        sku_phase = (i * 13) % 365
        trend_rate = ((i % 5) - 2) * 0.00024
        for day_idx, dt in enumerate(dates):
            # Weekly behavior, slow trend, annual seasonal effect, and category behavior.
            dow = dt.dayofweek
            weekly = 1.18 if dow in (4, 5, 6) else (0.91 if dow == 0 else 1.0)
            yearly = 1 + 0.22 * np.sin(2 * np.pi * (dt.dayofyear + sku_phase) / 365.25)
            trend = max(0.78, 1 + trend_rate * day_idx)
            holiday = is_holiday(dt)
            holiday_lift = 1.35 if holiday else 1.0
            promo = planned_promotion(day_idx, i)
            promo_lift = 1.55 if promo else 1.0
            price_jitter = 0.0
            # Planned markdown weeks; observed price is included as an input signal.
            if promo:
                price_jitter = -0.08
            elif day_idx % 113 in (0, 1):
                price_jitter = 0.03
            unit_price = round(price * (1 + price_jitter), 2)
            elasticity = max(0.65, 1 - 0.65 * ((unit_price / price) - 1))
            mean = base * weekly * yearly * trend * holiday_lift * promo_lift * elasticity
            if pattern == "seasonal":
                winter = 1.0 + 0.8 * np.cos(2 * np.pi * (dt.dayofyear - 15) / 365.25)
                mean *= max(0.3, winter)
            elif pattern == "intermittent":
                if rng.random() < 0.52:
                    mean *= 0.05
                else:
                    mean *= 2.4
            elif pattern == "slow":
                mean *= 0.8
            # Negative-binomial-like overdispersion via Poisson with occasional shocks.
            if rng.random() < 0.008:
                mean *= rng.uniform(2.0, 4.0)
            units = int(rng.poisson(max(0.05, mean)))
            sales_rows.append({
                "date": dt.strftime("%Y-%m-%d"), "product_id": pid,
                "warehouse_id": store_id, "units_sold": units,
                "promotion_flag": promo, "holiday_flag": holiday,
                "unit_price": unit_price,
            })
    sales = pd.DataFrame(sales_rows)
    # Exactly 20 SKUs x 500 daily observations = 10,000 data points.
    assert len(sales) == N_PRODUCTS * n_days
    sales.to_csv(out / "sample_sales.csv", index=False)
    products.to_csv(out / "sample_products.csv", index=False)

    # Inventory scenarios are intentionally varied to demonstrate all action types.
    inventory_rows, supplier_rows = [], []
    history = sales.copy()
    history["date"] = pd.to_datetime(history["date"])
    for i, product in products.iterrows():
        pid = product["product_id"]
        p_hist = history[history.product_id == pid].sort_values("date")
        recent_daily = float(p_hist.tail(28).units_sold.mean())
        lead = int(base_products[i][8])
        review = int(base_products[i][9])
        pattern = base_products[i][10]
        scenario = i % 5
        if scenario == 0:  # critically low
            on_hand = max(0, int(round(recent_daily * rng.uniform(1.5, 3.5))))
            on_order = 0
        elif scenario == 1:  # healthy cover
            on_hand = max(1, int(round(recent_daily * rng.uniform(lead + review + 4, lead + review + 12))))
            on_order = int(round(recent_daily * 2))
        elif scenario == 2:  # excess
            on_hand = max(1, int(round(recent_daily * rng.uniform(70, 110))))
            on_order = 0
        elif scenario == 3:  # moderate risk
            on_hand = max(0, int(round(recent_daily * rng.uniform(4, max(5, lead * 0.8)))))
            on_order = int(round(recent_daily * 2))
        else:  # balanced, but some slower products can become overstocked
            on_hand = max(1, int(round(recent_daily * rng.uniform(12, 28))))
            on_order = int(round(recent_daily * rng.uniform(0, 4)))
        reserved = int(round(on_hand * rng.uniform(0.00, 0.10)))
        inventory_rows.append({
            "snapshot_date": dates[-1].strftime("%Y-%m-%d"), "product_id": pid,
            "warehouse_id": product["warehouse_id"], "on_hand": on_hand,
            "on_order": on_order, "reserved": min(reserved, on_hand),
        })
        supplier_rows.append({
            "product_id": pid, "supplier_id": f"SUP-{(i % 8) + 1:02d}",
            "lead_time_days": lead, "review_period_days": review,
        })
    pd.DataFrame(inventory_rows).to_csv(out / "sample_inventory.csv", index=False)
    pd.DataFrame(supplier_rows).to_csv(out / "sample_suppliers.csv", index=False)
    print(f"Generated {len(sales):,} synthetic daily sales observations across {N_PRODUCTS} SKUs.")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Files written to: {out.resolve()}")

if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
