import pandas as pd
from inventory_intelligence.service import generate_recommendations

def test_service_returns_recommendation():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    sales = pd.DataFrame({"date": dates, "product_id": "A", "units_sold": [5] * 60})
    inventory = pd.DataFrame([{"product_id": "A", "on_hand": 10, "on_order": 0, "reserved": 0}])
    products = pd.DataFrame([{"product_id": "A", "pack_size": 1, "minimum_order_quantity": 1}])
    suppliers = pd.DataFrame([{"product_id": "A", "lead_time_days": 7, "review_period_days": 7}])
    result = generate_recommendations(sales, inventory, products, suppliers, horizon=30)
    assert len(result["inventory_recommendations"]) == 1
    assert result["inventory_recommendations"][0]["recommendation_type"] in {"Reorder", "Hold", "Promote"}
    assert "reasoning" in result["inventory_recommendations"][0]
