from inventory_intelligence.inventory import inventory_metrics

def test_inventory_metrics_computes_cover_and_reorder_point():
    result = inventory_metrics(
        on_hand=20, on_order=5, reserved=2,
        daily_forecast=[4.0] * 30, lead_time_days=5,
        review_period_days=7, daily_std=1.0, service_level=0.95
    )
    assert result["available_stock"] == 18
    assert result["inventory_position"] == 23
    assert result["lead_time_demand"] == 20
    assert result["reorder_point"] > result["lead_time_demand"]
    assert result["days_of_cover"] == 4.5

def test_zero_demand_has_no_division_error():
    result = inventory_metrics(10, 0, 0, [0.0] * 30, 5)
    assert result["days_of_cover"] is None
    assert result["projected_excess"] >= 0
