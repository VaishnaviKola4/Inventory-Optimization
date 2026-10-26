# API / interface guide

Base URL in local development: `http://127.0.0.1:8000`

## `GET /health`

Returns service status.

```json
{"status":"ok","service":"inventory-intelligence"}
```

## `POST /recommendations`

Request body:

```json
{
  "sales": [{"date":"2026-01-01","product_id":"SKU1","units_sold":5}],
  "inventory": [{"product_id":"SKU1","on_hand":20,"on_order":0,"reserved":0}],
  "products": [{"product_id":"SKU1","pack_size":1,"minimum_order_quantity":1}],
  "suppliers": [{"product_id":"SKU1","lead_time_days":5,"review_period_days":7}],
  "horizon": 30,
  "service_level": 0.95,
  "minimum_history_days": 45
}
```

The example is schema-valid but intentionally too short for a meaningful seasonal forecast.

Response shape:

```json
{
  "generated_at": "ISO-8601 timestamp",
  "horizon_days": 30,
  "service_level": 0.95,
  "inventory_recommendations": [
    {
      "product_id": "SKU1",
      "recommendation_type": "Hold",
      "quantity": 0,
      "timing": "Monitor weekly",
      "reasoning": "Explanation...",
      "warehouse_id": null,
      "forecast_horizon_days": 30,
      "forecast_demand_units": 120.0,
      "forecast_method": "random_forest_lag_features",
      "supplier_lead_time_days": 5,
      "metrics": {
        "available_stock": 20.0,
        "inventory_position": 20.0,
        "daily_forecast_mean": 4.0,
        "lead_time_demand": 20.0,
        "safety_stock": 8.0,
        "reorder_point": 28.0,
        "target_stock": 48.0,
        "days_of_cover": 5.0,
        "projected_excess": 0.0
      }
    }
  ],
  "summary": {
    "products_evaluated": 1,
    "reorder_count": 0,
    "hold_count": 1,
    "promote_count": 0
  }
}
```

The exact recommendation and numeric values depend on submitted data. Invalid payloads return HTTP 422; unexpected errors return HTTP 500 without exposing internal exception details.
