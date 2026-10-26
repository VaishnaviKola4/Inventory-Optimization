from pathlib import Path
import pandas as pd
from inventory_intelligence.demo_model import forecast_all_products, load_model_bundle
from inventory_intelligence.demo_service import build_demo_result, load_demo_tables

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_dataset_has_exactly_ten_thousand_observations():
    sales = pd.read_csv(ROOT / "data" / "sample_sales.csv")
    assert len(sales) == 10_000
    assert sales["product_id"].nunique() == 20
    assert sales.groupby("product_id").size().eq(500).all()


def test_saved_model_forecasts_nonnegative_daily_demand():
    tables = load_demo_tables()
    bundle = load_model_bundle(ROOT / "models" / "retail_demand_model.joblib")
    forecasts = forecast_all_products(tables["sales"], tables["products"], bundle, horizon=7)
    assert len(forecasts) == 20
    for fc in forecasts.values():
        assert len(fc["daily_forecast"]) == 7
        assert all(value >= 0 for value in fc["daily_forecast"])


def test_demo_service_emits_all_recommendation_types():
    result = build_demo_result(horizon=14, service_level=0.95)
    actions = {row["recommendation_type"] for row in result["inventory_recommendations"]}
    assert actions == {"Reorder", "Hold", "Promote"}
    assert result["sales_rows"] == 10_000
    assert result["summary"]["products_evaluated"] == 20
    assert len(result["forecast_series"]) == 20
