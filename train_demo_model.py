from pathlib import Path
from inventory_intelligence.demo_model import train_and_save

ROOT = Path(__file__).resolve().parent
metrics = train_and_save(
    data_dir=ROOT / "data",
    model_dir=ROOT / "models",
    metrics_path=ROOT / "outputs" / "model_metrics.json",
    holdout_days=28,
)
print("Training complete")
print(f"Dataset rows: {metrics['sales_rows']:,}")
print(f"Training rows: {metrics['training_rows']:,}")
print(f"Holdout MAE: {metrics['model_mae_units_per_day']} units/day")
print(f"Holdout WAPE: {metrics['model_wape_percent']}%")
print(f"Seasonal naive MAE: {metrics['seasonal_naive_mae_units_per_day']} units/day")
print(f"Model artifact: {ROOT / 'models' / 'retail_demand_model.joblib'}")
