# How to use Stockwise

## Demo in three commands

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens in the synthetic demo workspace by default. It loads the 10,000-row CSV dataset, pre-trained Random Forest model, product catalog, inventory snapshot, supplier terms and holdout metrics. You do not need to train a model before your first demo.

## Regenerate / retrain

```bash
python data/generate_demo_data.py
python train_demo_model.py
```

The generator writes a daily history with 20 products × 500 days. The training script constructs lag and calendar features, holds the final 28 days aside for evaluation, compares the model with seasonal-naive predictions, and finally retrains on all valid history before persisting `models/retail_demand_model.joblib`.

Regenerating the data changes the ending date and values because the time range is relative to the current date. The fixed random seed ensures repeatable values for a given run date and setup.

## Use the dashboard

### Command center

Start with the executive snapshot. The action mix shows the count of items labelled Reorder, Hold or Promote. The stock-position chart compares available stock, policy reorder point and target stock. Review the priority queue first.

### Recommendations

Filter by action, category, or SKU/product name. Select a recommendation to inspect its explanation and related inventory metrics. Positive quantities indicate a suggested order; negative Promote quantities indicate an approximate excess quantity for markdown / clearance review.

### Forecasts

Choose an SKU to review the most recent 90 days of observed demand and the future model forecast. Forecasts are generated recursively, so earlier predicted values are used to produce lag features for later days.

### Model lab

Review the 28-day holdout MAE, RMSE/WAPE, and comparison with a seasonal-naive baseline. Feature importance describes how the fitted Random Forest used its inputs; it is not a causal explanation. Use **Retrain on synthetic dataset** to rerun the train/evaluate/save process from the interface.

### Data explorer

Review and download the data tables. **Download all demo data (ZIP)** includes sales, inventory, product and supplier CSVs plus the model metrics JSON.

### Exports

Download recommendation JSON for system integration, CSV for spreadsheet review, or a PDF summary for sharing. Exported recommendations are decision-support outputs, not purchase orders.

## Bring your own data

Choose **Upload your data** and supply all four normalized CSV/XLSX tables. Required columns:

- Sales: `date`, `product_id`, `units_sold`
- Inventory: `product_id`, `on_hand`
- Products: `product_id`
- Suppliers: `product_id`, `lead_time_days`

Optional columns are documented in `data_dictionary.md`. The four files must use consistent product IDs and units. Invalid dates, negative sales, and duplicate inventory/product keys are rejected by the validation layer. The upload workflow currently uses the general per-SKU forecasting path; the pre-trained pooled model is shipped for the generated demo data.

## Run CLI / API

```bash
python -m inventory_intelligence.cli \
  --sales data/sample_sales.csv \
  --inventory data/sample_inventory.csv \
  --products data/sample_products.csv \
  --suppliers data/sample_suppliers.csv \
  --horizon 30 --output outputs/recommendations.json \
  --csv outputs/recommendations.csv

uvicorn inventory_intelligence.api:app --reload
```

The API's interactive schema is available at `/docs`; the health endpoint is `/health`.
