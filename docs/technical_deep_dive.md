# Technical deep dive

## Architecture

```text
10,000 synthetic SKU-day sales rows
          |
          v
  Validation / product joins
          |
          v
  Feature engineering (19 signals)
          |
          v
  Global Random Forest regression
          |
          +---- time-based holdout evaluation (last 28 dates)
          |
          v
   Recursive daily demand forecast
          |
          v
 Inventory policy metrics + supplier lead time
          |
          v
 Explicit Reorder / Hold / Promote rules
          |
          v
 Streamlit UI · JSON · CSV · PDF · FastAPI · CLI
```

## Data synthesis approach

The included generator writes 20 product series over 500 dates each (10,000 records). Five categories and four warehouse zones are represented. Per-SKU base demand is modified by weekday effects, annual seasonality, category archetype (fast / steady / slow / seasonal / promotional / intermittent), product-specific promotion cycles, event dates, price changes, a small trend and occasional shocks. Demand is sampled as non-negative counts. The stock snapshot deliberately creates low-stock, healthy-cover, moderate-risk and excess-stock examples.

This is simulation design, not an empirical claim. It should not be used to represent a named retailer, estimate commercial outcomes, or substitute for actual operational data.

## Global model

A single `RandomForestRegressor` is trained on product-day observations. The pooled model can learn patterns shared across SKUs while using product, warehouse and category encodings to represent differences. Input features include price, promotion and holiday flags, day of week / month, annual Fourier terms, time trend, four demand lags and rolling mean/standard-deviation features.

Feature rows require 28 observations of warm-up history. To evaluate, rows from the last 28 dates are excluded from model fitting. For each SKU, predictions across the holdout are recursive: the prediction for day t is added to the history used to build day t+1's lag features. Holdout calendar, promotion and price signals are treated as known-ahead inputs. This is reasonable only when those future signals are part of an approved planning calendar.

After evaluation, the final saved model is refitted on every valid feature row in the complete generated history and serialized with joblib. `models/retail_demand_model.joblib` contains the fitted estimator, feature order, categorical mappings, synthetic origin date and product metadata. Do not load untrusted joblib artifacts.

## Metrics

- **MAE:** mean absolute error in units per SKU-day.
- **RMSE:** root mean squared error, with more emphasis on large misses.
- **WAPE:** total absolute error divided by total observed units, expressed as a percentage.
- **Bias:** mean forecast minus actual; positive values indicate average overforecasting.
- **Seasonal-naive MAE:** baseline that recursively repeats demand from seven days earlier.

Because the evaluation is synthetic and the series contain intermittent/zero demand, these scores only demonstrate the evaluation pipeline. A real project should repeat rolling-origin backtests by category, store and demand class, and measure service-level, stockout, fill-rate, and holding-cost outcomes in inventory simulations.

## Inventory policy

Available stock = `max(on_hand - reserved, 0)`.

Inventory position = available stock + open on-order units.

Lead-time demand is the sum of the forecast over the supplier's assumed lead time. Approximate safety stock uses a normal z-score for the configured service level multiplied by daily demand standard deviation and the square root of lead time. The reorder point is lead-time demand plus safety stock. Target stock covers lead time plus the inventory review period, plus safety stock.

The implementation is intentionally auditable. Reorder quantity is calculated to move inventory position toward target stock, rounded to order pack size and minimum order quantity. Promoting means reviewing a markdown/clearance action when available stock materially exceeds target. Hold is the default when neither threshold is met.

## Known limitations

- The safety-stock formula assumes a simplified relationship between daily demand variability and lead time; it is approximate for trending / intermittent demand.
- This demo does not optimize transfers between warehouses, expiry lots, cash/budget constraints, supplier delivery calendars or order quantity price breaks.
- Synthetic promotion schedules are known in advance. Production forecasts must not leak future actual sales or unapproved event information.
- Sales during stockout periods may be censored. Availability information improves demand estimates.
- The global demo model is only trained for the provided generated dataset. Uploaded data uses the separate general forecasting pipeline.
