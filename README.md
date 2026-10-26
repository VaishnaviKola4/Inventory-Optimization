# Stockwise — AI Inventory Intelligence

## Problem Statement

Retail inventory management constantly faces a difficult balancing act: **stockouts** lead to lost revenue, missed customer service level agreements, and diminished customer loyalty, while **overstocking** ties up valuable working capital, inflates warehousing holding costs, and heightens the risk of inventory obsolescence, perishability, and forced discount markdowns.

Traditional replenishment methods rely heavily on static reorder points, aggregate rules of thumb, or disconnected spreadsheets that fail to:
- Adapt dynamically to multi-factor demand drivers (such as promotions, seasonality, calendar events, day-of-week trends, and recent sales velocity).
- Account for variable supplier lead times and targeted customer service levels simultaneously.
- Provide clear, transparent, and explainable decision support (prescribing *what* to order, *when* to order, or *when* to hold/promote stock).

**Stockwise** solves this problem by integrating machine learning demand forecasting with automated inventory optimization policies. It forecasts SKU-level demand recursively, computes dynamic safety stocks and reorder thresholds, and generates transparent, auditable inventory action recommendations (`Reorder`, `Hold`, or `Promote`) to balance service levels with working capital efficiency.

> **A self-contained retail inventory analytics demo, powered by a pre-trained demand model and 10,000 synthetic daily sales observations.** No Kaggle account, external dataset download, or API key is required.

Stockwise helps inventory planners answer three practical questions: **What may sell next? Where is stock at risk? What should we do now?** It brings demand forecasting, supplier lead-time policy, stock-cover analysis and explainable action recommendations into a polished Streamlit dashboard.

> **Demo-data notice:** The included sales history, product catalog, stock snapshot and supplier terms are synthetic. They were generated to mimic retail patterns for demonstration and testing. They are not real Walmart, retailer, or supplier records. Forecast metrics measured on synthetic holdout data do not establish real-world performance.

## At a glance

- **10,000** daily SKU sales observations
- **20** products across **5** product categories and **4** warehouse zones
- **500 days** of historical demand per product
- **19 engineered features** across calendar, price, promotion, SKU/location, lag and rolling-demand signals
- A trained **global Random Forest regression model** saved in `models/retail_demand_model.joblib`
- A separate 28-day holdout report compared with a seasonal-naive baseline
- Explainable `Reorder`, `Hold`, and `Promote` actions
- Streamlit dashboard with charts, filters, data explorer, model evaluation, and JSON / CSV / PDF downloads
- FastAPI and CLI workflows for integration and repeatable runs

## Dashboard preview

When launched, the dashboard opens on the synthetic demo by default. It includes:

1. **Executive snapshot:** assortment size, data volume, reorder queue, excess-stock review count and forecast horizon.
2. **Command center:** recommended action mix, available stock against target / reorder thresholds, and priority queue.
3. **Recommendation workbench:** filter by action and category, search SKUs, inspect the reasoning and download filtered output.
4. **Demand forecast studio:** browse observed daily demand and a recursively generated forecast for each product.
5. **Model lab:** holdout MAE / WAPE, seasonal baseline comparison, feature importance, model metadata and one-click retraining.
6. **Data explorer:** browse and download the 10,000-row sales file, product catalog, inventory snapshot and supplier terms.

## Quick start

### 1. Requirements

- Python 3.10 or later
- About 1 GB of available memory is a comfortable starting point for local development
- Dependencies listed in `requirements.txt`

### 2. Create an environment and install packages

**Windows PowerShell**

```powershell
cd inventory-intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd inventory-intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, commonly `http://localhost:8501`. The bundled model and data are loaded automatically. On Windows, you can also launch `run_demo.bat`; on macOS / Linux, run `./run_demo.sh` to create an environment, install dependencies and start the app.

### 4. Optional: regenerate the dataset and train the model again

```bash
python data/generate_demo_data.py
python train_demo_model.py
```

This deterministically regenerates the four sample CSV files using a fixed random seed and then trains the global model. The model artifact is saved to `models/retail_demand_model.joblib`; evaluation metrics are saved to `outputs/model_metrics.json`.

### 5. Optional: run the API or CLI

```bash
uvicorn inventory_intelligence.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Run the standard-data CLI:

```bash
python -m inventory_intelligence.cli \
  --sales data/sample_sales.csv \
  --inventory data/sample_inventory.csv \
  --products data/sample_products.csv \
  --suppliers data/sample_suppliers.csv \
  --horizon 30 \
  --service-level 0.95 \
  --output outputs/recommendations.json \
  --csv outputs/recommendations.csv
```

### 6. Run tests

```bash
pytest -q
```

## Dataset design

The generated dataset has one row for each SKU-day combination: 20 SKUs × 500 days = exactly 10,000 rows. The date range is set relative to the day the generator runs, ending on the previous day.

### Included files

| File | Contents |
|---|---|
| `data/sample_sales.csv` | 10,000 rows of daily sales, product and warehouse IDs, selling price, promotion flag and holiday/event flag |
| `data/sample_products.csv` | Product name, category, warehouse, price, cost, pack size, MOQ, shelf-life field and synthetic demand archetype |
| `data/sample_inventory.csv` | One synthetic stock snapshot per SKU: on-hand, reserved and on-order stock |
| `data/sample_suppliers.csv` | Synthetic supplier IDs, lead time and review period |
| `data/generate_demo_data.py` | Reproducible generator for all sample CSVs |

### Synthetic patterns included

The generator deliberately creates a mix of fast movers, steady movers, slower SKUs, intermittent-demand items and seasonal products. Demand is influenced by weekday patterns, annual seasonality, gradual trend, planned promotional cycles, calendar events, price changes and occasional demand spikes. The inventory snapshot has varied low-stock, healthy-cover and excess-stock situations so all three recommendation types are represented in the demo.

These patterns make the demo interesting; they are assumptions authored for this project, not observations from a real chain. The synthetic dataset is created to be realistic enough for a workflow demonstration, not to claim a statistical replica of any particular retailer.

## Model architecture

```text
Synthetic generator (20 SKU × 500 days)
                 │
                 ▼
        Data validation & joins
                 │
                 ▼
  Feature engineering for each SKU-day
  ├── SKU / warehouse / category codes
  ├── Price, promotion, holiday flags
  ├── Weekday / month / annual Fourier terms
  └── Lags 1 / 7 / 14 / 28; rolling means / volatility
                 │
                 ▼
   Global RandomForestRegressor (pooled SKUs)
                 │
        ┌────────┴─────────┐
        ▼                  ▼
28-day time holdout     Full-history model
MAE / RMSE / WAPE       models/retail_demand_model.joblib
                             │
                             ▼
          Recursive daily demand forecasts
                             │
                             ▼
 Inventory policy (lead demand + safety stock)
                             │
                             ▼
      Reorder / Hold / Promote recommendation
                             │
          ┌──────────────────┼─────────────────┐
          ▼                  ▼                 ▼
      Dashboard          JSON / CSV          PDF report
```

### Forecasting method

The demo uses a pooled Random Forest regressor rather than fitting 20 unrelated models. Each training row represents a product-day and includes:

- Encoded SKU, warehouse and category
- Selling price, promotion indicator and holiday/event indicator
- Day of week, day of month, month, annual sine and cosine terms
- Lagged demand for 1, 7, 14 and 28 days earlier
- Rolling 7-day and 28-day means and 28-day standard deviation
- Days since the start of the series

The model predicts non-negative demand. Multi-day forecasts are recursive: each prediction becomes an input lag for later forecast dates. The synthetic demo has a deterministic future promotion calendar, which the forecast generator knows in advance. In a real implementation, future promotion flags should come from an approved promotion plan, not from future actual sales.

### Training and evaluation discipline

The training script reserves the final 28 dates as a time-based holdout, fits a temporary model on earlier data, and evaluates recursive daily forecasts against the held-out sales. It reports MAE, RMSE, WAPE, bias and a seasonal-naive baseline. It then fits the bundled model on all available feature rows so the demo forecasts can use the full synthetic history.

The holdout is valid only for this generated dataset. Real deployments should evaluate on retailer history using rolling-origin splits and compare against simple baselines. WAPE is more informative than MAPE for demand series containing zeros, but it is still sensitive to the mix and volume of the evaluated SKUs.

## Inventory policy and recommendation rules

The inventory engine uses the forecast and the stock snapshot to calculate:

- **Available stock:** `max(on_hand - reserved, 0)`
- **Inventory position:** available stock + on-order quantity
- **Lead-time demand:** sum of the forecast over supplier lead time
- **Safety stock:** `z(service_level) × daily_demand_std × sqrt(lead_time_days)`
- **Reorder point:** lead-time demand + safety stock
- **Target stock:** expected demand across lead time + review period, plus safety stock
- **Projected excess:** `max(available_stock - target_stock, 0)`

Rules are intentionally explicit:

- **Reorder:** inventory position is at or below the reorder point. Quantity aims to restore stock to the target, rounded to pack size and minimum order quantity.
- **Promote:** stock materially exceeds the target level. A negative quantity indicates units to consider for markdown, bundles, transfer, or another clearance action.
- **Hold:** neither a reorder nor excess-stock threshold is triggered.

These are transparent decision-support heuristics. The safety-stock formula is an approximation, especially for intermittent, seasonal, trending, or highly correlated demand.

## Using your own data

Select **Upload your data** in the dashboard and upload four CSV or Excel files. Use the canonical schemas documented in [`docs/data_dictionary.md`](docs/data_dictionary.md):

- `sales.csv`: `date`, `product_id`, `units_sold` (optional `warehouse_id`, `promotion_flag`, `holiday_flag`, `unit_price`)
- `inventory.csv`: `product_id`, `on_hand` (optional `warehouse_id`, `on_order`, `reserved`)
- `products.csv`: `product_id` (optional `category`, `unit_cost`, `unit_price`, `pack_size`, `minimum_order_quantity`)
- `suppliers.csv`: `product_id`, `lead_time_days` (optional `supplier_id`, `review_period_days`)

The upload mode currently uses the modular per-SKU forecasting pipeline from `inventory_intelligence/service.py`. Do not simply change column names if the original meanings differ; confirm that units, return handling, store IDs and dates align first.

## Output format

Each inventory recommendation includes:

```json
{
  "product_id": "SKU-1001",
  "product_name": "Granola Snack Box",
  "category": "Grocery",
  "warehouse_id": "WH-NORTH",
  "recommendation_type": "Reorder",
  "quantity": 88,
  "timing": "Order today",
  "reasoning": "Inventory position is at or below reorder point...",
  "forecast_horizon_days": 30,
  "forecast_demand_units": 245.6,
  "forecast_method": "Pre-trained global Random Forest",
  "supplier_lead_time_days": 5,
  "metrics": {
    "available_stock": 27,
    "inventory_position": 27,
    "lead_time_demand": 42.1,
    "safety_stock": 25.2,
    "reorder_point": 67.3,
    "target_stock": 94.5,
    "days_of_cover": 3.3,
    "projected_excess": 0
  }
}
```

The values above illustrate the schema; generated quantities depend on the current data and selected settings. The full response includes run metadata, summary counts, model evaluation data and forecast series.

## Project structure

```text
inventory-intelligence/
├── app.py                                # Polished Streamlit dashboard
├── train_demo_model.py                   # Train / evaluate / serialize the model
├── run_demo.bat / run_demo.sh             # Convenience launchers
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   ├── sample_sales.csv                  # Exactly 10,000 daily SKU observations
│   ├── sample_inventory.csv
│   ├── sample_products.csv
│   ├── sample_suppliers.csv
│   ├── sample_payload.json               # Small API example payload
│   └── generate_demo_data.py
├── models/
│   └── retail_demand_model.joblib        # Pre-trained global Random Forest
├── inventory_intelligence/
│   ├── demo_model.py                     # Feature engineering / train / forecast
│   ├── demo_service.py                   # Demo pipeline and stock recommendations
│   ├── data.py                           # Uploaded-data validation
│   ├── forecasting.py                    # General fallback / per-SKU model path
│   ├── inventory.py                      # Inventory policy calculations
│   ├── recommendations.py                # Modular recommendation engine
│   ├── service.py                        # General input orchestration
│   ├── reporting.py                      # Executive PDF export
│   ├── api.py                            # FastAPI endpoints
│   └── cli.py                            # JSON / CSV CLI
├── docs/
│   ├── data_dictionary.md
│   ├── how_to_use.md
│   ├── technical_deep_dive.md
│   ├── use_cases.md
│   └── api_guide.md
├── tests/
└── outputs/
    ├── model_metrics.json                # Synthetic holdout metrics
    ├── demo_recommendations.json         # Example structured output
    ├── demo_recommendations.csv          # Example flat output
    └── demo_inventory_report.pdf         # Example executive report
```

## Algorithm Overview

Stockwise combines supervised machine learning regression with stochastic inventory control theory. The end-to-end decision-making algorithm executes through four sequential phases:

```text
1. Feature Extraction ──► 2. Recursive Demand ──► 3. Inventory Policy ──► 4. Decision
   & Encoding                Forecasting              Evaluation             Recommendation
   (Lags, Fourier,           (Pooled Random           (Safety Stock,         (Reorder, Hold,
    Rolling Stats)            Forest Regressor)        ROP, Target Stock)     or Promote)
```

---

### Step 1: Feature Engineering & Preprocessing
Historical sales and catalog records are processed into 19 predictive signals for each SKU-day:
- **Calendar & Seasonality Features:** Day-of-week, day-of-month, month, linear trend index (`days_since_start`), and annual cyclic Fourier terms ($\sin(2\pi \cdot \text{day}/365.25)$, $\cos(2\pi \cdot \text{day}/365.25)$).
- **Pricing & Promotions:** Current selling price, active promotional flag (`promotion_flag`), and holiday/event indicators.
- **Autoregressive Lags:** Historical demand values at $t-1$, $t-7$, $t-14$, and $t-28$ days to capture immediate demand momentum and day-of-week patterns.
- **Rolling Statistics:** 7-day and 28-day rolling demand averages ($\mu_7, \mu_{28}$) and rolling 28-day volatility/standard deviation ($\sigma_{28}$).
- **Entity Embeddings/Encoding:** SKU, warehouse zone, and product category identifiers.

### Step 2: Recursive Machine Learning Demand Forecasting
- **Pooled Random Forest Regressor:** Instead of fitting isolated models per SKU, a single global Random Forest regressor is trained across all products, sharing cross-SKU learnings and generalized promotional response dynamics.
- **Recursive Multi-Step Prediction:** To predict across an $H$-day forecast horizon:
  1. The model predicts demand $\hat{y}_{t}$ for day $t$.
  2. Non-negativity is enforced: $\hat{y}_{t} = \max(0, \hat{y}_{t})$.
  3. The predicted $\hat{y}_{t}$ is fed back as lag inputs (e.g., $t-1$ lag for day $t+1$) and used to update rolling averages for subsequent steps through $t + H$.

### Step 3: Stochastic Inventory Policy & Threshold Calculation
Using the generated demand forecasts and SKU supplier parameters, the algorithm computes key replenishment metrics:
1. **Available Stock ($I_{\text{avail}}$):**
   $$\text{Available Stock} = \max(\text{On-Hand} - \text{Reserved}, 0)$$
2. **Inventory Position ($IP$):**
   $$IP = I_{\text{avail}} + \text{On-Order}$$
3. **Lead-Time Demand ($LTD$):** Total forecasted sales during supplier replenishment lead time ($L$ days):
   $$LTD = \sum_{t=1}^{L} \hat{y}_{t}$$
4. **Safety Stock ($SS$):** Buffer against demand uncertainty and lead-time variability based on the desired service level $Z$ (e.g., $Z = 1.645$ for 95% service level) and historical daily demand standard deviation ($\sigma_d$):
   $$SS = Z \times \sigma_d \times \sqrt{L}$$
5. **Reorder Point ($ROP$):** Trigger threshold for initiating a replenishment purchase order:
   $$ROP = LTD + SS$$
6. **Target Stock Level ($S$):** Base stock target covering demand across the lead time plus the review period ($R$ days), plus safety stock:
   $$S = \sum_{t=1}^{L + R} \hat{y}_t + SS$$
7. **Days of Cover ($DoC$):** Current available inventory divided by the daily average forecast.

### Step 4: Rule-Based Action Prescription & Order Sizing
The algorithm applies deterministic decision logic to generate an actionable, explainable operational recommendation for each SKU:

- **`Reorder`** (Triggered when $IP \le ROP$):
  - *Recommended Quantity:* Raw replenishment quantity is $\Delta = S - IP$.
  - *Pack Size & MOQ Constraints:* The order quantity is rounded up to the nearest multiple of `pack_size` and bounded from below by the `minimum_order_quantity` (MOQ).
  - *Timing:* Flagged as `"Order today"` or flagged based on urgency ($DoC < L$).
- **`Promote`** (Triggered when $I_{\text{avail}} > S \times 1.25$ and $DoC > (L + R) \times 1.5$):
  - Flags substantial excess capital tie-up.
  - Suggests promotional bundling, temporary discounts, or cross-warehouse transfers.
  - Projected excess quantity: $\max(I_{\text{avail}} - S, 0)$.
- **`Hold`** (Triggered when $ROP < IP \le S \times 1.25$):
  - Inventory is in a healthy equilibrium. No purchase order or markdown action required.

---

## Contribution guidelines

1. Open an issue describing the proposed change or problem.
2. Add tests for new feature calculations, policy rules and validation behavior.
3. Run `pytest -q` before submitting a change.
4. Keep input assumptions and recommendation reasons explicit and auditable.
5. Do not commit real customer or retailer data, secrets, or large raw exports.

## Limitations and production roadmap

- This dataset is synthetic. It is useful for a demo, coding exercise and end-to-end architecture illustration, but not for estimating performance on real businesses.
- On-hand inventory and supplier terms are synthetic scenario values. Replace them with a fresh ERP/WMS snapshot and supplier receipt history before operational use.
- Future promotion flags assume a known promotion plan in this synthetic demo.
- Observed sales may understate true demand during stockouts. Production training needs availability or lost-sales flags.
- Inventory policy does not yet model expiry lots, supplier capacity, order calendars, substitution, transfer optimization, or purchase budget constraints.
- Add authentication, access control, logging, monitoring, alerting, drift detection, model registry, and business approval workflows before production deployment.

## License

The source code and project documentation are provided under the MIT License. See [`LICENSE`](LICENSE). All bundled sample data is synthetic and generated by the included script.
