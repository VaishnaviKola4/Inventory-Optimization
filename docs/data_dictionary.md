# Data dictionary

CSV files are UTF-8. IDs should be stable strings. Dates are interpreted as calendar dates; keep timezone handling consistent in a real deployment. Sales, stock and suggested order quantities must share the same unit of measure (eaches or cases).

## `sample_sales.csv` / uploaded `sales.csv`

| Column | Required | Meaning |
|---|---|---|
| `date` | Yes | Date of observed demand, parseable as YYYY-MM-DD |
| `product_id` | Yes | Stable SKU identifier |
| `units_sold` | Yes | Non-negative units sold |
| `warehouse_id` | No | Store / warehouse / fulfillment location |
| `promotion_flag` | No | 1 where a promotion is active; otherwise 0 |
| `holiday_flag` | No | 1 for a modeled holiday/event date; otherwise 0 |
| `unit_price` | No | Observed selling price for that product-date |

The synthetic dataset has 10,000 rows: 20 SKUs × 500 daily observations. All four optional signals are populated. For uploaded data, duplicate daily rows are aggregated by the validation module. Returns, cancellations and stockout-censored sales should be treated consistently before training.

## `sample_inventory.csv` / uploaded `inventory.csv`

| Column | Required | Meaning |
|---|---|---|
| `snapshot_date` | No | Date when the stock snapshot was taken |
| `product_id` | Yes | SKU identifier |
| `warehouse_id` | No | Stock location |
| `on_hand` | Yes | Physical units currently held |
| `on_order` | No | Open purchase-order units not yet received |
| `reserved` | No | Units committed to orders / unavailable for general demand |

`inventory position = max(on_hand - reserved, 0) + on_order`.

## `sample_products.csv` / uploaded `products.csv`

| Column | Required | Meaning |
|---|---|---|
| `product_id` | Yes | Unique SKU identifier |
| `product_name` | No | Readable product name |
| `category` | No | Category used for grouping and dashboard filters |
| `warehouse_id` | No | Primary location used in the synthetic demo |
| `unit_cost` | No | Cost per unit, used to estimate stock value |
| `unit_price` | No | Regular or baseline selling price |
| `pack_size` | No | Supplier order multiple; default 1 |
| `minimum_order_quantity` | No | Minimum order quantity; default 1 |
| `shelf_life_days` | No | Nominal shelf life; expiry lots are not modeled yet |
| `base_daily_demand` | No | Synthetic generator parameter, not required for uploaded data |
| `demand_pattern` | No | Synthetic archetype, e.g. fast / seasonal / intermittent |

## `sample_suppliers.csv` / uploaded `suppliers.csv`

| Column | Required | Meaning |
|---|---|---|
| `product_id` | Yes | SKU identifier |
| `lead_time_days` | Yes | Expected replenishment lead time in calendar days |
| `supplier_id` | No | Supplier reference |
| `review_period_days` | No | Days until the next inventory review; defaults to 7 |

The bundled supplier values are generated assumptions. Replace them with actual purchase-order and goods-receipt history before making real replenishment decisions.

## Data quality checklist

- Distinguish a true zero-sales day from a missing data extract.
- Keep on-hand and open purchase orders current.
- Flag stockout periods because observed sales can understate latent demand.
- Ensure selling prices and future promotion calendars are available at forecast time.
- Check unit of measure and pack-size mappings before ordering.
- Never present synthetic stock or supplier fields as actual observations.
