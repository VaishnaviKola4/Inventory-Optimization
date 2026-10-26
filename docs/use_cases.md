# Use cases

## 1. Replenishment planning for fast-moving goods

A retailer sells high-velocity snacks and personal-care products. The system forecasts demand over the supplier lead time, adds a safety-stock buffer, and signals Reorder when inventory position is at or below the reorder point. A planner reviews quantity, pack size, supplier timing and current purchase orders before approving a replenishment.

## 2. Seasonal assortment planning

Products such as jackets and seasonal home goods exhibit annual demand variation. The sample data includes annual seasonal features and event signals, which can be used to demonstrate forecasting and early replenishment. For real planning, use several years of history where possible and include planned events, weather or other relevant context if they are reliable and available at decision time.

## 3. Excess-stock and working-capital review

A slow-moving item may have more stock than its projected demand and target buffer require. The Promote action quantifies estimated excess units for consideration in markdowns, bundles, vendor returns or transfers. A human planner should check margin, season-end date and commercial commitments before acting.

## 4. Promotional demand scenarios

The demo uses scheduled synthetic promotion periods and a planned event calendar. It illustrates how promotions may lift demand, and the forecast pipeline uses known-ahead promo/event flags. Production systems should provide an approved future campaign calendar; future actual sales must never leak into model inputs.

## 5. Multi-location inventory review

The included sample maps products to four warehouse zones. This release creates inventory signals at the product-location level for the supplied catalog. A complete transfer optimizer would also need source stock, destination stock, in-transit quantities, transfer lead time/cost, source protection stock and destination demand.

## 6. New product launch

A newly launched product may have too little history for reliable lags or seasonal patterns. Use analog products, category-level priors, merchandising plans and conservative min/max controls, then begin monitoring backtest accuracy once meaningful demand history accumulates. The bundled pretrained model is only appropriate for the bundled synthetic products and IDs; uploaded data uses the general per-SKU pipeline.
