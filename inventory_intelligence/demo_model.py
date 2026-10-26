from __future__ import annotations

from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "product_code", "warehouse_code", "category_code", "price",
    "promotion_flag", "holiday_flag", "day_of_week", "day_of_month", "month",
    "dayofyear_sin", "dayofyear_cos", "trend_day",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_7", "rolling_28", "rolling_std_28",
]


def _holiday_flag(dt: pd.Timestamp) -> int:
    dt = pd.Timestamp(dt)
    thanksgiving = pd.Timestamp(year=dt.year, month=11, day=1)
    thanksgiving = thanksgiving + pd.Timedelta(days=(3 - thanksgiving.dayofweek) % 7 + 21)
    black_friday = thanksgiving + pd.Timedelta(days=1)
    cyber_monday = thanksgiving + pd.Timedelta(days=4)
    ranges = [
        (pd.Timestamp(dt.year, 1, 1), pd.Timestamp(dt.year, 1, 3)),
        (pd.Timestamp(dt.year, 2, 12), pd.Timestamp(dt.year, 2, 15)),
        (pd.Timestamp(dt.year, 7, 1), pd.Timestamp(dt.year, 7, 5)),
        (pd.Timestamp(dt.year, 8, 20), pd.Timestamp(dt.year, 9, 10)),
        (pd.Timestamp(dt.year, 11, 20), pd.Timestamp(dt.year, 11, 30)),
        (pd.Timestamp(dt.year, 12, 15), pd.Timestamp(dt.year, 12, 31)),
        (thanksgiving, thanksgiving), (black_friday, black_friday), (cyber_monday, cyber_monday),
    ]
    return int(any(a <= dt <= b for a, b in ranges))


def _promo_flag(dt: pd.Timestamp, sku_index: int, origin_date: pd.Timestamp) -> int:
    day_index = int((pd.Timestamp(dt).normalize() - pd.Timestamp(origin_date).normalize()).days)
    cycle = 47 + (sku_index % 5) * 3
    offset = (sku_index * 7 + 11) % cycle
    return int((day_index + offset) % cycle in {0, 1, 2, 3})


def _mappings(products: pd.DataFrame) -> dict:
    p = products.copy()
    sku_ids = sorted(p["product_id"].astype(str).unique().tolist())
    warehouse_ids = sorted(p["warehouse_id"].astype(str).unique().tolist())
    categories = sorted(p["category"].astype(str).unique().tolist())
    return {
        "product_codes": {x: i for i, x in enumerate(sku_ids)},
        "warehouse_codes": {x: i for i, x in enumerate(warehouse_ids)},
        "category_codes": {x: i for i, x in enumerate(categories)},
        "sku_order": sku_ids,
    }


def _feature_row(dt, pid: str, price: float, promo: int, holiday: int,
                 history_units: list[float], metadata: dict, maps: dict,
                 origin_date: pd.Timestamp) -> dict:
    dt = pd.Timestamp(dt)
    hist = list(history_units)
    def lag(n):
        return float(hist[-n]) if len(hist) >= n else (float(np.mean(hist)) if hist else 0.0)
    last7 = hist[-7:]
    last28 = hist[-28:]
    doy = dt.dayofyear
    return {
        "product_code": maps["product_codes"].get(str(pid), 0),
        "warehouse_code": maps["warehouse_codes"].get(str(metadata.get("warehouse_id", "WH-NORTH")), 0),
        "category_code": maps["category_codes"].get(str(metadata.get("category", "Other")), 0),
        "price": float(price), "promotion_flag": int(promo), "holiday_flag": int(holiday),
        "day_of_week": int(dt.dayofweek), "day_of_month": int(dt.day), "month": int(dt.month),
        "dayofyear_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "dayofyear_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "trend_day": int((dt.normalize() - pd.Timestamp(origin_date).normalize()).days),
        "lag_1": lag(1), "lag_7": lag(7), "lag_14": lag(14), "lag_28": lag(28),
        "rolling_7": float(np.mean(last7)) if last7 else 0.0,
        "rolling_28": float(np.mean(last28)) if last28 else 0.0,
        "rolling_std_28": float(np.std(last28, ddof=1)) if len(last28) > 1 else 0.0,
    }


def build_feature_frame(sales: pd.DataFrame, products: pd.DataFrame, maps: dict | None = None,
                        origin_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    sales = sales.copy()
    products = products.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    if maps is None:
        maps = _mappings(products)
    if origin_date is None:
        origin_date = sales["date"].min()
    origin_date = pd.Timestamp(origin_date)
    product_meta = products.set_index(products["product_id"].astype(str)).to_dict("index")
    rows = []
    for pid, group in sales.groupby(sales["product_id"].astype(str), sort=False):
        group = group.sort_values("date")
        md = product_meta.get(str(pid), {})
        history = []
        for row in group.itertuples(index=False):
            dt = pd.Timestamp(row.date)
            # The model always sees demand from dates before the row being predicted.
            if len(history) >= 28:
                feat = _feature_row(
                    dt, str(pid), float(getattr(row, "unit_price", md.get("unit_price", 0))),
                    int(getattr(row, "promotion_flag", 0)), int(getattr(row, "holiday_flag", 0)),
                    history, md, maps, origin_date
                )
                feat.update({"date": dt, "product_id": str(pid), "target": float(row.units_sold)})
                rows.append(feat)
            history.append(float(row.units_sold))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Not enough data to build ML features; provide at least 29 daily observations per SKU.")
    return frame, maps, origin_date


def _predict_recursive(model, pid: str, history_group: pd.DataFrame, future_rows: pd.DataFrame,
                       metadata: dict, maps: dict, origin_date: pd.Timestamp) -> list[float]:
    history_group = history_group.sort_values("date")
    history_units = history_group["units_sold"].astype(float).tolist()
    preds = []
    for row in future_rows.sort_values("date").itertuples(index=False):
        feat = _feature_row(
            pd.Timestamp(row.date), str(pid), float(getattr(row, "unit_price", metadata.get("unit_price", 0))),
            int(getattr(row, "promotion_flag", 0)), int(getattr(row, "holiday_flag", 0)),
            history_units, metadata, maps, origin_date
        )
        x = pd.DataFrame([[feat[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        pred = max(0.0, float(model.predict(x)[0]))
        preds.append(pred)
        history_units.append(pred)
    return preds


def _future_rows(pid: str, metadata: dict, start_date: pd.Timestamp, horizon: int,
                 sku_index: int, origin_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start_date) + pd.Timedelta(days=1), periods=horizon, freq="D")
    price = float(metadata.get("unit_price", 0.0))
    rows = []
    for dt in dates:
        promo = _promo_flag(dt, sku_index, origin_date)
        rows.append({"date": dt, "product_id": str(pid),
                     "unit_price": round(price * (0.92 if promo else 1.0), 2),
                     "promotion_flag": promo, "holiday_flag": _holiday_flag(dt)})
    return pd.DataFrame(rows)


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=140, max_depth=17, min_samples_leaf=2,
        max_features=0.9, random_state=42, n_jobs=-1
    )


def train_and_save(data_dir: str | Path, model_dir: str | Path, metrics_path: str | Path,
                   holdout_days: int = 28) -> dict:
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(data_dir / "sample_sales.csv", parse_dates=["date"])
    products = pd.read_csv(data_dir / "sample_products.csv")
    if len(sales) < 5000:
        raise ValueError(f"Expected at least 5,000 daily observations, received {len(sales):,}.")
    frame, maps, origin_date = build_feature_frame(sales, products)
    max_date = sales.date.max()
    cutoff = max_date - pd.Timedelta(days=holdout_days - 1)
    train_frame = frame[frame.date < cutoff]
    if train_frame.empty:
        raise ValueError("Training set is empty after applying the holdout window.")
    eval_model = _new_model()
    eval_model.fit(train_frame[FEATURE_COLUMNS], train_frame["target"])

    # True multi-step holdout: predictions become lag inputs for subsequent days.
    product_meta = products.set_index(products.product_id.astype(str)).to_dict("index")
    predictions, actuals, baseline_preds = [], [], []
    per_sku = []
    sku_index = maps["product_codes"]
    holdout_sales = sales[sales.date >= cutoff]
    for pid in maps["sku_order"]:
        hist = sales[(sales.product_id.astype(str) == pid) & (sales.date < cutoff)].sort_values("date")
        fut = holdout_sales[holdout_sales.product_id.astype(str) == pid].sort_values("date")
        if hist.empty or fut.empty:
            continue
        md = product_meta[pid]
        pred = _predict_recursive(eval_model, pid, hist, fut, md, maps, origin_date)
        actual = fut.units_sold.astype(float).tolist()
        baseline_history = hist.units_sold.astype(float).tolist()
        naive = []
        for _ in fut.itertuples(index=False):
            p = baseline_history[-7] if len(baseline_history) >= 7 else float(np.mean(baseline_history))
            p = max(0.0, float(p))
            naive.append(p)
            baseline_history.append(p)
        predictions.extend(pred)
        actuals.extend(actual)
        baseline_preds.extend(naive)
        per_sku.append({
            "product_id": pid,
            "mae": round(float(mean_absolute_error(actual, pred)), 3),
            "wape_percent": round(float(np.abs(np.array(actual)-np.array(pred)).sum()/max(np.sum(actual),1)*100), 2),
            "holdout_units": int(np.sum(actual)),
        })
    y = np.asarray(actuals, dtype=float)
    yp = np.asarray(predictions, dtype=float)
    yn = np.asarray(baseline_preds, dtype=float)
    if len(y) == 0:
        raise ValueError("No held-out observations were available for evaluation.")
    metrics = {
        "dataset": "Synthetic retail inventory demo dataset",
        "synthetic_data": True,
        "sales_rows": int(len(sales)),
        "sku_count": int(sales.product_id.nunique()),
        "warehouse_count": int(sales.warehouse_id.nunique()),
        "history_start": str(sales.date.min().date()),
        "history_end": str(sales.date.max().date()),
        "training_rows": int(len(train_frame)),
        "holdout_train_rows": int(len(train_frame)),
        "final_model_training_rows": int(len(frame)),
        "holdout_days": int(holdout_days),
        "holdout_observations": int(len(y)),
        "model": "RandomForestRegressor (global pooled SKU model)",
        "features": FEATURE_COLUMNS,
        "model_mae_units_per_day": round(float(mean_absolute_error(y, yp)), 3),
        "model_rmse_units_per_day": round(float(np.sqrt(mean_squared_error(y, yp))), 3),
        "model_wape_percent": round(float(np.abs(y-yp).sum()/max(y.sum(),1)*100), 2),
        "model_bias_units_per_day": round(float(np.mean(yp-y)), 3),
        "seasonal_naive_mae_units_per_day": round(float(mean_absolute_error(y, yn)), 3),
        "seasonal_naive_wape_percent": round(float(np.abs(y-yn).sum()/max(y.sum(),1)*100), 2),
        "sku_holdout_metrics": per_sku,
        "feature_importance": [],
        "note": "Evaluation is a synthetic-data holdout, not evidence of performance on a real retailer. The final model is retrained on all generated history for demo forecasts.",
    }
    final_model = _new_model()
    final_model.fit(frame[FEATURE_COLUMNS], frame["target"])
    importances = sorted(zip(FEATURE_COLUMNS, final_model.feature_importances_), key=lambda x: x[1], reverse=True)
    metrics["feature_importance"] = [{"feature": f, "importance": round(float(v), 5)} for f, v in importances]
    bundle = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "maps": maps,
        "origin_date": str(origin_date.date()),
        "model_name": "RandomForestRegressor (global pooled SKU model)",
        "trained_through": str(max_date.date()),
        "products": products.to_dict("records"),
    }
    joblib.dump(bundle, model_dir / "retail_demand_model.joblib", compress=3)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model_bundle(model_path: str | Path) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-trained demo model not found: {path}. Run python train_demo_model.py")
    return joblib.load(path)


def forecast_all_products(sales: pd.DataFrame, products: pd.DataFrame, bundle: dict,
                          horizon: int = 30) -> dict[str, dict]:
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    products = products.copy()
    products["product_id"] = products.product_id.astype(str)
    sales["product_id"] = sales.product_id.astype(str)
    meta_map = products.set_index("product_id").to_dict("index")
    maps = bundle["maps"]
    origin_date = pd.Timestamp(bundle["origin_date"])
    model = bundle["model"]
    results = {}
    max_date = sales.date.max()
    for pid in sorted(products.product_id.unique()):
        hist = sales[sales.product_id == pid].sort_values("date")
        if hist.empty:
            continue
        md = meta_map[pid]
        sku_idx = maps["product_codes"].get(pid, 0)
        future = _future_rows(pid, md, max_date, horizon, sku_idx, origin_date)
        preds = _predict_recursive(model, pid, hist, future, md, maps, origin_date)
        raw = hist.units_sold.astype(float).to_numpy()
        results[pid] = {
            "dates": [str(d.date()) for d in future.date],
            "daily_forecast": [round(float(v), 3) for v in preds],
            "forecast_demand_units": round(float(np.sum(preds)), 2),
            "daily_mean": round(float(np.mean(preds)) if preds else 0, 3),
            "daily_std": round(float(np.std(raw, ddof=1)) if len(raw) > 1 else 0, 3),
            "history_days": int(len(hist)),
            "history_total_units": round(float(np.sum(raw)), 2),
            "historical_daily_mean": round(float(np.mean(raw)), 3),
            "historical_daily_std": round(float(np.std(raw, ddof=1)) if len(raw) > 1 else 0, 3),
            "history_dates": [str(d.date()) for d in hist.tail(90).date],
            "history_units": [int(v) for v in hist.tail(90).units_sold],
            "forecast_method": "Pre-trained global Random Forest",
        }
    return results
