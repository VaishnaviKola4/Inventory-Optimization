from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def _features(series: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(series.index)
    frame = pd.DataFrame(index=idx)
    frame["dow"] = idx.dayofweek
    frame["day"] = idx.day
    frame["month"] = idx.month
    frame["dayofyear_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    frame["rolling_7"] = shifted.rolling(7, min_periods=1).mean()
    frame["rolling_28"] = shifted.rolling(28, min_periods=1).mean()
    return frame

def forecast_product(history: pd.DataFrame, horizon: int = 30, minimum_history_days: int = 45) -> dict:
    """Forecast one product's daily demand. Returns non-negative forecast and diagnostics."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    daily = (history.set_index("date")["units_sold"].sort_index()
             .groupby(level=0).sum())
    if daily.empty:
        return {"daily_forecast": [0.0] * horizon, "method": "no_history",
                "daily_mean": 0.0, "daily_std": 0.0}
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0).astype(float)
    method = "seasonal_naive_7d"
    future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    values = None

    if len(daily) >= minimum_history_days and daily.sum() > 0:
        try:
            X = _features(daily).iloc[28:]
            y = daily.iloc[28:]
            valid = X.notna().all(axis=1)
            X, y = X.loc[valid], y.loc[valid]
            if len(X) >= 30:
                model = RandomForestRegressor(
                    n_estimators=120, max_depth=10, min_samples_leaf=2,
                    random_state=42, n_jobs=-1
                )
                model.fit(X, y)
                extended = daily.copy()
                predictions = []
                for dt in future_dates:
                    f = _features(extended).iloc[[-1]].copy()
                    f.index = [dt]
                    pred = max(0.0, float(model.predict(f)[0]))
                    predictions.append(pred)
                    extended.loc[dt] = pred
                values = predictions
                method = "random_forest_lag_features"
        except (ValueError, IndexError):
            values = None

    if values is None:
        arr = daily.to_numpy(dtype=float)
        if len(arr) >= 14:
            recent_daily = daily.iloc[-min(56, len(daily)):]
            weekday_means = recent_daily.groupby(recent_daily.index.dayofweek).mean().to_dict()
            recent_mean = float(recent_daily.mean())
            values = [
                max(0.0, 0.7 * float(weekday_means.get(dt.dayofweek, recent_mean)) + 0.3 * recent_mean)
                for dt in future_dates
            ]
        else:
            recent = arr[-min(14, len(arr)):]
            mean = float(np.mean(recent)) if len(recent) else 0.0
            values = [max(0.0, mean)] * horizon
            method = "rolling_mean_fallback"

    residual = daily.diff().dropna()
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return {
        "daily_forecast": [round(float(v), 3) for v in values],
        "method": method,
        "daily_mean": round(float(np.mean(values)), 3),
        "daily_std": round(max(0.0, daily_std), 3),
        "history_days": int(len(daily)),
        "history_total_units": round(float(daily.sum()), 3),
    }

def forecast_all(sales: pd.DataFrame, horizon: int = 30, minimum_history_days: int = 45) -> dict:
    result = {}
    for product_id in sales["product_id"].astype(str).unique():
        hist = sales[sales["product_id"].astype(str) == product_id][["date", "units_sold"]]
        result[product_id] = forecast_product(hist, horizon, minimum_history_days)
    return result
