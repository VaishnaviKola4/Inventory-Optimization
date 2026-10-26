from __future__ import annotations
import math
from statistics import NormalDist

def inventory_metrics(on_hand: float, on_order: float, reserved: float,
                      daily_forecast: list[float], lead_time_days: int,
                      review_period_days: int = 7, daily_std: float = 0.0,
                      service_level: float = 0.95) -> dict:
    if not 0.5 <= service_level <= 0.99:
        raise ValueError("service_level must be between 0.50 and 0.99")
    on_hand = max(0.0, float(on_hand))
    on_order = max(0.0, float(on_order))
    reserved = max(0.0, float(reserved))
    available = max(0.0, on_hand - reserved)
    position = available + on_order
    lead = max(0, int(math.ceil(lead_time_days)))
    review = max(1, int(math.ceil(review_period_days)))
    daily_mean = sum(daily_forecast) / len(daily_forecast) if daily_forecast else 0.0
    lead_demand = sum(daily_forecast[:lead]) if lead else 0.0
    protection_days = lead + review
    protection_demand = sum(daily_forecast[:min(protection_days, len(daily_forecast))])
    if protection_days > len(daily_forecast):
        protection_demand += max(0, protection_days - len(daily_forecast)) * daily_mean
    z = NormalDist().inv_cdf(service_level)
    safety_stock = z * max(0.0, daily_std) * math.sqrt(max(1, lead))
    reorder_point = lead_demand + safety_stock
    target_stock = protection_demand + safety_stock
    return {
        "available_stock": round(available, 2),
        "inventory_position": round(position, 2),
        "daily_forecast_mean": round(daily_mean, 3),
        "lead_time_demand": round(lead_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "target_stock": round(target_stock, 2),
        "days_of_cover": round(available / daily_mean, 1) if daily_mean > 0 else None,
        "projected_excess": round(max(0.0, available - target_stock), 2),
    }
