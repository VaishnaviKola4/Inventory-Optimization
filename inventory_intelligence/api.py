from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from .service import generate_recommendations
from .data import DataValidationError

app = FastAPI(
    title="AI Inventory Intelligence API",
    version="1.0.0",
    description="Demand forecasting and explainable inventory recommendations."
)

class RecommendationRequest(BaseModel):
    sales: list[dict[str, Any]] = Field(min_length=1)
    inventory: list[dict[str, Any]] = Field(min_length=1)
    products: list[dict[str, Any]] = Field(min_length=1)
    suppliers: list[dict[str, Any]] = Field(min_length=1)
    horizon: int = Field(default=30, ge=1, le=365)
    service_level: float = Field(default=0.95, ge=0.50, le=0.99)
    minimum_history_days: int = Field(default=45, ge=14, le=3650)

@app.get("/health")
def health():
    return {"status": "ok", "service": "inventory-intelligence"}

@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    try:
        return generate_recommendations(
            request.sales, request.inventory, request.products, request.suppliers,
            horizon=request.horizon, service_level=request.service_level,
            minimum_history_days=request.minimum_history_days
        )
    except (DataValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Recommendation generation failed") from exc
