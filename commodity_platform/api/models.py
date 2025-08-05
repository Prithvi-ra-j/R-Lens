from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class CommodityPrice(BaseModel):
    commodity: str
    price: float
    timestamp: datetime
    source: str = "API"

class AlertRule(BaseModel):
    id: Optional[int] = None
    commodity: str
    condition: str  # "above" or "below"
    threshold: float
    active: bool = True
    created_at: Optional[datetime] = None

class Alert(BaseModel):
    id: Optional[int] = None
    rule_id: int
    commodity: str
    price: float
    condition: str
    threshold: float
    triggered_at: datetime
    message: str

class PredictionRequest(BaseModel):
    commodity: str
    days_ahead: int = 7

class PredictionResponse(BaseModel):
    commodity: str
    predictions: List[dict]  # [{"date": "2024-01-01", "predicted_price": 1850.0}, ...]
    confidence_score: float
    model_accuracy: float