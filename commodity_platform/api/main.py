from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import schedule
import time
from datetime import datetime

from .models import CommodityPrice, AlertRule, Alert, PredictionRequest, PredictionResponse
from .db import DatabaseHandler
from .prices import PriceFetcher
from .alert_logic import AlertEngine
from .ml_model import CommodityPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Commodity Price Monitoring API",
    description="API for monitoring commodity prices, managing alerts, and price predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_handler = DatabaseHandler()
price_fetcher = PriceFetcher()
alert_engine = AlertEngine(db_handler)
predictor = CommodityPredictor()

# Background task for periodic price updates
async def update_prices_background():
    """Background task to periodically fetch and process prices"""
    while True:
        try:
            prices = price_fetcher.fetch_all_prices()
            
            for commodity, price in prices.items():
                # Process price update (store in DB and check alerts)
                alerts = alert_engine.process_price_update(price)
                
                if alerts:
                    print(f"Triggered {len(alerts)} alerts for {commodity}")
            
        except Exception as e:
            print(f"Error in background price update: {e}")
        
        # Wait 5 minutes before next update
        await asyncio.sleep(300)

# Start background task on startup
@app.on_event("startup")
async def startup_event():
    # Start background price monitoring
    asyncio.create_task(update_prices_background())

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Commodity Price Monitoring API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/prices/current", response_model=Dict[str, CommodityPrice])
async def get_current_prices():
    """Get current prices for all commodities"""
    try:
        prices = price_fetcher.fetch_all_prices()
        return prices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prices/{commodity}")
async def get_commodity_price(commodity: str):
    """Get current price for a specific commodity"""
    try:
        price = price_fetcher.fetch_price(commodity)
        if price:
            return price
        else:
            raise HTTPException(status_code=404, detail=f"Price not found for {commodity}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prices/history/{commodity}")
async def get_price_history(commodity: str, limit: int = 100):
    """Get price history for a commodity"""
    try:
        prices = db_handler.get_prices(commodity=commodity, limit=limit)
        return {"commodity": commodity, "prices": prices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/rules")
async def create_alert_rule(commodity: str, condition: str, threshold: float):
    """Create a new alert rule"""
    try:
        if condition.lower() not in ["above", "below"]:
            raise HTTPException(status_code=400, detail="Condition must be 'above' or 'below'")
        
        rule_id = alert_engine.create_alert_rule(commodity, condition, threshold)
        return {"rule_id": rule_id, "message": "Alert rule created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/rules")
async def get_alert_rules():
    """Get all alert rules"""
    try:
        rules = db_handler.get_alert_rules(active_only=False)
        return {"rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    try:
        alerts = db_handler.get_alerts(limit=limit)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/summary")
async def get_alert_summary():
    """Get alert summary"""
    try:
        summary = alert_engine.get_alert_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictions/{commodity}")
async def predict_commodity_prices(commodity: str, days_ahead: int = 7):
    """Predict future prices for a commodity"""
    try:
        if days_ahead < 1 or days_ahead > 30:
            raise HTTPException(status_code=400, detail="Days ahead must be between 1 and 30")
        
        predictions = predictor.predict_prices(commodity, days_ahead)
        
        if predictions:
            return predictions
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for {commodity} or insufficient data"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/train/{commodity}")
async def train_model(commodity: str, background_tasks: BackgroundTasks):
    """Train ML model for a specific commodity"""
    try:
        # Get historical data
        df = db_handler.get_historical_data(commodity, days=200)
        
        if len(df) < 70:  # Need enough data for training
            raise HTTPException(
                status_code=400, 
                detail="Insufficient historical data for training"
            )
        
        # Train model (this might take a while, so we could make it a background task)
        result = predictor.train_model(commodity, df)
        
        if result['success']:
            return {
                "message": f"Model trained successfully for {commodity}",
                "metrics": result
            }
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/retrain-all")
async def retrain_all_models():
    """Retrain all ML models"""
    try:
        results = predictor.retrain_all_models(db_handler)
        return {"training_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        prices = db_handler.get_prices(limit=1)
        
        # Test price fetching
        test_price = price_fetcher.fetch_price("gold")
        
        return {
            "status": "healthy",
            "database": "connected",
            "price_fetcher": "working" if test_price else "limited",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stats")
async def get_stats():
    """Get platform statistics"""
    try:
        all_prices = db_handler.get_prices(limit=1000)
        alert_summary = alert_engine.get_alert_summary()
        
        # Calculate some basic stats
        commodities = list(set([p['commodity'] for p in all_prices]))
        total_prices = len(all_prices)
        
        return {
            "total_price_records": total_prices,
            "supported_commodities": commodities,
            "active_alert_rules": alert_summary['active_rules_count'],
            "alerts_today": alert_summary['total_alerts_today'],
            "available_models": list(predictor.models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)