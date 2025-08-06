from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import schedule
import time
from datetime import datetime

from api.models import CommodityPrice, AlertRule, Alert, PredictionRequest, PredictionResponse
from .db import DatabaseHandler
from .prices import PriceFetcher, get_latest_prices
from .alert_logic import AlertEngine
from .ml_model import CommodityPredictor

# Import analytics modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analytics.spark_etl import CommoditySparkETL
from analytics.scheduler import get_scheduler, start_scheduler

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

# Initialize analytics services
analytics_etl = None
etl_scheduler = None

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
    global etl_scheduler
    
    # Start background price monitoring
    asyncio.create_task(update_prices_background())
    
    # Start ETL scheduler
    try:
        etl_scheduler = start_scheduler()
        print("✅ ETL Scheduler started successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not start ETL scheduler: {e}")
        print("PySpark analytics will not be available")

@app.on_event("shutdown")
async def shutdown_event():
    global etl_scheduler
    
    # Stop ETL scheduler
    if etl_scheduler:
        try:
            etl_scheduler.stop()
            print("✅ ETL Scheduler stopped")
        except Exception as e:
            print(f"Warning: Error stopping ETL scheduler: {e}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Commodity Price Monitoring API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/prices/latest")
async def get_latest_prices_endpoint():
    """Get latest prices exactly as specified in requirements"""
    try:
        prices = get_latest_prices()
        return prices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/subscribe")
async def subscribe(alert: dict):
    """Subscribe to alert rule exactly as specified"""
    try:
        db_handler.save_alert(alert)
        return {"message": "Alert subscription created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-alerts")
async def check_alerts():
    """Check alerts exactly as specified"""
    try:
        current = get_latest_prices()
        alerts = db_handler.get_all_alerts()  # Use the new method
        triggered = []
        
        for alert_rule in alerts:
            commodity = alert_rule['commodity']
            condition = alert_rule['condition']
            target_price = alert_rule['target_price']
            
            # Check if commodity exists in current prices
            current_price = None
            for name, price in current.items():
                if name.lower() == commodity.lower() or commodity.lower() in name.lower():
                    current_price = price
                    break
            
            if current_price:
                triggered_alert = False
                if condition == "above" and current_price > target_price:
                    triggered_alert = True
                elif condition == "below" and current_price < target_price:
                    triggered_alert = True
                
                if triggered_alert:
                    triggered.append({
                        "commodity": commodity,
                        "target_price": target_price,
                        "condition": condition,
                        "triggered_price": current_price,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Log triggered alerts using the new method
        db_handler.log_triggered(triggered)
        
        return {"triggered_alerts": triggered}
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

@app.get("/predict-price")
async def predict_price(commodity: str, days: int = 3):
    """Predict price exactly as specified in requirements using PyTorch"""
    try:
        # Get historical prices for the commodity
        from .ml_model import get_historical_prices, predict_next_prices
        
        history = get_historical_prices(commodity, days=60)
        
        if len(history) < 10:
            raise HTTPException(status_code=400, detail="Insufficient historical data for prediction")
        
        # Use the PyTorch ML model to predict
        next_prices = predict_next_prices(history, days)
        
        return {
            "commodity": commodity,
            "next_prices": next_prices,
            "framework": "PyTorch"
        }
        
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
        
        # Get analytics status
        analytics_status = "unavailable"
        analytics_path = "/workspace/commodity_platform/data/analytics"
        if os.path.exists(os.path.join(analytics_path, "overall_kpis")):
            analytics_status = "available"
        
        return {
            "total_price_records": total_prices,
            "supported_commodities": commodities,
            "active_alert_rules": alert_summary['active_rules_count'],
            "alerts_today": alert_summary['total_alerts_today'],
            "available_models": list(predictor.models.keys()),
            "analytics_status": analytics_status,
            "etl_scheduler_running": etl_scheduler.scheduler.running if etl_scheduler else False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints
@app.get("/analytics/{commodity}")
async def get_commodity_analytics(commodity: str, kpi_type: str = "overall"):
    """Get PySpark analytics KPIs for a specific commodity"""
    try:
        # Validate kpi_type
        valid_kpi_types = ["overall", "daily", "weekly", "monthly", "momentum"]
        if kpi_type not in valid_kpi_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid kpi_type. Must be one of: {valid_kpi_types}"
            )
        
        # Try to load analytics data using file system (faster than Spark for API)
        analytics_path = "/workspace/commodity_platform/data/analytics"
        parquet_path = os.path.join(analytics_path, f"{kpi_type}_kpis")
        
        if not os.path.exists(parquet_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Analytics data not found. Please run ETL job first: POST /analytics/etl/run"
            )
        
        # Load data using pandas for faster API response
        import pandas as pd
        
        # Read parquet files
        try:
            df = pd.read_parquet(parquet_path)
            
            # Filter for specific commodity
            commodity_data = df[df['commodity'].str.upper() == commodity.upper()]
            
            if commodity_data.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No analytics data found for commodity: {commodity}"
                )
            
            # Convert to dict and return
            data = commodity_data.to_dict('records')
            
            return {
                "commodity": commodity.upper(),
                "kpi_type": kpi_type,
                "data": data,
                "record_count": len(data),
                "generated_at": datetime.now().isoformat(),
                "source": "PySpark ETL"
            }
            
        except Exception as e:
            # Fallback to Spark if pandas fails
            global analytics_etl
            if analytics_etl is None:
                analytics_etl = CommoditySparkETL()
            
            result = analytics_etl.load_analytics_data(commodity, kpi_type)
            
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No analytics data found for {commodity}"
                )
            
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading analytics: {str(e)}")

@app.get("/analytics/{commodity}/summary")
async def get_commodity_analytics_summary(commodity: str):
    """Get comprehensive analytics summary for a commodity"""
    try:
        analytics_path = "/workspace/commodity_platform/data/analytics"
        
        # Check if analytics data exists
        if not os.path.exists(analytics_path):
            raise HTTPException(
                status_code=404,
                detail="Analytics data not available. Run ETL job first."
            )
        
        summary = {}
        kpi_types = ["overall", "daily", "weekly", "monthly", "momentum"]
        
        import pandas as pd
        
        for kpi_type in kpi_types:
            parquet_path = os.path.join(analytics_path, f"{kpi_type}_kpis")
            
            if os.path.exists(parquet_path):
                try:
                    df = pd.read_parquet(parquet_path)
                    commodity_data = df[df['commodity'].str.upper() == commodity.upper()]
                    
                    if not commodity_data.empty:
                        # Get latest record for summary
                        latest_record = commodity_data.iloc[-1].to_dict()
                        summary[kpi_type] = latest_record
                        
                except Exception as e:
                    summary[kpi_type] = {"error": str(e)}
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"No analytics data found for commodity: {commodity}"
            )
        
        return {
            "commodity": commodity.upper(),
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/etl/run")
async def run_etl_job(background_tasks: BackgroundTasks, num_records: int = 1000000):
    """Trigger PySpark ETL job manually"""
    try:
        def run_etl():
            """Background function to run ETL"""
            try:
                etl = CommoditySparkETL(app_name="ManualETL")
                success = etl.run_etl_pipeline(use_sample_data=True, num_records=num_records)
                
                # Log the result
                import json
                log_entry = {
                    "job": "manual_etl",
                    "status": "SUCCESS" if success else "FAILED",
                    "start_time": datetime.now().isoformat(),
                    "num_records": num_records,
                    "triggered_by": "API"
                }
                
                log_file = "/workspace/commodity_platform/data/analytics/manual_etl_log.json"
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                # Save log
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                else:
                    logs = []
                
                logs.append(log_entry)
                logs = logs[-50:]  # Keep last 50 entries
                
                with open(log_file, 'w') as f:
                    json.dump(logs, f, indent=2)
                    
            except Exception as e:
                print(f"ETL job failed: {e}")
        
        # Add to background tasks
        background_tasks.add_task(run_etl)
        
        return {
            "message": "ETL job started in background",
            "num_records": num_records,
            "estimated_duration": "2-5 minutes",
            "status_endpoint": "/analytics/etl/status"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/etl/status")
async def get_etl_status():
    """Get ETL job status and history"""
    try:
        global etl_scheduler
        
        status_info = {
            "scheduler_running": False,
            "scheduled_jobs": [],
            "manual_job_history": []
        }
        
        # Get scheduler status
        if etl_scheduler:
            scheduler_status = etl_scheduler.get_job_status()
            status_info["scheduler_running"] = scheduler_status.get("scheduler_running", False)
            status_info["scheduled_jobs"] = scheduler_status.get("jobs", [])
        
        # Get manual job history
        manual_log_file = "/workspace/commodity_platform/data/analytics/manual_etl_log.json"
        if os.path.exists(manual_log_file):
            with open(manual_log_file, 'r') as f:
                status_info["manual_job_history"] = json.load(f)
        
        # Get analytics files info
        analytics_path = "/workspace/commodity_platform/data/analytics"
        analytics_files = []
        
        if os.path.exists(analytics_path):
            for item in os.listdir(analytics_path):
                item_path = os.path.join(analytics_path, item)
                if os.path.isdir(item_path) and item.endswith('_kpis'):
                    # Get file stats
                    size = sum(
                        os.path.getsize(os.path.join(item_path, f))
                        for f in os.listdir(item_path)
                        if os.path.isfile(os.path.join(item_path, f))
                    )
                    analytics_files.append({
                        "name": item,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(item_path)
                        ).isoformat()
                    })
        
        status_info["analytics_files"] = analytics_files
        
        return status_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/commodities")
async def get_available_analytics_commodities():
    """Get list of commodities with available analytics data"""
    try:
        analytics_path = "/workspace/commodity_platform/data/analytics"
        overall_kpis_path = os.path.join(analytics_path, "overall_kpis")
        
        if not os.path.exists(overall_kpis_path):
            return {
                "commodities": [],
                "message": "No analytics data available. Run ETL job first."
            }
        
        # Read parquet to get available commodities
        import pandas as pd
        df = pd.read_parquet(overall_kpis_path)
        commodities = sorted(df['commodity'].unique().tolist())
        
        return {
            "commodities": commodities,
            "total_count": len(commodities),
            "available_kpi_types": ["overall", "daily", "weekly", "monthly", "momentum"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)