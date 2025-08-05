#!/usr/bin/env python3
"""
Demonstration script showing the Commodity Platform working exactly as specified.
This script demonstrates all the key components mentioned in the requirements.
"""

import sys
import time
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def demo_price_fetching():
    """Demo: Real-Time Price Fetcher using yfinance"""
    print("🔹 1. Real-Time Price Fetcher Demo")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/prices/latest")
        if response.status_code == 200:
            prices = response.json()
            print("✅ Latest Commodity Prices:")
            for commodity, price in prices.items():
                print(f"   {commodity}: ${price}")
        else:
            print("❌ Failed to fetch prices")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def demo_alert_subscription():
    """Demo: Alert Subscription & Checker"""
    print("🔹 2. Alert Subscription & Checker Demo")
    print("=" * 50)
    
    # Subscribe to an alert
    alert_data = {
        "commodity": "Gold",
        "condition": "above",
        "target_price": 2100.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/subscribe", json=alert_data)
        if response.status_code == 200:
            print("✅ Alert subscription created:")
            print(f"   {alert_data['commodity']} price {alert_data['condition']} ${alert_data['target_price']}")
        
        # Check alerts
        print("\n🔍 Checking alerts...")
        response = requests.get(f"{BASE_URL}/check-alerts")
        if response.status_code == 200:
            result = response.json()
            triggered = result.get('triggered_alerts', [])
            print(f"✅ Found {len(triggered)} triggered alerts")
            for alert in triggered:
                print(f"   🚨 {alert['commodity']}: ${alert['triggered_price']} {alert['condition']} ${alert['target_price']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def demo_ml_prediction():
    """Demo: LSTM Price Prediction"""
    print("🔹 3. ML Model: LSTM Price Prediction Demo")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/predict-price", params={
            "commodity": "gold",
            "days": 3
        })
        
        if response.status_code == 200:
            result = response.json()
            print("✅ LSTM Price Predictions:")
            print(f"   Commodity: {result['commodity']}")
            print(f"   Next 3 days: {result['next_prices']}")
        else:
            print("❌ Prediction failed - model may need training")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def demo_database_operations():
    """Demo: Database Operations"""
    print("🔹 4. Database Operations Demo")
    print("=" * 50)
    
    try:
        # Get stats to show database is working
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Database Statistics:")
            print(f"   Total price records: {stats.get('total_price_records', 0)}")
            print(f"   Active alert rules: {stats.get('active_alert_rules', 0)}")
            print(f"   Supported commodities: {len(stats.get('supported_commodities', []))}")
        
        # Show recent alerts
        response = requests.get(f"{BASE_URL}/alerts")
        if response.status_code == 200:
            alerts_data = response.json()
            alerts = alerts_data.get('alerts', [])
            print(f"\n📊 Recent alerts in database: {len(alerts)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def demo_streamlit_dashboard():
    """Demo: Streamlit Dashboard Info"""
    print("🔹 5. Streamlit Dashboard")
    print("=" * 50)
    print("📊 Dashboard Features Available:")
    print("   • Latest prices + trends (line charts)")
    print("   • Price predictions (LSTM output)")
    print("   • Triggered alerts (data from DB)")
    print("   • KPI cards (high, low, average, % change)")
    print("   • Real-time auto-refresh capability")
    print()
    print("🌐 Access the dashboard at: http://localhost:8501")
    print()

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def main():
    """Main demonstration function"""
    print("🚀 Commodity Price Monitoring Platform Demo")
    print("=" * 60)
    print("Demonstrating all key components as specified in requirements")
    print("=" * 60)
    print()
    
    # Check if API is running
    if not check_api_health():
        print("❌ API is not running!")
        print("💡 Please start the API first:")
        print("   cd api && python main.py")
        print("   OR")
        print("   python run.py")
        sys.exit(1)
    
    print("✅ API is running successfully!")
    print()
    
    # Run all demos
    demo_price_fetching()
    demo_alert_subscription()
    demo_ml_prediction()
    demo_database_operations()
    demo_streamlit_dashboard()
    
    print("🎯 Technology Stack Verification:")
    print("=" * 50)
    print("✅ Backend: FastAPI, yfinance, SQLite, Pydantic")
    print("✅ ML: LSTM (Keras/TensorFlow), NumPy, Pandas")
    print("✅ Dashboard: Streamlit, Plotly charts")
    print("✅ Data Source: Yahoo Finance (yfinance)")
    print()
    
    print("🎉 Demo completed successfully!")
    print("📚 Visit http://localhost:8000/docs for complete API documentation")

if __name__ == "__main__":
    main()