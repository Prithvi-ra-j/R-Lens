import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

def generate_realistic_price_data(commodity, start_price, start_date, end_date, volatility=0.02):
    """Generate realistic price data with trends and volatility"""
    
    # Create date range (hourly data)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    prices = []
    
    current_price = start_price
    trend_factor = np.random.uniform(-0.0001, 0.0001)  # Small trend
    
    for i, date in enumerate(dates):
        # Add trend
        current_price *= (1 + trend_factor)
        
        # Add random walk with volatility
        random_change = np.random.normal(0, volatility)
        current_price *= (1 + random_change)
        
        # Add some cyclical patterns (daily and weekly)
        hour_factor = 0.001 * np.sin(2 * np.pi * date.hour / 24)
        day_factor = 0.002 * np.sin(2 * np.pi * date.weekday() / 7)
        
        current_price *= (1 + hour_factor + day_factor)
        
        # Prevent negative prices
        current_price = max(current_price, start_price * 0.5)
        
        prices.append({
            'commodity': commodity.upper(),
            'price': round(current_price, 2),
            'timestamp': date.isoformat(),
            'source': 'Historical Data'
        })
    
    return prices

def create_sample_data():
    """Create sample historical data for all commodities"""
    
    # Define base prices and parameters
    commodities_config = {
        'gold': {'price': 2000.0, 'volatility': 0.015},
        'silver': {'price': 25.0, 'volatility': 0.025},
        'oil': {'price': 75.0, 'volatility': 0.030},
        'gas': {'price': 3.5, 'volatility': 0.040}
    }
    
    # Date range for historical data (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    all_data = []
    
    print("Generating sample historical data...")
    
    for commodity, config in commodities_config.items():
        print(f"  Generating data for {commodity.upper()}...")
        
        prices = generate_realistic_price_data(
            commodity=commodity,
            start_price=config['price'],
            start_date=start_date,
            end_date=end_date,
            volatility=config['volatility']
        )
        
        all_data.extend(prices)
        
        # Save individual CSV files
        df = pd.DataFrame(prices)
        csv_path = f"/workspace/commodity_platform/data/historical/{commodity}_historical.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved {len(prices)} records to {csv_path}")
    
    # Save combined data to database
    db_path = "/workspace/commodity_platform/data.db"
    print(f"\nInserting data into database: {db_path}")
    
    # Create database if it doesn't exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT DEFAULT 'API'
        )
    ''')
    
    # Insert all data
    for record in all_data:
        cursor.execute('''
            INSERT INTO prices (commodity, price, timestamp, source)
            VALUES (?, ?, ?, ?)
        ''', (record['commodity'], record['price'], record['timestamp'], record['source']))
    
    conn.commit()
    conn.close()
    
    print(f"Inserted {len(all_data)} total records into database")
    
    # Create summary CSV
    summary_df = pd.DataFrame(all_data)
    summary_path = "/workspace/commodity_platform/data/historical/all_commodities_historical.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved combined data to {summary_path}")
    
    return all_data

def create_sample_alerts():
    """Create some sample alert rules"""
    
    db_path = "/workspace/commodity_platform/data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create alert_rules table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alert_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity TEXT NOT NULL,
            condition TEXT NOT NULL,
            threshold REAL NOT NULL,
            active BOOLEAN DEFAULT 1,
            created_at TEXT NOT NULL
        )
    ''')
    
    # Sample alert rules
    sample_rules = [
        ('GOLD', 'above', 2100.0),
        ('GOLD', 'below', 1900.0),
        ('SILVER', 'above', 30.0),
        ('SILVER', 'below', 20.0),
        ('OIL', 'above', 85.0),
        ('OIL', 'below', 65.0),
        ('GAS', 'above', 4.0),
        ('GAS', 'below', 3.0)
    ]
    
    for commodity, condition, threshold in sample_rules:
        cursor.execute('''
            INSERT INTO alert_rules (commodity, condition, threshold, active, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (commodity, condition, threshold, True, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    
    print(f"Created {len(sample_rules)} sample alert rules")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("/workspace/commodity_platform/data/historical", exist_ok=True)
    
    # Generate sample data
    data = create_sample_data()
    
    # Create sample alert rules
    create_sample_alerts()
    
    print("\n✅ Sample data generation completed!")
    print("\nGenerated files:")
    print("  - /workspace/commodity_platform/data.db (SQLite database)")
    print("  - /workspace/commodity_platform/data/historical/*.csv (CSV files)")
    print(f"\nTotal records: {len(data)}")
    
    # Show summary statistics
    df = pd.DataFrame(data)
    print("\nData summary by commodity:")
    summary = df.groupby('commodity').agg({
        'price': ['count', 'mean', 'min', 'max', 'std']
    }).round(2)
    print(summary)