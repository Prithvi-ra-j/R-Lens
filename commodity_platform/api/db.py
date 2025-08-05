import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Optional
from .models import CommodityPrice, AlertRule, Alert

class DatabaseHandler:
    def __init__(self, db_path: str = "/workspace/commodity_platform/data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
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
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id INTEGER NOT NULL,
                commodity TEXT NOT NULL,
                price REAL NOT NULL,
                condition TEXT NOT NULL,
                threshold REAL NOT NULL,
                triggered_at TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (rule_id) REFERENCES alert_rules (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_price(self, price: CommodityPrice):
        """Insert a new price record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prices (commodity, price, timestamp, source)
            VALUES (?, ?, ?, ?)
        ''', (price.commodity, price.price, price.timestamp.isoformat(), price.source))
        
        conn.commit()
        conn.close()
    
    def get_prices(self, commodity: Optional[str] = None, limit: int = 100) -> List[dict]:
        """Get price records"""
        conn = sqlite3.connect(self.db_path)
        
        if commodity:
            query = "SELECT * FROM prices WHERE commodity = ? ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(commodity, limit))
        else:
            query = "SELECT * FROM prices ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        return df.to_dict('records')
    
    def create_alert_rule(self, rule: AlertRule) -> int:
        """Create a new alert rule"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_rules (commodity, condition, threshold, active, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (rule.commodity, rule.condition, rule.threshold, rule.active, 
              datetime.now().isoformat()))
        
        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return rule_id
    
    def get_alert_rules(self, active_only: bool = True) -> List[dict]:
        """Get alert rules"""
        conn = sqlite3.connect(self.db_path)
        
        if active_only:
            query = "SELECT * FROM alert_rules WHERE active = 1"
        else:
            query = "SELECT * FROM alert_rules"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict('records')
    
    def insert_alert(self, alert: Alert):
        """Insert a new alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (rule_id, commodity, price, condition, threshold, triggered_at, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert.rule_id, alert.commodity, alert.price, alert.condition, 
              alert.threshold, alert.triggered_at.isoformat(), alert.message))
        
        conn.commit()
        conn.close()
    
    def get_alerts(self, limit: int = 50) -> List[dict]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        return df.to_dict('records')
    
    def get_historical_data(self, commodity: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for ML model"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT commodity, price, timestamp 
            FROM prices 
            WHERE commodity = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(commodity, days * 24))  # Assuming hourly data
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df