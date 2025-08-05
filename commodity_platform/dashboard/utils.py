import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def get_current_prices(self) -> Dict:
        """Get current prices for all commodities"""
        try:
            response = requests.get(f"{self.base_url}/prices/current")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching current prices: {e}")
            return {}
    
    def get_price_history(self, commodity: str, limit: int = 100) -> Dict:
        """Get price history for a commodity"""
        try:
            response = requests.get(f"{self.base_url}/prices/history/{commodity}?limit={limit}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching price history for {commodity}: {e}")
            return {}
    
    def get_alerts(self, limit: int = 50) -> Dict:
        """Get recent alerts"""
        try:
            response = requests.get(f"{self.base_url}/alerts?limit={limit}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return {"alerts": []}
    
    def get_alert_rules(self) -> Dict:
        """Get alert rules"""
        try:
            response = requests.get(f"{self.base_url}/alerts/rules")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching alert rules: {e}")
            return {"rules": []}
    
    def create_alert_rule(self, commodity: str, condition: str, threshold: float) -> Dict:
        """Create a new alert rule"""
        try:
            response = requests.post(
                f"{self.base_url}/alerts/rules",
                params={
                    "commodity": commodity,
                    "condition": condition,
                    "threshold": threshold
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error creating alert rule: {e}")
            return {}
    
    def get_predictions(self, commodity: str, days_ahead: int = 7) -> Dict:
        """Get price predictions for a commodity"""
        try:
            response = requests.post(f"{self.base_url}/predictions/{commodity}?days_ahead={days_ahead}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching predictions for {commodity}: {e}")
            return {}
    
    def get_stats(self) -> Dict:
        """Get platform statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching stats: {e}")
            return {}
    
    def train_model(self, commodity: str) -> Dict:
        """Train model for a commodity"""
        try:
            response = requests.post(f"{self.base_url}/models/train/{commodity}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error training model for {commodity}: {e}")
            return {}

class ChartGenerator:
    @staticmethod
    def create_price_chart(price_data: List[Dict], commodity: str, predictions: Optional[List[Dict]] = None):
        """Create a price chart with optional predictions"""
        if not price_data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create main price trace
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name=f'{commodity.upper()} Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions if available
        if predictions:
            pred_dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions]
            pred_prices = [p['predicted_price'] for p in predictions]
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'{commodity.upper()} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_current_prices_chart(current_prices: Dict):
        """Create a bar chart of current prices"""
        if not current_prices:
            return None
        
        commodities = []
        prices = []
        sources = []
        
        for commodity, price_data in current_prices.items():
            commodities.append(commodity.upper())
            prices.append(price_data['price'])
            sources.append(price_data['source'])
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        fig = go.Figure(data=[
            go.Bar(
                x=commodities,
                y=prices,
                text=[f'${p:.2f}' for p in prices],
                textposition='auto',
                marker_color=colors[:len(commodities)],
                hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<br>Source: %{customdata}<extra></extra>',
                customdata=sources
            )
        ])
        
        fig.update_layout(
            title='Current Commodity Prices',
            xaxis_title='Commodity',
            yaxis_title='Price ($)',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_alerts_timeline(alerts: List[Dict]):
        """Create a timeline of recent alerts"""
        if not alerts:
            return None
        
        df = pd.DataFrame(alerts)
        df['triggered_at'] = pd.to_datetime(df['triggered_at'])
        df = df.sort_values('triggered_at')
        
        # Create color mapping
        color_map = {'above': 'red', 'below': 'orange'}
        colors = [color_map.get(condition, 'blue') for condition in df['condition']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['triggered_at'],
            y=df['commodity'],
            mode='markers',
            marker=dict(
                size=15,
                color=colors,
                symbol='circle'
            ),
            text=df['message'],
            hovertemplate='<b>%{y}</b><br>%{text}<br>%{x}<extra></extra>',
            name='Alerts'
        ))
        
        fig.update_layout(
            title='Recent Alerts Timeline',
            xaxis_title='Time',
            yaxis_title='Commodity',
            height=300,
            showlegend=False
        )
        
        return fig

def format_price(price: float) -> str:
    """Format price with appropriate currency symbol"""
    return f"${price:,.2f}"

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return timestamp_str

def calculate_price_change(current: float, previous: float) -> tuple:
    """Calculate price change and percentage"""
    change = current - previous
    change_pct = (change / previous) * 100 if previous != 0 else 0
    return change, change_pct

def get_status_color(condition: str) -> str:
    """Get color for alert condition"""
    color_map = {
        'above': '#ff4444',
        'below': '#ff8800',
        'active': '#44ff44',
        'inactive': '#888888'
    }
    return color_map.get(condition.lower(), '#000000')