import requests
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
from .models import CommodityPrice

# Ticker mapping as specified in requirements
TICKERS = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Silver": "SI=F"
}

def get_latest_prices():
    """Get latest prices exactly as specified in requirements"""
    prices = {}
    for name, ticker in TICKERS.items():
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                prices[name] = round(data["Close"].iloc[-1], 2)
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    return prices

class PriceFetcher:
    def __init__(self):
        # Mapping of commodity names to Yahoo Finance symbols
        self.symbols = {
            "gold": "GC=F",      # Gold futures
            "silver": "SI=F",    # Silver futures
            "oil": "CL=F",       # Crude Oil futures
            "gas": "NG=F",       # Natural Gas futures
            "brent": "BZ=F"      # Brent Oil futures
        }
        
        # Alternative API endpoints for backup
        self.alternative_endpoints = {
            "gold": "https://api.metals.live/v1/spot/gold",
            "silver": "https://api.metals.live/v1/spot/silver"
        }
    
    def fetch_yahoo_price(self, commodity: str) -> Optional[CommodityPrice]:
        """Fetch price from Yahoo Finance"""
        try:
            symbol = self.symbols.get(commodity.lower())
            if not symbol:
                return None
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
            
            latest_price = float(data['Close'].iloc[-1])
            
            return CommodityPrice(
                commodity=commodity.upper(),
                price=latest_price,
                timestamp=datetime.now(),
                source="Yahoo Finance"
            )
        
        except Exception as e:
            print(f"Error fetching {commodity} from Yahoo Finance: {e}")
            return None
    
    def fetch_metals_api_price(self, commodity: str) -> Optional[CommodityPrice]:
        """Fetch price from alternative metals API"""
        try:
            endpoint = self.alternative_endpoints.get(commodity.lower())
            if not endpoint:
                return None
            
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data.get('price', 0))
            
            if price > 0:
                return CommodityPrice(
                    commodity=commodity.upper(),
                    price=price,
                    timestamp=datetime.now(),
                    source="Metals API"
                )
            
        except Exception as e:
            print(f"Error fetching {commodity} from Metals API: {e}")
        
        return None
    
    def fetch_price(self, commodity: str) -> Optional[CommodityPrice]:
        """Fetch price with fallback to alternative sources"""
        # Try Yahoo Finance first
        price = self.fetch_yahoo_price(commodity)
        if price:
            return price
        
        # Fallback to alternative API for gold/silver
        price = self.fetch_metals_api_price(commodity)
        if price:
            return price
        
        # Generate mock data if all sources fail (for development)
        return self._generate_mock_price(commodity)
    
    def _generate_mock_price(self, commodity: str) -> CommodityPrice:
        """Generate mock price data for development/testing"""
        import random
        
        base_prices = {
            "gold": 2000.0,
            "silver": 25.0,
            "oil": 75.0,
            "gas": 3.5,
            "brent": 80.0
        }
        
        base_price = base_prices.get(commodity.lower(), 100.0)
        # Add some realistic variation (±2%)
        variation = random.uniform(-0.02, 0.02)
        mock_price = base_price * (1 + variation)
        
        return CommodityPrice(
            commodity=commodity.upper(),
            price=round(mock_price, 2),
            timestamp=datetime.now(),
            source="Mock Data"
        )
    
    def fetch_all_prices(self) -> Dict[str, CommodityPrice]:
        """Fetch prices for all supported commodities"""
        commodities = ["gold", "silver", "oil", "gas"]
        prices = {}
        
        for commodity in commodities:
            price = self.fetch_price(commodity)
            if price:
                prices[commodity] = price
        
        return prices