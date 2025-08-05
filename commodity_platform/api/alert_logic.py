from datetime import datetime
from typing import List
from .models import CommodityPrice, AlertRule, Alert
from .db import DatabaseHandler

class AlertEngine:
    def __init__(self, db_handler: DatabaseHandler):
        self.db = db_handler
    
    def check_alerts(self, price: CommodityPrice) -> List[Alert]:
        """Check if current price triggers any alerts"""
        triggered_alerts = []
        
        # Get active alert rules for this commodity
        rules = self.db.get_alert_rules(active_only=True)
        commodity_rules = [rule for rule in rules if rule['commodity'].upper() == price.commodity.upper()]
        
        for rule in commodity_rules:
            alert = self._evaluate_rule(rule, price)
            if alert:
                triggered_alerts.append(alert)
                # Log the alert to database
                self.db.insert_alert(alert)
        
        return triggered_alerts
    
    def _evaluate_rule(self, rule: dict, price: CommodityPrice) -> Alert:
        """Evaluate if a single rule is triggered"""
        condition = rule['condition'].lower()
        threshold = rule['threshold']
        current_price = price.price
        
        triggered = False
        
        if condition == "above" and current_price > threshold:
            triggered = True
        elif condition == "below" and current_price < threshold:
            triggered = True
        
        if triggered:
            message = f"{price.commodity} price ${current_price:.2f} is {condition} threshold ${threshold:.2f}"
            
            return Alert(
                rule_id=rule['id'],
                commodity=price.commodity,
                price=current_price,
                condition=condition,
                threshold=threshold,
                triggered_at=datetime.now(),
                message=message
            )
        
        return None
    
    def process_price_update(self, price: CommodityPrice):
        """Process a new price update - check alerts and store price"""
        # Store the price
        self.db.insert_price(price)
        
        # Check for triggered alerts
        alerts = self.check_alerts(price)
        
        return alerts
    
    def create_alert_rule(self, commodity: str, condition: str, threshold: float) -> int:
        """Create a new alert rule"""
        rule = AlertRule(
            commodity=commodity.upper(),
            condition=condition.lower(),
            threshold=threshold,
            active=True
        )
        
        return self.db.create_alert_rule(rule)
    
    def get_alert_summary(self) -> dict:
        """Get summary of recent alerts"""
        recent_alerts = self.db.get_alerts(limit=10)
        active_rules = self.db.get_alert_rules(active_only=True)
        
        return {
            "recent_alerts": recent_alerts,
            "active_rules_count": len(active_rules),
            "total_alerts_today": len([a for a in recent_alerts 
                                     if datetime.fromisoformat(a['triggered_at']).date() == datetime.now().date()])
        }