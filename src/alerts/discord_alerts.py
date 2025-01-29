import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("../../../config/.env")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

class AlertManager:
    def __init__(self):
        self.cooldown = {}
    
    def send_alert(self, symbol: str, confidence: float, price: float):
        if self.cooldown.get(symbol, 0) > datetime.now().timestamp():
            return
        
        payload = {
            "content": f"🚨 **ALERT**: {symbol} @ ${price:.4f}",
            "embeds": [{
                "title": "Prediction Details",
                "fields": [
                    {"name": "Confidence", "value": f"{confidence:.2%}", "inline": True},
                    {"name": "RSI", "value": "70.5", "inline": True},
                    {"name": "Volume", "value": "5x Avg", "inline": True}
                ]
            }]
        }
        
        response = requests.post(WEBHOOK_URL, json=payload)
        if response.status_code == 204:
            self.cooldown[symbol] = datetime.now().timestamp() + 3600  # 1-hour cooldown