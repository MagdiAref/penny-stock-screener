import redis
import json
from typing import Dict, Optional
from dotenv import load_dotenv
import os
import time 

load_dotenv("../config/.env")

class RedisManager:
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.password = os.getenv("REDIS_PASSWORD")
        self.db = int(os.getenv("REDIS_DB", 0))
        
        self.connection = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=True,
            socket_timeout=5,
            health_check_interval=30
        )
    
    def cache_stock_data(self, symbol: str, data: Dict) -> bool:
        try:
            return self.connection.hset(f"stock:{symbol}", mapping=data)
        except redis.RedisError as e:
            print(f"Redis cache error: {str(e)}")
            return False
    
    def get_cached_data(self, symbol: str) -> Optional[Dict]:
        try:
            return self.connection.hgetall(f"stock:{symbol}")
        except redis.RedisError:
            return None
    
    def log_alert(self, symbol: str, alert_type: str, data: Dict):
        log_entry = {
            "timestamp": int(time.time()),
            "symbol": symbol,
            "type": alert_type,
            "data": data
        }
        self.connection.lpush("alerts", json.dumps(log_entry))
    
    def flush_all(self):
        self.connection.flushall()