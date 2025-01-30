import os
import redis
from dotenv import load_dotenv

load_dotenv("../../config/.env")
r = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, db=0)

def get_latest_price(symbol: str) -> float:
    """Get latest price from Redis."""
    price = r.hget(f"stock:{symbol}", "price")
    return float(price) if price else 0.0