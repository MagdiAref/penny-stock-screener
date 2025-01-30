import asyncio
import os
import requests
import redis
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv("../../config/.env")
r = redis.Redis(host=os.getenv("REDIS_HOST"), port=6379, db=0)

def get_active_penny_stocks():
    """Fetch stocks under $5 with recent volume >1M using Polygon API."""
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&sort=ticker&limit=1000&apiKey={os.getenv('POLYGON_KEY')}"
    response = requests.get(url).json()
    
    penny_stocks = []
    for stock in response['results']:
        if 0.1 <= stock.get('lastTrade', {}).get('p', 0) <= 5.0:
            penny_stocks.append(stock['ticker'])
    return penny_stocks[:100]  # Limit to top 100

async def watchlist_generator():
    """Refresh watchlist every 5 minutes and yield symbols."""
    while True:
        symbols = get_active_penny_stocks()
        for symbol in symbols:
            yield symbol
        await asyncio.sleep(300)  # 5 minutes