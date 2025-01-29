import os
import asyncio
import websockets
import json
import redis
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("../config/.env")

POLYGON_KEY = os.getenv("POLYGON_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")

r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

async def handle_trade(trade_data):
    symbol = trade_data.get('sym')
    price = float(trade_data.get('p', 0))
    volume = int(trade_data.get('s', 0))
    timestamp = datetime.fromtimestamp(trade_data.get('t') / 1000).isoformat()

    # Store raw trade
    r.rpush(f"trades:{symbol}", json.dumps({
        "price": price,
        "volume": volume,
        "timestamp": timestamp
    }))

    # Update latest price
    r.hset(f"stock:{symbol}", mapping={
        "price": price,
        "volume": volume,
        "timestamp": timestamp
    })

async def polygon_websocket():
    async with websockets.connect("wss://socket.polygon.io/stocks") as ws:
        await ws.send(json.dumps({"action":"auth", "params": POLYGON_KEY}))
        await ws.send(json.dumps({"action":"subscribe", "params":"T.*"}))
        
        while True:
            try:
                data = await ws.recv()
                trades = json.loads(data)
                for trade in trades:
                    await handle_trade(trade)
            except Exception as e:
                print(f"Error: {e}. Reconnecting...")
                await asyncio.sleep(5)
                await polygon_websocket()

if __name__ == "__main__":
    asyncio.run(polygon_websocket())