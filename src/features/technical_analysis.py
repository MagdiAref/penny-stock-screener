import talib
import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv

load_dotenv("../../config/.env")

REDIS_HOST = os.getenv("REDIS_HOST")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

def calculate_ta(symbol: str, window: int = 14):
    trades = r.lrange(f"trades:{symbol}", 0, -1)
    if len(trades) < window * 2:
        return None
    
    closes = [float(json.loads(t)['price']) for t in trades]
    closes = pd.Series(closes)
    
    rsi = talib.RSI(closes, window).iloc[-1]
    macd, macd_signal, _ = talib.MACD(closes)
    upper_band, middle_band, lower_band = talib.BBANDS(closes)
    
    return {
        "rsi": rsi,
        "macd": macd.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
        "bb_upper": upper_band.iloc[-1],
        "bb_lower": lower_band.iloc[-1]
    }