import pandas as pd
import numpy as np

def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def detect_volume_spikes(volume: pd.Series, lookback: int = 20, multiplier: int = 3) -> pd.Series:
    rolling_avg = volume.rolling(lookback).mean()
    return volume > (rolling_avg * multiplier)

def calculate_macd(close_prices: pd.Series, 
                  fast: int = 12, 
                  slow: int = 26, 
                  signal: int = 9) -> pd.DataFrame:
    ema_fast = close_prices.ewm(span=fast).mean()
    ema_slow = close_prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return pd.DataFrame({'MACD': macd, 'Signal': signal_line})