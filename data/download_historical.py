import yfinance as yf
import pandas as pd
import os
from pathlib import Path

def download_penny_stock(symbol: str, years: int = 3) -> pd.DataFrame:
    """Download historical data for a penny stock"""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)
    
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            interval='1d'
        )
        if df.empty:
            return pd.DataFrame()
            
        df = _add_technical_indicators(df)
        _save_raw_data(symbol, df)
        return df
        
    except Exception as e:
        print(f"Error downloading {symbol}: {str(e)}")
        return pd.DataFrame()

def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df['RSI'] = _compute_rsi(df['Close'])
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    return df.dropna()

def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index calculation"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _save_raw_data(symbol: str, df: pd.DataFrame):
    """Save raw data to CSV"""
    raw_path = Path("data/raw/historical")
    raw_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path / f"{symbol}_raw.csv", index=True)

if __name__ == "__main__":
    test_symbols = ["PHUN", "REVB", "BBAI"]
    for sym in test_symbols:
        download_penny_stock(sym)