import pandas as pd
from xgboost_model import StockClassifier
from features.technical_analysis import calculate_ta
from features.sentiment import SentimentAnalyzer
import joblib

def load_training_data():
    # Replace with your historical data (columns: timestamp, symbol, open, high, low, close, volume)
    df = pd.read_csv("data/processed/historical.csv")
    df['target'] = (df['close'].shift(-24) > df['close'] * 1.10).astype(int)  # 10% gain in 24h
    return df.dropna()

def extract_features(df: pd.DataFrame):
    analyzer = SentimentAnalyzer()
    features = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        ta_data = calculate_ta(symbol_df['close'])
        sentiment = analyzer.analyze(symbol_df['news_headline'].tolist())
        
        features.append({
            "symbol": symbol,
            "rsi": ta_data['rsi'],
            "macd_diff": ta_data['macd'] - ta_data['macd_signal'],
            "sentiment": sentiment,
            "target": symbol_df['target'].iloc[0]
        })
    
    return pd.DataFrame(features)

if __name__ == "__main__":
    df = load_training_data()
    features_df = extract_features(df)
    X, y = features_df.drop("target", axis=1), features_df['target']
    
    model = StockClassifier()
    model.train(X, y)
    model.save("models/trained/xgboost_model.pkl")