import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import yaml
from dotenv import load_dotenv
from ..features import technical_indicators as ti

load_dotenv("../config/.env")

def load_config():
    with open("../config/config.yaml") as f:
        return yaml.safe_load(f)

def load_and_process_data():
    config = load_config()
    dfs = []
    
    for symbol in config['data']['symbols']:
        try:
            df = pd.read_csv(f"data/raw/historical/{symbol}_raw.csv")
            df = _engineer_features(df)
            dfs.append(df)
        except FileNotFoundError:
            continue
            
    return pd.concat(dfs).dropna()

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI'] = ti.calculate_rsi(df['Close'])
    df['Volume_Spike'] = ti.detect_volume_spikes(df['Volume'])
    df['MACD_Cross'] = df['MACD'] > df['Signal']
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    return df

def train_model():
    df = load_and_process_data()
    features = ['RSI', 'Volume_Spike', 'MACD_Cross', 'Price_Change']
    
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        eval_metric='logloss',
        early_stopping_rounds=20
    )
    
    model.fit(X_train, y_train, 
             eval_set=[(X_test, y_test)],
             verbose=True)
    
    save_path = Path("models/trained")
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path / "xgboost_model.pkl")
    
    print(f"Model trained with accuracy: {model.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    train_model()