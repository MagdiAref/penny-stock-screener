import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StockClassifier:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print(f"Model Accuracy: {accuracy_score(y_test, preds):.2%}")
    
    def predict(self, X: pd.DataFrame) -> float:
        return self.model.predict_proba(X)[0][1]
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.model = joblib.load(path)
        return instance