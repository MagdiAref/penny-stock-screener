import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any

# Local imports
from data.watchlist import watchlist_generator
from data.redis_helpers import get_latest_price
from features.technical_analysis import calculate_ta
from features.sentiment import SentimentAnalyzer
from models.xgboost_model import StockClassifier
from alerts.discord_alerts import AlertManager
from utils.logger import setup_logger
from models.reinforcement_agent import RLAgent

# Configuration
load_dotenv("../config/.env")
logger = setup_logger("main")

class PennyStockScreener:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.alert_manager = AlertManager()
        self.model = StockClassifier.load("models/trained/xgboost_model.pkl")
        self.rl_agent = RLAgent("models/trained/rl_agent_ppo")
        self.portfolio: Dict[str, Any] = {
            'cash': 10000.00,
            'positions': {},
            'history': []
        }

    async def process_symbol(self, symbol: str):
        try:
            ta_data = calculate_ta(symbol)
            if not ta_data or ta_data['rsi'] == 0:
                logger.warning(f"Insufficient data for {symbol}")
                return

            news = self._get_news(symbol)
            sentiment = self.sentiment_analyzer.analyze(news)
            features = self._create_feature_vector(symbol, ta_data, sentiment)
            
            # Get predictions
            ml_confidence = self.model.predict(features)
            prediction = self.rl_agent.predict(features.drop('symbol', axis=1).values.flatten())
            rl_action = prediction['action']
            
            # Execute trade
            current_price = get_latest_price(symbol)
            self._execute_trade(symbol, rl_action, current_price, ml_confidence)
            
            # Alert if needed
            if ml_confidence > 0.9 or rl_action != 0:  # 0=hold
                self.alert_manager.send_alert(
                    symbol=symbol,
                    confidence=ml_confidence,
                    price=current_price,
                    action=rl_action
                )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)

    def _create_feature_vector(self, symbol: str, ta_data: dict, sentiment: float) -> pd.DataFrame:
        return pd.DataFrame([{
            'symbol': symbol,
            'rsi': ta_data['rsi'],
            'macd_diff': ta_data['macd'] - ta_data['macd_signal'],
            'bb_width': ta_data['bb_upper'] - ta_data['bb_lower'],
            'sentiment': sentiment,
            'volume_ratio': ta_data['volume'] / ta_data['avg_volume'],
            'price': get_latest_price(symbol)
        }])

    def _execute_trade(self, symbol: str, action: int, price: float, confidence: float):
        max_position = self.portfolio['cash'] * 0.05
        quantity = int(max_position // price)
        
        if action == 1:  # Buy
            if self.portfolio['cash'] >= price * quantity:
                self.portfolio['cash'] -= price * quantity
                self.portfolio['positions'][symbol] = self.portfolio['positions'].get(symbol, 0) + quantity
        elif action == 2:  # Sell
            if self.portfolio['positions'].get(symbol, 0) >= quantity:
                self.portfolio['cash'] += price * quantity
                self.portfolio['positions'][symbol] -= quantity
                
        self.portfolio['history'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': ['hold', 'buy', 'sell'][action],
            'quantity': quantity,
            'price': price,
            'confidence': confidence
        })

    def _get_news(self, symbol: str) -> list:
        # Implement NewsAPI integration here
        return [f"News placeholder for {symbol}"]

    def save_portfolio(self):
        with open("data/processed/portfolio.json", "w") as f:
            json.dump(self.portfolio, f, default=str)

async def main():
    screener = PennyStockScreener()
    try:
        async for symbol in watchlist_generator():
            await screener.process_symbol(symbol)
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        screener.save_portfolio()

if __name__ == "__main__":
    asyncio.run(main())