import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Optional, Tuple

class TradingEnv(Env):
    """Custom Trading Environment for Penny Stocks"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, window_size: int = 30):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Spaces
        self.action_space = Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size + 3,),  # OHLCV + portfolio state
            dtype=np.float32
        )
        
        # Reset
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.history = []
        return self._next_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        current_price = self.df.iloc[self.current_step]['close']
        self._take_action(action, current_price)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self._calculate_reward(current_price)
        
        return self._next_observation(), reward, done, False, {}

    def _take_action(self, action: int, current_price: float):
        if action == 1:  # Buy
            max_possible = int(self.balance // current_price)
            if max_possible > 0:
                self.shares_held += max_possible
                self.balance -= max_possible * current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

    def _calculate_reward(self, current_price: float) -> float:
        portfolio_value = self.balance + self.shares_held * current_price
        return portfolio_value - self.initial_balance

    def _next_observation(self) -> np.ndarray:
        frame = self.df.iloc[self.current_step-self.window_size:self.current_step]
        features = frame[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        portfolio_state = np.array([
            self.balance,
            self.shares_held,
            self.shares_held * frame['close'].iloc[-1]
        ])
        return np.concatenate([features, portfolio_state])