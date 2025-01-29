from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []

    def _on_step(self) -> bool:
        # Log portfolio value
        env = self.training_env.envs[0]
        current_price = env.df.iloc[env.current_step]['close']
        portfolio_value = env.balance + env.shares_held * current_price
        self.logger.record('portfolio/value', portfolio_value)
        
        # Log position size
        self.logger.record('position/shares', env.shares_held)
        return True