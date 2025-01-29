import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environments import TradingEnv
from callbacks import TensorboardCallback

def train_rl_model():
    # Load historical data
    df = pd.read_csv("../data/processed/historical.csv")
    
    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df=df)])
    
    # Initialize model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="../logs/rl_training"
    )
    
    # Train with callbacks
    model.learn(
        total_timesteps=100_000,
        callback=TensorboardCallback(),
        tb_log_name="ppo_penny_stocks"
    )
    
    # Save model
    model.save("../models/trained/rl_agent_ppo")

if __name__ == "__main__":
    train_rl_model()