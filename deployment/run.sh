#!/bin/bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis

# Train models
python src/models/train.py              # XGBoost
python src/reinforcement/train_rl.py    # RL Agent

# Start screener
docker build -t penny-screener .
docker run -d \
  --env-file config/.env \
  --network host \
  penny-screener