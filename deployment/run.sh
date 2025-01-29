#!/bin/bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis

# Train model
python src/models/train.py

# Start screener
docker build -t stock-screener .
docker run -d \
  --env-file config/.env \
  --network host \
  stock-screener