# Penny Stock Screener

Real-time AI-powered penny stock alerts.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env`:
   ```plaintext
   POLYGON_KEY="your_key"
   NEWSAPI_KEY="your_key"
   DISCORD_WEBHOOK="your_url"
   REDIS_HOST="localhost"

   Train model: python src/models/train.py

Deploy: bash deployment/run.sh

Architecture
Real-time data: Polygon.io WebSocket → Redis

ML: XGBoost + FinBERT

Alerts: Discord webhooks

Copy

---

### **Final Steps**
1. **Add Historical Data**:  
   - Collect historical penny stock data (CSV with `timestamp, symbol, open, high, low, close, volume`).  
   - Place in `data/processed/historical.csv`.

2. **Run Tests**:  
   ```bash
   pytest tests/unit/test_redis.py
Deploy:

bash
Copy
chmod +x deployment/run.sh && ./deployment/run.sh


Replace placeholder API keys in .env.

Monitor logs: tail -f app.log.

Refine the model with your historical data.

