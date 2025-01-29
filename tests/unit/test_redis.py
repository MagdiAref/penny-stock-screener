import pytest
from src.data.redis_helpers import get_latest_price
import redis

@pytest.fixture
def redis_client():
    return redis.Redis(host="localhost", port=6379, db=0)

def test_get_latest_price(redis_client):
    redis_client.hset("stock:XYZ", "price", "0.50")
    assert get_latest_price("XYZ") == 0.50