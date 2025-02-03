import unittest
from src.utils.redis_manager import RedisManager

class TestRedis(unittest.TestCase):
    def setUp(self):
        self.redis = RedisManager()
        self.test_data = {
            "price": "0.85",
            "volume": "4200000",
            "rsi": "65.3"
        }

    def test_connection(self):
        self.assertTrue(self.redis.connection.ping())
        
    def test_data_caching(self):
        self.redis.cache_stock_data("PHUN_TEST", self.test_data)
        cached = self.redis.get_cached_data("PHUN_TEST")
        self.assertEqual(cached["rsi"], "65.3")
        
    def tearDown(self):
        self.redis.flush_all()

if __name__ == "__main__":
    unittest.main()