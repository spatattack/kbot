# sanity_test.py
from config import load_config
from kalshi_client import KalshiClient

config = load_config()
kc = KalshiClient(config.kalshi)

print("Base URL:", kc.base_url)
events = kc.get_events(limit=3)
print("Got", len(events), "events")
print(events[:1])
