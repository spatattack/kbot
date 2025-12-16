"""
Utility script: dump current Kalshi events and markets to CSV for analysis.

Usage:
    cd kalshi-pol-bot
    python dump_kalshi_events.py

This uses your existing .env and config to connect.
"""

import asyncio
import csv
from pathlib import Path
from typing import Any, Dict, List

from config import load_config
from kalshi_client import KalshiClient


async def main() -> None:
    config = load_config()
    kalshi_cfg = config.kalshi

    client = KalshiClient(
        kalshi_cfg,
        minimum_time_remaining_hours=config.minimum_time_remaining_hours,
        max_markets_per_event=config.max_markets_per_event,
        max_close_ts=None,  # pull full set of open markets
    )

    await client.login()

    print("Fetching all open events from Kalshi...")
    # We’re using the internal pagination helper – fine for this analysis script
    events: List[Dict[str, Any]] = await client._fetch_all_events()
    print(f"Fetched {len(events)} events")

    out_path = Path("kalshi_events_dump.csv")
    fieldnames = [
        "event_ticker",
        "event_title",
        "event_category",
        "event_sub_category",
        "strike_date",
        "market_ticker",
        "market_title",
        "market_subtitle",
        "market_volume",
        "market_volume_24h",
        "market_open_interest",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in events:
            event_ticker = ev.get("event_ticker", "")
            event_title = ev.get("title", "")
            event_category = ev.get("category", "")
            event_sub_category = ev.get("sub_category", "")
            strike_date = ev.get("strike_date", "")

            markets = ev.get("markets", []) or []
            if not markets:
                writer.writerow(
                    {
                        "event_ticker": event_ticker,
                        "event_title": event_title,
                        "event_category": event_category,
                        "event_sub_category": event_sub_category,
                        "strike_date": strike_date,
                        "market_ticker": "",
                        "market_title": "",
                        "market_subtitle": "",
                        "market_volume": "",
                        "market_volume_24h": "",
                        "market_open_interest": "",
                    }
                )
                continue

            for m in markets:
                writer.writerow(
                    {
                        "event_ticker": event_ticker,
                        "event_title": event_title,
                        "event_category": event_category,
                        "event_sub_category": event_sub_category,
                        "strike_date": strike_date,
                        "market_ticker": m.get("ticker", ""),
                        "market_title": m.get("title", ""),
                        "market_subtitle": m.get("subtitle", ""),
                        "market_volume": m.get("volume", ""),
                        "market_volume_24h": m.get("volume_24h", ""),
                        "market_open_interest": m.get("open_interest", ""),
                    }
                )

    print(f"✅ Wrote {out_path} – open it in Excel/Sheets to inspect categories and titles.")


if __name__ == "__main__":
    asyncio.run(main())
