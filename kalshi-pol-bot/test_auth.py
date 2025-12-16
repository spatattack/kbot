#!/usr/bin/env python3
"""
Test Kalshi authentication.
"""

import sys
import asyncio
from config import load_config
from kalshi_client import KalshiClient

async def test_auth():
    """Test Kalshi authentication step by step."""
    print("="*50)
    print("🔐 Testing Kalshi Authentication...")
    print("="*50)
    print()
    
    # Step 1: Load config
    print("1. Loading configuration...")
    try:
        config = load_config()
        print(f"   ✅ Config loaded")
        print(f"   API URL: {config.kalshi.base_url}")
        print(f"   API Key: {config.kalshi.api_key[:20]}...")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Step 2: Create client
    print("2. Creating Kalshi client...")
    try:
        client = KalshiClient(config.kalshi)
        print(f"   ✅ Client created")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Step 3: Login and test public endpoint (no auth)
    print("3. Testing public API endpoint...")
    try:
        await client.login()
        print(f"   ✅ Connected to Kalshi API")
        events = await client.get_events(limit=5)
        print(f"   ✅ Success! Found {len(events)} events")
        if events:
            print(f"   First event: {events[0].get('title', 'Unknown')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
        print("   This might be a network issue.")
        print("   Check your internet connection.")
        await client.close()
        return False
    
    # Step 4: Test authenticated endpoint
    print("4. Testing authenticated endpoint (balance)...")
    try:
        balance = await client.get_balance()
        balance_dollars = balance.get('balance', 0) / 100
        print(f"   ✅ Success! Account balance: ${balance_dollars:.2f}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
        await client.close()
        return False
    
    # Step 5: Test positions endpoint
    print("5. Testing positions endpoint...")
    try:
        positions = await client.get_user_positions()
        print(f"   ✅ Success! Found {len(positions)} positions")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print()
        await client.close()
        return False
    
    # Cleanup
    await client.close()
    
    # Success!
    print("="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)
    print()
    print("Your Kalshi credentials are working correctly.")
    print("You can now run the trading bot:")
    print("  python -m trading_bot.py")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_auth())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
