#!/usr/bin/env python3
"""
Test script to verify Perplexity API integration is working.
This confirms that the bot is actually calling Perplexity for research.
"""

import asyncio
import sys
from config import load_config
from perplexity_client import PerplexityClient

async def test_perplexity():
    """Test Perplexity API connectivity and response."""
    
    print("\n" + "="*70)
    print("🧪 Testing Perplexity API Integration")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    
    # Check if Perplexity is configured
    if not config.perplexity.enabled:
        print("❌ Perplexity is DISABLED in config")
        print(f"   Set PERPLEXITY_ENABLED=true in .env")
        return False
    
    if not config.perplexity.api_key:
        print("❌ Perplexity API key is MISSING")
        print(f"   Set PERPLEXITY_API_KEY in .env")
        return False
    
    print(f"✅ Config check passed")
    print(f"   Model: {config.perplexity.model}")
    print(f"   API Key: {'*' * 8}{config.perplexity.api_key[-8:]}")
    print()
    
    # Initialize client
    try:
        print("Initializing Perplexity client...")
        client = PerplexityClient(
            api_key=config.perplexity.api_key,
            model=config.perplexity.model
        )
        print("✅ Client initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Test API call
    print("Making test API call to Perplexity...")
    print("  Query: What are the latest developments in US politics?")
    print("  (This will call the Perplexity API with real-time web search)\n")
    
    try:
        response = await client.fetch_event_research(
            event_ticker="TEST",
            event_title="Test Event",
            event_description="Testing Perplexity API integration",
            category="politics",
            sub_category="test",
        )
        
        if response and len(response) > 100:
            print(f"✅ PERPLEXITY API CALL SUCCESSFUL!\n")
            print(f"   Response length: {len(response)} characters")
            print(f"   First 200 chars: {response[:200]}...\n")
            print("🎉 Perplexity integration is WORKING!")
            print("   The bot WILL use Perplexity for real-time web research when trading.\n")
            
            await client.close()
            return True
        else:
            print(f"⚠️  API response was empty or too short: {response}")
            await client.close()
            return False
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        import traceback
        traceback.print_exc()
        await client.close()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_perplexity())
    sys.exit(0 if result else 1)
