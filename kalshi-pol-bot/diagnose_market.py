#!/usr/bin/env python3
"""
Diagnostic: Test one market manually to see where it fails.
"""

import asyncio
from config import load_config
from kalshi_client import KalshiClient
from research_client import OctagonClient
from perplexity_client import PerplexityClient
import openai

async def diagnose():
    print("="*60)
    print("🔍 Market Diagnostic Tool")
    print("="*60)
    print()
    
    # Load config
    config = load_config()
    
    # Create clients
    kalshi = KalshiClient(config.kalshi, live_trading=False)
    octagon = OctagonClient(config.octagon)
    perplexity = PerplexityClient(config.perplexity.api_key, config.perplexity.model) if config.perplexity.enabled else None
    openai_client = openai.AsyncOpenAI(api_key=config.openai.api_key)
    
    # Get one market
    print("1. Fetching a market...")
    events = kalshi.get_events(limit=5, status="open")
    if not events:
        print("   ❌ No events found")
        return
    
    event = events[0]
    print(f"   ✅ Event: {event.get('title', 'Unknown')}")
    
    markets = kalshi.get_markets_for_event(event.get('event_ticker', ''))
    if not markets:
        print("   ❌ No markets found")
        return
    
    market = markets[0]
    ticker = market.get('ticker', '')
    print(f"   ✅ Market: {ticker}")
    print()
    
    # Get research
    print("2. Getting research...")
    octagon_research = await octagon.research_event(event, [market])
    print(f"   ✅ Octagon: {len(octagon_research)} chars")
    
    if perplexity:
        perplexity_research = await perplexity.fetch_event_research(
            event_ticker=event.get('event_ticker', ''),
            event_title=event.get('title', ''),
            event_description=event.get('subtitle', ''),
            category=event.get('category', ''),
            sub_category=event.get('sub_category', ''),
        )
        print(f"   ✅ Perplexity: {len(perplexity_research) if perplexity_research else 0} chars")
        
        combined = f"{octagon_research}\n\n{perplexity_research}" if perplexity_research else octagon_research
    else:
        combined = octagon_research
    
    print(f"   ✅ Combined: {len(combined)} chars")
    print()
    
    # Extract probability using OpenAI
    print("3. Extracting probability...")
    print()
    
    prompt = f"""Based on this research about the market "{market.get('title', '')}", what is the probability this event will happen?

Research:
{combined[:2000]}

Respond with ONLY a number between 0 and 100 representing the percentage probability."""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a probability analyst. Extract probabilities from research."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        result = response.choices[0].message.content
        print(f"   Raw response: {result}")
        
        # Try to parse
        import re
        numbers = re.findall(r'\d+\.?\d*', result)
        if numbers:
            prob = float(numbers[0]) / 100
            print(f"   ✅ Extracted probability: {prob*100:.1f}%")
        else:
            print(f"   ❌ Could not parse probability from: {result}")
            prob = None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        prob = None
    
    print()
    
    # Get market price
    print("4. Getting market price...")
    market_with_odds = kalshi.get_market_with_odds(ticker)
    market_price = market_with_odds.get('yes_ask', 50) / 100
    print(f"   Market price: {market_price*100:.1f}%")
    print()
    
    # Calculate edge
    if prob:
        edge = abs(prob - market_price) * 100
        print(f"5. Edge calculation:")
        print(f"   Research: {prob*100:.1f}%")
        print(f"   Market:   {market_price*100:.1f}%")
        print(f"   Edge:     {edge:.1f}%")
        print()
        
        if edge >= 5.0:
            print("   ✅ WOULD PASS edge filter (5%+)")
        else:
            print(f"   ❌ WOULD FAIL edge filter (need 5%+, got {edge:.1f}%)")
    
    # Cleanup
    await octagon.close()
    if perplexity:
        await perplexity.close()
    kalshi.close()

if __name__ == "__main__":
    asyncio.run(diagnose())