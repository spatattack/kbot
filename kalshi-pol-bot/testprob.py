#!/usr/bin/env python3
"""
Simple diagnostic: Just test probability extraction from sample research.
"""

import asyncio
import openai
from config import load_config

async def test_probability_extraction():
    print("="*60)
    print("🔍 Testing Probability Extraction")
    print("="*60)
    print()
    
    # Sample research (like what your bot gets)
    sample_research = """
Based on recent polling data and expert analysis, here are the key factors:

Recent polls show:
- Candidate A: 52%
- Candidate B: 48%

Historical trends suggest Candidate A has a 55% chance of winning based on:
1. Strong fundraising numbers
2. Better ground game in swing states
3. Recent momentum in polling

Expert consensus: 54-56% probability for Candidate A.
"""

    print("Sample Research:")
    print("-" * 60)
    print(sample_research[:300] + "...")
    print("-" * 60)
    print()
    
    # Load config
    config = load_config()
    openai_client = openai.AsyncOpenAI(api_key=config.openai.api_key)
    
    # Try to extract probability
    print("Attempting to extract probability...")
    print()
    
    prompt = f"""Based on this research, what is the probability this event will happen?

Research:
{sample_research}

Respond with ONLY a number between 0 and 100 representing the percentage probability.
Do not include any explanation, just the number."""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a probability analyst. Extract probabilities from research. Respond with only a number."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip()
        print(f"Raw OpenAI response: '{result}'")
        print()
        
        # Try to parse
        import re
        numbers = re.findall(r'\d+\.?\d*', result)
        
        if numbers:
            prob = float(numbers[0])
            if prob > 1:  # Assume percentage
                prob = prob / 100
            print(f"✅ Successfully extracted: {prob*100:.1f}%")
            print()
            print("✅ PROBABILITY EXTRACTION WORKS!")
            return True
        else:
            print(f"❌ Could not parse number from: '{result}'")
            print()
            print("❌ PROBABILITY EXTRACTION FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error calling OpenAI: {e}")
        print()
        print("❌ PROBABILITY EXTRACTION FAILED")
        return False

async def test_with_real_research():
    """Test with actual Perplexity-style research."""
    print()
    print("="*60)
    print("🔍 Testing with Real-Style Research")
    print("="*60)
    print()
    
    # This is like what Perplexity returns
    real_research = """
The most recent and relevant information indicates that Donald Trump won the 2024 U.S. presidential 
election. However, you're asking about the NEXT election, which would be 2028.

As of December 2025, the 2028 election is still 3 years away. Current early indicators suggest:

Vice President JD Vance is considered the frontrunner for the Republican nomination with an estimated 
40-45% probability of winning the nomination.

On the Democratic side, several candidates are positioning themselves:
- Gavin Newsom: 25-30% probability
- Pete Buttigieg: 15-20% probability  
- Gretchen Whitmer: 15-20% probability

For the general election, if Vance is the Republican nominee, polls suggest a competitive race 
with roughly 48-52% win probability depending on economic conditions.

Historical patterns show the opposition party often has an advantage in the election following a 
two-term presidency, though this is not guaranteed.
"""

    config = load_config()
    openai_client = openai.AsyncOpenAI(api_key=config.openai.api_key)
    
    print("Research excerpt:")
    print("-" * 60)
    print(real_research[:400] + "...")
    print("-" * 60)
    print()
    
    prompt = f"""Based on this research about the 2028 presidential election, what is the probability 
that the Republican party will win?

Research:
{real_research}

Respond with ONLY a number between 0 and 100 representing the percentage probability.
Do not include any explanation, just the number."""
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a probability analyst. Extract probabilities from research. Respond with only a number."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip()
        print(f"Raw response: '{result}'")
        
        import re
        numbers = re.findall(r'\d+\.?\d*', result)
        
        if numbers:
            prob = float(numbers[0])
            if prob > 1:
                prob = prob / 100
            print(f"✅ Extracted: {prob*100:.1f}%")
            print()
            
            if 45 <= prob*100 <= 55:
                print("✅ Reasonable probability (45-55% range)")
            else:
                print(f"⚠️  Probability seems off (expected 45-55%, got {prob*100:.1f}%)")
            
            return True
        else:
            print(f"❌ Could not parse: '{result}'")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Run all tests."""
    print()
    
    # Test 1: Simple extraction
    test1 = await test_probability_extraction()
    
    # Test 2: Real-world research
    test2 = await test_with_real_research()
    
    # Summary
    print()
    print("="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if test1 and test2:
        print("✅ Probability extraction is WORKING")
        print()
        print("The bot should be able to extract probabilities.")
        print("If it's not proposing trades, the issue is elsewhere:")
        print("  1. All extracted probabilities match market prices (no edge)")
        print("  2. R-scores are too low")
        print("  3. Mispricing conviction is low")
    else:
        print("❌ Probability extraction is FAILING")
        print()
        print("This is why you're getting no trades.")
        print("The bot can't extract probabilities from research.")
        print()
        print("Possible fixes:")
        print("  1. Check OpenAI API key is valid")
        print("  2. Try a different model (gpt-4 instead of gpt-4o)")
        print("  3. Improve the extraction prompt")

if __name__ == "__main__":
    asyncio.run(main())