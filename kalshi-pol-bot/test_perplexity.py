#!/usr/bin/env python3
"""
Test script to validate Perplexity API integration.
"""

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from perplexity_client import PerplexityClient

console = Console()


async def test_perplexity():
    """Test Perplexity API connectivity and functionality."""
    
    # Load environment
    load_dotenv()
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        console.print("[red]❌ PERPLEXITY_API_KEY not set in environment[/red]")
        return False
    
    console.print("[cyan]🧪 Testing Perplexity API...[/cyan]")
    console.print(f"[dim]API Key: {'*' * 8}{api_key[-8:]}[/dim]")
    console.print()
    
    try:
        # Initialize client
        console.print("[cyan]1. Initializing Perplexity client...[/cyan]")
        client = PerplexityClient(api_key=api_key, model="sonar")
        console.print("[green]✓ Client initialized successfully[/green]\n")
        
        # Test event research
        console.print("[cyan]2. Testing event research fetch...[/cyan]")
        event_data = {
            'event_ticker': 'TEST-EVENT',
            'event_title': 'Will Donald Trump win the 2024 US Presidential Election?',
            'event_description': 'Prediction market on the outcome of the 2024 US presidential election',
            'category': 'politics',
            'sub_category': 'elections',
        }
        
        console.print(f"[dim]Event: {event_data['event_title']}[/dim]")
        
        research = await client.fetch_event_research(
            event_ticker=event_data['event_ticker'],
            event_title=event_data['event_title'],
            event_description=event_data['event_description'],
            category=event_data['category'],
            sub_category=event_data['sub_category'],
        )
        
        if research:
            console.print(f"[green]✓ Research fetched: {len(research)} characters[/green]")
            console.print(f"[green]✓ First 200 chars: {research[:200]}...[/green]\n")
            
            console.print("[cyan]3. Research Content Sample:[/cyan]")
            console.print("[dim]" + "=" * 70 + "[/dim]")
            console.print(research[:500] + "...")
            console.print("[dim]" + "=" * 70 + "[/dim]\n")
            
            result = True
        else:
            console.print("[red]❌ No research returned[/red]\n")
            result = False
        
        # Close client
        await client.close()
        console.print("[green]✓ Client closed[/green]")
        
        return result
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_perplexity())
    
    if success:
        console.print("\n[bold green]✅ Perplexity API is working correctly![/bold green]")
    else:
        console.print("\n[bold red]❌ Perplexity API test failed[/bold red]")
        console.print("[yellow]Check your API key and internet connection[/yellow]")
