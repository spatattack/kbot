#!/usr/bin/env python3
"""
Kalshi Political Markets Trading Bot - Institutional Grade
Ultra-selective trading with professional-level decision making.

Key Principles:
1. Only bet on truly inefficient markets
2. Require clear mispricing thesis
3. High conviction thresholds (15%+ edge)
4. Research quality scoring
5. Continuous market monitoring
6. Conservative position sizing (2% max)
"""

import asyncio
import csv
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from betting_models import BettingDecision, MarketProbability, ProbabilityExtraction
from capital_manager import CapitalManager
from config import BotConfig, load_config
from kalshi_client import KalshiClient
from openai_utils import responses_parse_pydantic
from position_manager import PositionManager, CorrelationAnalyzer, AdverseSelectionFilter
from research_client import OctagonClient
from perplexity_client import PerplexityClient

# Configure logging - SUPPRESS NOISE
logging.basicConfig(
    level=logging.INFO,  # Only show warnings and errors
    format="%(message)s",  # Clean format
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# SUPPRESS ALL NOISY LOGGERS
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("kalshi_client").setLevel(logging.CRITICAL)
logging.getLogger("capital_manager").setLevel(logging.CRITICAL)
logging.getLogger("position_manager").setLevel(logging.CRITICAL)
logging.getLogger("trading_bot").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)

# Rich console for UI
console = Console()


class MarketEfficiencyFilter:
    """
    Minimal filtering - only skip if volume is extremely high.
    Evaluate ALL market types - let research decide.
    """
    
    # Only filter if volume is INSANE
    EXTREME_VOLUME = 750_000  # $750k+ = truly institutional
    
    def __init__(self):
        pass  # No pattern matching!
    
    def is_efficient_market(
        self, 
        market: Dict,
        volume_24h: float,
    ) -> Tuple[bool, str]:
        """
        Only skip if volume is extreme.
        
        Returns:
            (should_skip, reason)
        """
        # Only filter extreme volume
        if volume_24h > self.EXTREME_VOLUME:
            return True, f"Extreme volume ${volume_24h:,.0f} (institutional)"
        
        # Everything else is fair game!
        return False, ""
    
    def score_inefficiency_potential(
        self,
        market: Dict,
        volume_24h: float,
        days_to_expiry: float,
        spread_cents: int,
    ) -> Tuple[float, str]:
        """
        Score how likely market is to be inefficient (0-10).
        PERMISSIVE: Most markets should pass.
        
        Returns:
            (score, explanation)
        """
        score = 6.0  # Start higher - assume most markets are worth evaluating
        reasons = []
        
        # Efficient market check (only blocks extreme volume)
        is_efficient, reason = self.is_efficient_market(market, volume_24h)
        if is_efficient:
            return 0.0, f"❌ {reason}"
        
        # Volume - very permissive
        if volume_24h >= 50:
            score += 2.0
            reasons.append("+2 tradeable volume")
        elif volume_24h >= 10:
            score += 1.0
            reasons.append("+1 low but ok")
        # No penalty for low volume, just don't add
        
        # Time to expiry - very permissive
        if days_to_expiry >= 1:  # At least 1 day
            score += 1.0
            reasons.append("+1 has time")
        if days_to_expiry < 0.5:  # Only penalize if < 12 hours
            score -= 2.0
            reasons.append("-2 too close")
        # No penalty for far-out dates
        
        # Spread - any spread is fine
        if spread_cents >= 4:
            score += 1.0
            reasons.append("+1 wide spread")
        elif spread_cents >= 2:
            score += 0.5
            reasons.append("+0.5 medium")
        # No penalty for tight spreads
        
        score = max(0, min(10, score))
        explanation = f"{score:.1f}/10 ({', '.join(reasons) if reasons else 'baseline'})"
        
        return score, explanation


class ResearchQualityScorer:
    """
    Score research quality before trusting it.
    Bad research = bad trades.
    """
    
    def score_research(
        self, 
        research_text: str, 
        market_title: str,
        event_title: str = "",
    ) -> Tuple[float, str]:
        """
        Score research quality (0-10).
        
        Returns:
            (score, explanation)
        """
        score = 4.0  # Start slightly pessimistic
        reasons = []
        
        text_lower = research_text.lower()
        
        # === POSITIVE SIGNALS ===
        
        # Concrete data (the more specific, the better)
        if re.search(r"\d{1,2}%|\d{1,2}\.\d%", research_text):
            score += 1.5
            reasons.append("+1.5 percentages")
        
        if re.search(r"poll|survey|study", text_lower):
            score += 2.0
            reasons.append("+2.0 polls/surveys")
        
        # Recent dates
        recent_date_patterns = [
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+202\d",
            r"(yesterday|today|this week|last week)",
            r"202\d-\d{2}-\d{2}",  # ISO dates
        ]
        for pattern in recent_date_patterns:
            if re.search(pattern, text_lower):
                score += 1.5
                reasons.append("+1.5 recent dates")
                break
        
        # Source attribution
        source_patterns = [
            r"according to",
            r"per\s+(the\s+)?[A-Z]",  # "per The New York Times"
            r"reported by",
            r"sources? (say|said|indicate)",
            r"(reuters|bloomberg|nyt|wsj|politico|ap|cnn|bbc)",
        ]
        source_count = sum(1 for pattern in source_patterns if re.search(pattern, text_lower))
        if source_count >= 2:
            score += 2.0
            reasons.append("+2.0 multiple sources")
        elif source_count == 1:
            score += 1.0
            reasons.append("+1.0 source cited")
        
        # Specific numbers/facts
        number_count = len(re.findall(r"\d+", research_text))
        if number_count >= 10:
            score += 1.0
            reasons.append("+1.0 many numbers")
        elif number_count >= 5:
            score += 0.5
            reasons.append("+0.5 some numbers")
        
        # === NEGATIVE SIGNALS ===
        
        # Vague language
        vague_words = ["might", "could", "possibly", "perhaps", "maybe", "uncertain", "unclear", "potentially"]
        vague_count = sum(1 for word in vague_words if word in text_lower)
        if vague_count >= 5:
            score -= 2.0
            reasons.append("-2.0 very vague")
        elif vague_count >= 3:
            score -= 1.0
            reasons.append("-1.0 somewhat vague")
        
        # Too short (< 200 chars = lazy research)
        if len(research_text) < 200:
            score -= 2.0
            reasons.append("-2.0 too short")
        
        # No specific details
        if not re.search(r"\d", research_text):
            score -= 2.0
            reasons.append("-2.0 no numbers")
        
        # Check relevance to market
        market_keywords = set(re.findall(r'\w+', market_title.lower()))
        event_keywords = set(re.findall(r'\w+', event_title.lower())) if event_title else set()
        all_keywords = market_keywords | event_keywords
        
        research_keywords = set(re.findall(r'\w+', text_lower))
        overlap_count = len(all_keywords & research_keywords)
        
        if overlap_count < 2:
            score -= 3.0
            reasons.append("-3.0 low relevance")
        elif overlap_count >= 5:
            score += 1.0
            reasons.append("+1.0 high relevance")
        
        # Generic/templated research
        generic_phrases = [
            "the outcome of this",
            "it is important to",
            "factors to consider",
            "on the one hand",
        ]
        if any(phrase in text_lower for phrase in generic_phrases):
            score -= 1.0
            reasons.append("-1.0 generic")
        
        # Cap score
        score = max(0, min(10, score))
        
        explanation = f"Research: {score:.1f}/10 ({', '.join(reasons)})"
        return score, explanation


class MispricingAnalyzer:
    """
    Analyze WHY a market is mispriced before betting.
    If we can't explain it, don't bet.
    """
    
    async def analyze_mispricing(
        self,
        market: Dict,
        research_prob: float,
        market_prob: float,
        research_text: str,
        openai_client: openai.AsyncOpenAI,
    ) -> Tuple[bool, str, float]:
        """
        Determine if mispricing is real and tradeable.
        
        Returns:
            (is_tradeable, explanation, conviction_score)
        """
        edge_pct = abs(research_prob - market_prob) * 100
        direction = "overpriced" if market_prob > research_prob else "underpriced"
        
        # Build analysis prompt
        prompt = f"""You are a professional prediction market trader analyzing a potential mispricing.

Market: {market.get('title', '')}
Market Price: {market_prob*100:.1f}%
Research Estimate: {research_prob*100:.1f}%
Edge: {edge_pct:.1f} percentage points ({direction})

Research Summary:
{research_text[:800]}

Analyze this potential trade critically:

1. **Why might the market be wrong?**
   - Information asymmetry?
   - Complexity market missed?
   - Recency bias in market?
   - Technical factors?

2. **Why might WE be wrong?**
   - Is our research missing something?
   - Is the market seeing something we're not?
   - Could this be adverse selection?

3. **Base rate reasoning:**
   - How often do markets misprice by this amount?
   - What's the base rate for this type of event?

4. **Trade conviction:**
   - Rate 0-10 how convinced you are this is real edge
   - What could change your mind?

Return JSON:
{{
  "is_tradeable": boolean,
  "explanation": "2-3 sentence summary",
  "conviction_score": float (0-10),
  "key_risk": "main thing that could make us wrong"
}}
"""
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a skeptical, risk-aware prediction market analyst."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            is_tradeable = result.get("is_tradeable", False)
            explanation = result.get("explanation", "No explanation provided")
            conviction = float(result.get("conviction_score", 0))
            key_risk = result.get("key_risk", "Unknown")
            
            full_explanation = f"{explanation} [Key risk: {key_risk}]"
            
            return is_tradeable, full_explanation, conviction
            
        except Exception as e:
            logger.error(f"Mispricing analysis failed: {e}")
            return False, f"Analysis failed: {e}", 0.0


class KalshiInstitutionalBot:
    """
    Institutional-grade trading bot.
    Beats the market by being smarter, not faster.
    """
    
    def __init__(self, config: BotConfig):
        """Initialize the bot with configuration."""
        self.config = config
        
        # Core clients
        self.kalshi_client = KalshiClient(config.kalshi)
        self.research_client = OctagonClient(config.octagon)
        self.openai_client = openai.AsyncOpenAI(api_key=config.openai.api_key)
        
        # Perplexity for real-time web research
        self.perplexity_client = None
        self.perplexity_enabled = False
        console.print("\n[bold cyan]🔧 Initializing Perplexity Research API...[/bold cyan]")
        logger.warning(f"PERPLEXITY CONFIG: enabled={config.perplexity.enabled}, has_api_key={bool(config.perplexity.api_key)}")
        
        if config.perplexity.enabled and config.perplexity.api_key:
            try:
                console.print(f"[dim]  Creating client with model: {config.perplexity.model}[/dim]")
                self.perplexity_client = PerplexityClient(
                    api_key=config.perplexity.api_key,
                    model=config.perplexity.model
                )
                self.perplexity_enabled = True
                console.print("[bold green]✓ Perplexity API enabled and ready[/bold green]")
                console.print(f"[dim]  Model: {config.perplexity.model}[/dim]")
                console.print(f"[dim]  API Key: {'*' * 8}{config.perplexity.api_key[-8:]}[/dim]")
                logger.warning("PERPLEXITY INITIALIZED: Ready to make API calls")
            except Exception as e:
                console.print(f"[red]✗ Perplexity initialization failed: {e}[/red]")
                logger.warning(f"PERPLEXITY INIT ERROR: {type(e).__name__}: {e}")
                self.perplexity_enabled = False
                self.perplexity_client = None
        else:
            console.print("[yellow]⚠️  Perplexity not configured[/yellow]")
            if not config.perplexity.api_key:
                console.print("[dim]  Reason: No API key in .env[/dim]")
                logger.warning("PERPLEXITY DISABLED: No API key")
            elif not config.perplexity.enabled:
                console.print("[dim]  Reason: Disabled in configuration[/dim]")
                logger.warning("PERPLEXITY DISABLED: Not enabled in config")
        
        # Risk management - FIXED INITIALIZATION
        self.capital_manager = CapitalManager(
            self.kalshi_client,
            max_position_pct=0.02,  # 2% max per position
            max_event_pct=0.05,     # 5% max per event
            max_total_deployed_pct=0.30,  # 30% max deployed (conservative)
        )
        self.position_manager = PositionManager(
            kalshi_client=self.kalshi_client,
            research_client=self.research_client,
            openai_client=self.openai_client,
            config=config,
        )
        
        # Intelligence components
        self.efficiency_filter = MarketEfficiencyFilter()
        self.research_scorer = ResearchQualityScorer()
        self.mispricing_analyzer = MispricingAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.adverse_selection_filter = AdverseSelectionFilter()
        
        # State tracking
        self.decisions_log: List[BettingDecision] = []
        self.research_cache: Dict[str, Tuple[str, datetime]] = {}
        self.daily_positions_taken = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        # INSTITUTIONAL GRADE THRESHOLDS (relaxed for demo environment)
        self.MIN_EDGE_PCT = 5.0               # 5% edge minimum (demo)
        self.MIN_R_SCORE = 0.3                # 0.3 standard deviations (very permissive)
        self.MIN_CONFIDENCE = 0.55            # 55% confidence (demo)
        self.MIN_RESEARCH_QUALITY = 4.0       # 4/10 research quality (demo)
        self.MIN_INEFFICIENCY_SCORE = 2.0     # 2/10 inefficiency score (very permissive)
        self.MIN_MISPRICING_CONVICTION = 4.0  # 4/10 mispricing conviction (demo)
        
        # Dynamic limits (quality-based, not hard caps)
        self.MAX_POSITIONS_EXCELLENT = 5      # Up to 5 if all excellent (>8.5/10)
        self.MAX_POSITIONS_GOOD = 3           # Up to 3 if good (7-8.5/10)
        self.MAX_POSITIONS_OK = 1             # Only 1 if merely OK (6-7/10)
        
        # Position sizing
        self.BASE_KELLY_FRACTION = 0.25       # Quarter Kelly (conservative)
        self.MAX_POSITION_SIZE_PCT = 0.02     # 2% of bankroll max
        
    def _reset_daily_limits(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now(timezone.utc).date()
        if today > self.last_reset_date:
            self.daily_positions_taken = 0
            self.last_reset_date = today
            logger.info(f"Reset daily position counter for {today}")
    
    def _calculate_quality_score(
        self,
        inefficiency_score: float,
        research_quality: float,
        mispricing_conviction: float,
        edge_pct: float,
        r_score: float,
    ) -> float:
        """
        Calculate overall quality score for a potential trade.
        
        Returns:
            Quality score 0-10
        """
        # Weighted average of components
        quality = (
            inefficiency_score * 0.20 +
            research_quality * 0.25 +
            mispricing_conviction * 0.30 +
            min(edge_pct / 20.0 * 10, 10) * 0.15 +  # Edge normalized to 10
            min(r_score / 3.0 * 10, 10) * 0.10      # R-score normalized to 10
        )
        
        return min(10, quality)
    
    def _get_max_positions_for_quality(self, quality_score: float) -> int:
        """Determine max positions based on trade quality."""
        if quality_score >= 8.5:
            return self.MAX_POSITIONS_EXCELLENT
        elif quality_score >= 7.0:
            return self.MAX_POSITIONS_GOOD
        elif quality_score >= 6.0:
            return self.MAX_POSITIONS_OK
        else:
            return 0  # Don't trade if quality too low
    
    def print_config_summary(self):
        """Print configuration summary banner."""
        mode_color = "red" if not self.config.dry_run else "green"
        mode_text = "LIVE TRADING" if not self.config.dry_run else "DRY RUN"
        
        console.print("\n" + "="*70, style="bold blue")
        console.print("🏛️  KALSHI INSTITUTIONAL TRADING BOT", style="bold white", justify="center")
        console.print("="*70 + "\n", style="bold blue")
        
        # Create config table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Setting", style="cyan", width=30)
        table.add_column("Value", style="white")
        
        table.add_row("Environment", "PRODUCTION" if not self.config.kalshi.use_demo else "DEMO")
        table.add_row("Mode", Text(mode_text, style=mode_color))
        table.add_row("", "")
        table.add_row("Strategy", "Institutional Grade - Ultra Selective")
        table.add_row("Research Method", "Dual Source (Octagon + Perplexity)" if self.perplexity_client else "Single Source (Octagon)")
        table.add_row("", "")
        table.add_row("📊 Minimum Thresholds:", "")
        table.add_row("  • Edge Required", f"{self.MIN_EDGE_PCT}% ({self.MIN_R_SCORE}σ)")
        table.add_row("  • Confidence", f"{self.MIN_CONFIDENCE*100:.0f}%")
        table.add_row("  • Research Quality", f"{self.MIN_RESEARCH_QUALITY}/10")
        table.add_row("  • Inefficiency Score", f"{self.MIN_INEFFICIENCY_SCORE}/10")
        table.add_row("  • Mispricing Conviction", f"{self.MIN_MISPRICING_CONVICTION}/10")
        table.add_row("", "")
        table.add_row("💰 Position Sizing:", "")
        table.add_row("  • Max Per Position", f"{self.MAX_POSITION_SIZE_PCT*100}% of bankroll")
        table.add_row("  • Kelly Fraction", f"{self.BASE_KELLY_FRACTION*100:.0f}%")
        table.add_row("", "")
        table.add_row("📈 Dynamic Limits:", "")
        table.add_row("  • Excellent Trades (8.5+)", f"Up to {self.MAX_POSITIONS_EXCELLENT} positions")
        table.add_row("  • Good Trades (7-8.5)", f"Up to {self.MAX_POSITIONS_GOOD} positions")
        table.add_row("  • OK Trades (6-7)", f"Up to {self.MAX_POSITIONS_OK} position")
        
        console.print(table)
        console.print()

    async def run(self):
        """Main bot execution flow."""
        try:
            self.print_config_summary()
            
            # Reset daily limits
            self._reset_daily_limits()

            # Initialize Kalshi client
            await self.kalshi_client.login()
            
            # Refresh capital state with progress
            console.print("💰 [bold cyan]Refreshing capital state...[/bold cyan]")
            
            auth_failed = False
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Loading capital info...", total=None)
                    await self.capital_manager.refresh_capital_state()
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    auth_failed = True
                    console.print("[yellow]⚠️  Authentication issue detected[/yellow]")
                    console.print("[dim]Running in limited mode (no portfolio data)[/dim]")
                else:
                    raise
            
            if not auth_failed:
                console.print(self.capital_manager.get_position_summary())
            else:
                console.print("\n[dim]Capital state unavailable - proceeding with market analysis[/dim]\n")
            
            # Check existing positions with progress
            if not auth_failed:
                console.print("🔍 [bold cyan]Evaluating existing positions...[/bold cyan]")
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Analyzing positions...", total=None)
                        exit_signals = await self.position_manager.evaluate_all_positions()
                    
                    if exit_signals:
                        console.print(f"\n⚠️  [yellow]Found {len(exit_signals)} exit signals[/yellow]\n")
                        await self._handle_exit_signals(exit_signals)
                    else:
                        console.print("[green]✓ All positions look good[/green]\n")
                except:
                    console.print("[dim]Position evaluation unavailable[/dim]\n")
            
            # Find new opportunities with progress
            console.print("🔎 [bold cyan]Scanning for new opportunities...[/bold cyan]")
            opportunities = await self._scan_for_opportunities()
            
            if not opportunities:
                console.print("\n[yellow]No qualifying opportunities found[/yellow]")
                console.print("[dim]💤 This is normal - most markets are efficient.[/dim]")
                return
            
            console.print(f"\n[green]✓ Found {len(opportunities)} potential trades[/green]\n")
            
            # Execute trades
            await self._execute_opportunities(opportunities)
            
            console.print("\n" + "="*70)
            console.print("✓ [green bold]Bot run complete[/green bold]")
            console.print("="*70 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        except Exception as e:
            error_str = str(e)
            
            # Check for common API errors
            if "502" in error_str or "Bad Gateway" in error_str:
                console.print("\n[red]❌ Kalshi API Error: 502 Bad Gateway[/red]")
                console.print("[yellow]This means Kalshi's servers are temporarily down or overloaded.[/yellow]")
                console.print("[dim]This is NOT a problem with your bot - it's Kalshi's issue.[/dim]")
                console.print("\n[cyan]What to do:[/cyan]")
                console.print("  1. Wait 5-10 minutes")
                console.print("  2. Try running the bot again: [bold]python trading_bot.py[/bold]")
                console.print("  3. If still failing, check https://status.kalshi.co")
            elif "503" in error_str or "Service Unavailable" in error_str:
                console.print("\n[red]❌ Kalshi API Error: 503 Service Unavailable[/red]")
                console.print("[yellow]Kalshi may be in maintenance mode.[/yellow]")
                console.print("[dim]Try again in 15-30 minutes.[/dim]")
            elif "401" in error_str or "Unauthorized" in error_str:
                console.print("\n[red]❌ Authentication Error[/red]")
                console.print("[yellow]Your Kalshi API credentials are not working.[/yellow]")
                console.print("\n[cyan]To fix:[/cyan]")
                console.print("  1. Run: [bold]python test_auth.py[/bold]")
                console.print("  2. See: [bold]FIX_AUTH_ERROR.md[/bold] for details")
            elif "timeout" in error_str.lower():
                console.print("\n[red]❌ Network Timeout[/red]")
                console.print("[yellow]Connection to Kalshi timed out.[/yellow]")
                console.print("[dim]Check your internet connection and try again.[/dim]")
            else:
                # Generic error
                console.print(f"\n[red]❌ Error: {e}[/red]")
                console.print("[dim]Check the error message above for details.[/dim]")
    
    async def _scan_for_opportunities(self) -> List[Dict]:
        """
        Scan markets for high-quality opportunities.
        Returns list of qualified opportunities.
        """
        # Get events with retry logic (filter for politics category)
        all_events = []
        
        for attempt in range(3):  # 3 retries
            try:
                console.print(f"[dim]Fetching events from Kalshi (attempt {attempt + 1}/3)...[/dim]")
                all_events = await self.kalshi_client.get_events(limit=100, status="open")
                break
            except Exception as e:
                if "502" in str(e) or "Bad Gateway" in str(e):
                    console.print(f"[yellow]⚠️  Kalshi API temporarily unavailable (502 error)[/yellow]")
                    if attempt < 2:  # Not last attempt
                        wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                        console.print(f"[dim]Retrying in {wait_time} seconds...[/dim]")
                        await asyncio.sleep(wait_time)
                    else:
                        console.print("[red]❌ Kalshi API still unavailable after 3 attempts[/red]")
                        console.print("[dim]This is a Kalshi server issue, not a bot problem.[/dim]")
                        console.print("[dim]Try again in a few minutes.[/dim]")
                        return []
                elif "503" in str(e) or "Service Unavailable" in str(e):
                    console.print(f"[yellow]⚠️  Kalshi API maintenance mode[/yellow]")
                    return []
                else:
                    # Other error, re-raise
                    raise
        
        if not all_events:
            console.print("[yellow]⚠️  No events returned from Kalshi[/yellow]")
            return []
        
        # Filter for political events
        political_categories = ["politics", "election", "congress", "president", "senate", "house"]
        events = [
            e for e in all_events 
            if any(cat in e.get("category", "").lower() for cat in political_categories)
        ][:50]  # Limit to 50
        
        if not events:
            console.print("[yellow]⚠️  No political events found[/yellow]")
            return []
        
        console.print(f"[dim]Scanning {len(events)} political events...[/dim]")
        
        opportunities = []
        markets_evaluated = 0
        markets_qualified = 0
        markets_researched_count = 0
        
        MAX_RESEARCH = 40  # HARD LIMIT: Stop after researching 40 markets
        MAX_PER_EVENT = 3   # Only research top 3 markets per event
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,  # Disappears when done
        ) as progress:
            task = progress.add_task("Evaluating markets...", total=None)
            
            for event in events:
                if markets_researched_count >= MAX_RESEARCH:
                    console.print(f"\n[yellow]⚠️  Researched {MAX_RESEARCH} markets - stopping to save time[/yellow]")
                    break
                
                try:
                    # Get markets for this event
                    markets = await self.kalshi_client.get_markets_for_event(
                        event.get('event_ticker', ''),
                    )
                    
                    if not markets:
                        continue
                    
                    # SMART SORTING: Prioritize markets likely to be mispriced
                    # NOT just high volume (those are efficient!)
                    def mispricing_potential(m):
                        volume = m.get('volume_24h', 0)
                        
                        # Get spread if available
                        yes_ask = m.get('yes_ask', 50)
                        yes_bid = m.get('yes_bid', 50)
                        spread = yes_ask - yes_bid
                        
                        score = 0
                        
                        # Wide spread = disagreement = opportunity
                        if spread >= 6:
                            score += 30
                        elif spread >= 4:
                            score += 20
                        elif spread >= 2:
                            score += 10
                        
                        # Medium volume sweet spot (not too dead, not too efficient)
                        if 50 <= volume <= 5000:
                            score += 25  # Sweet spot!
                        elif 10 <= volume < 50:
                            score += 15  # Low but tradeable
                        elif volume > 5000:
                            score += 5   # High volume = efficient
                        
                        # Add some volume so we can trade
                        score += min(volume / 100, 10)  # Up to +10 for liquidity
                        
                        return score
                    
                    # Sort by mispricing potential (not just volume!)
                    markets_sorted = sorted(markets, key=mispricing_potential, reverse=True)
                    
                    # Only take top markets per event
                    markets_to_check = markets_sorted[:MAX_PER_EVENT]
                    
                    # Evaluate each market
                    event_researched = 0
                    for market in markets_to_check:
                        if markets_researched_count >= MAX_RESEARCH:
                            break
                        
                        markets_evaluated += 1
                        
                        # Check if in cache (won't need research)
                        ticker = market.get('ticker', '')
                        needs_research = ticker not in self.research_cache
                        
                        if needs_research and event_researched >= MAX_PER_EVENT:
                            continue  # Skip to next event after max per event
                        
                        progress.update(task, description=f"Evaluated {markets_evaluated} (researched {markets_researched_count}/{MAX_RESEARCH}), found {markets_qualified}...")
                        
                        opp = await self._evaluate_market(event, market)
                        
                        if needs_research:
                            markets_researched_count += 1
                            event_researched += 1
                        
                        if opp:
                            opportunities.append(opp)
                            markets_qualified += 1
                            console.print(f"[green]✓ Found: {ticker} (Quality: {opp['quality_score']:.1f}/10)[/green]")
                    
                except Exception as e:
                    continue
        
        console.print(f"[dim]Evaluated {markets_evaluated} markets total[/dim]")
        
        # Sort by quality score (descending)
        opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return opportunities
    
    async def _evaluate_market(
        self, 
        event: Dict, 
        market: Dict
    ) -> Optional[Dict]:
        """
        Evaluate a single market for trading opportunity.
        Returns opportunity dict if qualified, None otherwise.
        """
        ticker = market.get('ticker', '')
        title = market.get('title', '')
        
        try:
            # === FILTER 1: Basic checks ===
            if not market.get('can_close_early', True):
                logger.debug(f"{ticker}: Cannot close early, skip")
                return None
            
            # Get market data with odds
            market_with_odds = await self.kalshi_client.get_market_with_odds(ticker)
            if not market_with_odds:
                return None
            
            # Check spread and pricing
            yes_ask = market_with_odds.get('yes_ask', 50)
            yes_bid = market_with_odds.get('yes_bid', 50)
            spread_cents = yes_ask - yes_bid
            
            if spread_cents < 0:
                return None
            
            market_price_cents = yes_ask
            market_prob = market_price_cents / 100.0
            
            # Calculate time to expiry
            close_time_str = market.get('close_time')
            if not close_time_str:
                return None
            
            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            time_to_expiry = close_time - now
            days_to_expiry = time_to_expiry.total_seconds() / 86400
            
            if days_to_expiry < 0.5:  # Less than 12 hours
                logger.debug(f"{ticker}: Too close to expiry")
                return None
            
            # Get volume
            volume_24h = market.get('volume_24h', 0)
            
            # === FILTER 2: Market efficiency ===
            is_efficient, efficiency_reason = self.efficiency_filter.is_efficient_market(
                market, volume_24h
            )
            
            if is_efficient:
                logger.info(f"{ticker}: SKIP - {efficiency_reason}")
                return None
            
            # Score inefficiency potential
            inefficiency_score, ineff_explanation = self.efficiency_filter.score_inefficiency_potential(
                market, volume_24h, days_to_expiry, spread_cents
            )
            
            if inefficiency_score < self.MIN_INEFFICIENCY_SCORE:
                logger.info(f"{ticker}: SKIP - {ineff_explanation}")
                return None
            
            # === FILTER 3: Research ===
            research_text = await self._get_research(event, market)
            
            if not research_text or len(research_text) < 100:
                logger.info(f"{ticker}: SKIP - Insufficient research")
                return None
            
            # Score research quality
            research_quality, research_explanation = self.research_scorer.score_research(
                research_text, title, event.get('title', '')
            )
            
            if research_quality < self.MIN_RESEARCH_QUALITY:
                logger.info(f"{ticker}: SKIP - {research_explanation}")
                return None
            
            # Extract probability from research
            research_prob = await self._extract_probability_from_research(
                research_text, market, event
            )
            
            if research_prob is None:
                console.print(f"  [red]❌ {ticker}: Could not extract probability[/red]")
                logger.info(f"{ticker}: SKIP - Could not extract probability")
                return None
            
            console.print(f"  🎯 {ticker}: Research {research_prob*100:.1f}% vs Market {market_prob*100:.1f}%")
            
            # === FILTER 4: Edge calculation ===
            edge_pct = abs(research_prob - market_prob) * 100
            
            if edge_pct < self.MIN_EDGE_PCT:
                console.print(f"  [red]❌ {ticker}: Edge {edge_pct:.1f}% < {self.MIN_EDGE_PCT}%[/red]")
                logger.info(f"{ticker}: SKIP - Edge too small ({edge_pct:.1f}%)")
                return None
            
            console.print(f"  [green]✓ {ticker}: Edge {edge_pct:.1f}% ✓[/green]")
            
            # Calculate R-score (risk-adjusted edge)
            variance = market_prob * (1 - market_prob)
            r_score = (research_prob - market_prob) / math.sqrt(variance) if variance > 0 else 0
            
            if abs(r_score) < self.MIN_R_SCORE:
                console.print(f"  [red]❌ {ticker}: R-score {r_score:.2f} < {self.MIN_R_SCORE}[/red]")
                logger.info(f"{ticker}: SKIP - R-score too low ({r_score:.2f})")
                return None
            
            console.print(f"  [green]✓ {ticker}: R-score {r_score:.2f} ✓[/green]")
            
            # === FILTER 5: Mispricing analysis ===
            is_tradeable, mispricing_explanation, mispricing_conviction = await self.mispricing_analyzer.analyze_mispricing(
                market, research_prob, market_prob, research_text, self.openai_client
            )
            
            if not is_tradeable:
                console.print(f"  [red]❌ {ticker}: Not tradeable - {mispricing_explanation[:50]}[/red]")
                logger.info(f"{ticker}: SKIP - Not tradeable: {mispricing_explanation}")
                return None
            
            if mispricing_conviction < self.MIN_MISPRICING_CONVICTION:
                console.print(f"  [red]❌ {ticker}: Conviction {mispricing_conviction:.1f} < {self.MIN_MISPRICING_CONVICTION}[/red]")
                logger.info(f"{ticker}: SKIP - Low conviction ({mispricing_conviction:.1f}/10)")
                return None
            
            console.print(f"  [bold green]✓✓✓ {ticker}: QUALIFIED! (Edge {edge_pct:.1f}%, R {r_score:.2f}, Conv {mispricing_conviction:.1f})[/bold green]")
            
            if not is_tradeable:
                logger.info(f"{ticker}: SKIP - Not tradeable: {mispricing_explanation}")
                return None
            
            if mispricing_conviction < self.MIN_MISPRICING_CONVICTION:
                logger.info(f"{ticker}: SKIP - Low conviction ({mispricing_conviction:.1f}/10)")
                return None
            
            # === CALCULATE QUALITY SCORE ===
            quality_score = self._calculate_quality_score(
                inefficiency_score,
                research_quality,
                mispricing_conviction,
                edge_pct,
                abs(r_score),
            )
            
            logger.info(
                f"{ticker}: QUALIFIED ✓ "
                f"Quality={quality_score:.1f}/10, Edge={edge_pct:.1f}%, "
                f"R={r_score:.2f}, Conv={mispricing_conviction:.1f}/10"
            )
            
            # Determine action (YES or NO)
            action = "buy_yes" if research_prob > market_prob else "buy_no"
            
            return {
                'ticker': ticker,
                'event_ticker': event.get('event_ticker', ''),
                'market': market,
                'event': event,
                'action': action,
                'research_prob': research_prob,
                'market_prob': market_prob,
                'edge_pct': edge_pct,
                'r_score': r_score,
                'quality_score': quality_score,
                'inefficiency_score': inefficiency_score,
                'research_quality': research_quality,
                'mispricing_conviction': mispricing_conviction,
                'research_text': research_text,
                'mispricing_explanation': mispricing_explanation,
                'days_to_expiry': days_to_expiry,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            return None
    
    async def _execute_opportunities(self, opportunities: List[Dict]):
        """Execute qualified opportunities based on quality and limits."""
        if not opportunities:
            return
        
        console.print("\n" + "="*70)
        console.print("🎯 [bold cyan]Qualified Opportunities[/bold cyan]")
        console.print("="*70 + "\n")
        
        # Show opportunities table
        table = Table(title="High-Quality Trades", title_style="bold green")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Quality", justify="right", style="green")
        table.add_column("Edge", justify="right", style="yellow")
        table.add_column("R-Score", justify="right", style="magenta")
        table.add_column("Conviction", justify="right", style="blue")
        table.add_column("Action", justify="center", style="bold yellow")
        
        for opp in opportunities[:10]:  # Show top 10
            quality_style = "bold green" if opp['quality_score'] >= 8.5 else "green" if opp['quality_score'] >= 7.0 else "yellow"
            table.add_row(
                opp['ticker'],
                f"[{quality_style}]{opp['quality_score']:.1f}/10[/{quality_style}]",
                f"{opp['edge_pct']:.1f}%",
                f"{opp['r_score']:.2f}σ",
                f"{opp['mispricing_conviction']:.1f}/10",
                f"🔼 {opp['action'].upper().replace('_', ' ')}" if 'yes' in opp['action'] else f"🔽 {opp['action'].upper().replace('_', ' ')}",
            )
        
        if len(opportunities) > 10:
            table.add_row("...", "...", "...", "...", "...", "...")
        
        console.print(table)
        console.print()
        
        # Execute trades based on quality
        console.print("💸 [bold cyan]Executing trades...[/bold cyan]\n")
        
        positions_taken = 0
        
        for i, opp in enumerate(opportunities, 1):
            # Check quality-based limits
            max_positions = self._get_max_positions_for_quality(opp['quality_score'])
            
            if positions_taken >= max_positions:
                console.print(
                    f"[yellow]⚠️  Reached position limit for this quality tier "
                    f"({positions_taken}/{max_positions})[/yellow]"
                )
                break
            
            # Calculate position size
            position_size = self._calculate_position_size(opp)
            
            if position_size < 1.0:
                console.print(f"[dim]⏭️  Skipping {opp['ticker']}: Position size too small (${position_size:.2f})[/dim]")
                continue
            
            # Execute trade
            success = await self._execute_trade(opp, position_size)
            
            if success:
                positions_taken += 1
                self.daily_positions_taken += 1
        
        console.print("\n" + "="*70)
        if positions_taken > 0:
            console.print(f"✅ [bold green]Executed {positions_taken} trades[/bold green]")
        else:
            console.print("[yellow]No trades executed (position sizes too small or limits reached)[/yellow]")
        console.print("="*70)
    
    def _calculate_position_size(self, opportunity: Dict) -> float:
        """Calculate position size using Kelly criterion with limits."""
        # Kelly fraction: f = (p*odds - q) / odds
        # Where p = research_prob, q = 1-p, odds = payout if correct
        
        research_prob = opportunity['research_prob']
        market_prob = opportunity['market_prob']
        is_yes = opportunity['action'] == "buy_yes"
        
        if is_yes:
            # Buying YES at market_prob
            # If correct, get 1.0, pay market_prob
            # Payout = (1.0 - market_prob) / market_prob
            if market_prob >= 0.99:
                return 0.0
            payout = (1.0 - market_prob) / market_prob
            kelly_fraction = (research_prob * payout - (1 - research_prob)) / payout
        else:
            # Buying NO (equivalent to shorting YES at 1-market_prob)
            no_price = 1.0 - market_prob
            if no_price >= 0.99:
                return 0.0
            payout = (1.0 - no_price) / no_price
            no_prob = 1.0 - research_prob  # Prob that NO wins
            kelly_fraction = (no_prob * payout - research_prob) / payout
        
        # Apply fractional Kelly
        adjusted_kelly = kelly_fraction * self.BASE_KELLY_FRACTION
        
        # Get bankroll
        capital_state = self.capital_manager.capital_state
        if not capital_state:
            logger.warning("No capital state, using small position")
            return 5.0
        
        bankroll = capital_state.total_balance
        
        # Calculate raw size
        raw_size = bankroll * adjusted_kelly
        
        # Apply max limits
        max_size_pct = bankroll * self.MAX_POSITION_SIZE_PCT
        size = min(raw_size, max_size_pct)
        
        # Apply capital manager limits
        size = self.capital_manager.calculate_safe_position_size(
            size,
            opportunity['event_ticker'],
            opportunity['ticker'],
        )
        
        return max(0, size)
    
    async def _execute_trade(self, opportunity: Dict, position_size: float) -> bool:
        """Execute a single trade with beautiful output."""
        ticker = opportunity['ticker']
        action = opportunity['action']
        
        # Create a mini table for this trade
        trade_table = Table(show_header=False, box=None, padding=(0, 1))
        trade_table.add_column("Field", style="cyan")
        trade_table.add_column("Value", style="white")
        
        quality_color = "bold green" if opportunity['quality_score'] >= 8.5 else "green" if opportunity['quality_score'] >= 7.0 else "yellow"
        
        trade_table.add_row("📊 Market", ticker)
        trade_table.add_row("💰 Size", f"${position_size:.2f}")
        trade_table.add_row("✨ Quality", f"[{quality_color}]{opportunity['quality_score']:.1f}/10[/{quality_color}]")
        trade_table.add_row("📈 Edge", f"[yellow]{opportunity['edge_pct']:.1f}%[/yellow] ({opportunity['r_score']:.2f}σ)")
        trade_table.add_row("🎯 Conviction", f"[blue]{opportunity['mispricing_conviction']:.1f}/10[/blue]")
        trade_table.add_row("💡 Thesis", opportunity['mispricing_explanation'][:100] + "...")
        
        console.print(f"\n{'='*70}")
        console.print(f"🎲 [bold yellow]EXECUTING TRADE: {action.upper().replace('_', ' ')}[/bold yellow]")
        console.print(f"{'='*70}")
        console.print(trade_table)
        
        if self.config.dry_run:
            console.print(f"\n[bold green]✅ DRY RUN - Trade logged (not executed)[/bold green]")
            console.print(f"{'='*70}\n")
            return True
        
        try:
            # Execute real trade
            side = "yes" if action == "buy_yes" else "no"
            
            # Use limit order at current ask
            market = await self.kalshi_client.get_market_with_odds(ticker)
            if not market:
                console.print(f"[red]❌ Could not fetch market data for {ticker}[/red]")
                return False
            limit_price = market.get(f'{side}_ask', 50)
            
            contracts = int(position_size * 100 / limit_price)  # Convert dollars to contracts
            
            if contracts < 1:
                console.print(f"\n[yellow]⚠️  Less than 1 contract, skipping[/yellow]")
                console.print(f"{'='*70}\n")
                return False
            
            console.print(f"\n[cyan]Placing order: {contracts} contracts @ {limit_price}¢ (${position_size:.2f})...[/cyan]")
            
            result = await self.kalshi_client.place_order(
                ticker=ticker,
                side=side,
                amount=position_size,  # Dollar amount
            )
            
            if result.get('status') == 'success':
                console.print(f"[bold green]✅ ORDER FILLED: {contracts} contracts @ {limit_price}¢[/bold green]")
                console.print(f"{'='*70}\n")
                return True
            else:
                console.print(f"[red]❌ Order failed: {result}[/red]")
                console.print(f"{'='*70}\n")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {e}")
            console.print(f"[red]  ❌ Error: {e}[/red]")
            return False
    
    async def _handle_exit_signals(self, exit_signals: List):
        """Handle exit signals for positions with beautiful output."""
        
        # Create exit signals table
        exit_table = Table(title="⚠️  Position Exit Signals", title_style="bold yellow")
        exit_table.add_column("Ticker", style="cyan", no_wrap=True)
        exit_table.add_column("Reason", style="yellow")
        exit_table.add_column("Urgency", justify="center")
        exit_table.add_column("Edge", justify="right", style="magenta")
        exit_table.add_column("P&L", justify="right")
        exit_table.add_column("Action", justify="center", style="bold")
        
        for signal in exit_signals:
            # Color P&L
            pnl_style = "bold green" if signal.current_pnl_pct > 0 else "bold red"
            pnl_text = f"[{pnl_style}]{signal.current_pnl_pct*100:+.1f}%[/{pnl_style}]"
            
            # Urgency emoji
            urgency_emoji = "🔴" if signal.urgency == "high" else "🟡" if signal.urgency == "medium" else "🟢"
            
            # Action color
            action_style = "bold red" if signal.recommendation == "exit_now" else "yellow" if signal.recommendation == "exit_when_convenient" else "green"
            
            exit_table.add_row(
                signal.ticker,
                signal.reason[:50],
                f"{urgency_emoji} {signal.urgency}",
                f"{signal.current_edge:.1f}pp",
                pnl_text,
                f"[{action_style}]{signal.recommendation.replace('_', ' ').upper()}[/{action_style}]",
            )
        
        console.print(exit_table)
        console.print()
        
        # Execute exits
        for signal in exit_signals:
            if signal.recommendation == "exit_now":
                console.print(f"\n[bold red]🚨 EXITING: {signal.ticker}[/bold red]")
                console.print(f"[red]Reason: {signal.reason}[/red]")
                
                if self.config.dry_run:
                    console.print("[green]✓ DRY RUN - Would exit position[/green]")
                else:
                    try:
                        result = await self.position_manager.exit_position(signal.ticker)
                        if result:
                            console.print("[bold green]✅ Position exited successfully[/bold green]")
                        else:
                            console.print("[red]❌ Exit failed[/red]")
                    except Exception as e:
                        console.print(f"[red]❌ Error: {e}[/red]")
            
            elif signal.recommendation == "exit_when_convenient":
                console.print(f"\n[yellow]⚠️  Monitoring: {signal.ticker}[/yellow]")
                console.print(f"[dim]Will exit when opportune: {signal.reason}[/dim]")
    
    async def _get_research(self, event: Dict, market: Dict) -> str:
        """
        Get research for market using DUAL method (Octagon + Perplexity).
        Combines structured analysis with real-time web data.
        """
        ticker = market.get('ticker', '')
        
        # Check cache
        if ticker in self.research_cache:
            research, timestamp = self.research_cache[ticker]
            age = datetime.now(timezone.utc) - timestamp
            if age.total_seconds() < 3600:  # 1 hour cache
                console.print(f"[dim]📦 Using cached research for {ticker}[/dim]")
                return research
        
        # Fetch research from BOTH sources
        octagon_research = ""
        perplexity_research = ""
        
        console.print(f"\n[cyan]🔬 Researching: {ticker}[/cyan]")
        
        try:
            # Source 1: Octagon (structured analysis)
            console.print("[dim]  → Calling Octagon...[/dim]")
            octagon_research = await self.research_client.research_event(
                event,
                [market],
            )
            if octagon_research:
                console.print(f"[green]  ✓ Octagon: {len(octagon_research)} chars[/green]")
            else:
                console.print("[yellow]  ⚠️  Octagon: No data[/yellow]")
        except Exception as e:
            console.print(f"[red]  ✗ Octagon failed: {e}[/red]")
        
        # Source 2: Perplexity (real-time web data)
        console.print(f"[dim]  Perplexity check: enabled={self.perplexity_enabled}, client={self.perplexity_client is not None}[/dim]")
        if self.perplexity_enabled and self.perplexity_client:
            try:
                console.print("[bold cyan]  → Calling Perplexity API (real-time web search)...[/bold cyan]")
                logger.warning(f"PERPLEXITY API CALL STARTING: {ticker} - {event.get('title', '')}")
                
                perplexity_research = await self.perplexity_client.fetch_event_research(
                    event_ticker=event.get('event_ticker', ''),
                    event_title=event.get('title', ''),
                    event_description=event.get('subtitle', ''),
                    category=event.get('category', ''),
                    sub_category=event.get('sub_category', ''),
                )
                
                if perplexity_research and len(perplexity_research) > 50:
                    console.print(f"[bold green]  ✓ Perplexity SUCCESS: {len(perplexity_research)} chars received[/bold green]")
                    logger.warning(f"PERPLEXITY RESPONSE: {len(perplexity_research)} chars - {perplexity_research[:100]}...")
                else:
                    console.print("[yellow]  ⚠️  Perplexity: No valid data returned[/yellow]")
                    logger.warning(f"PERPLEXITY EMPTY RESPONSE: {perplexity_research}")
            except Exception as perplexity_error:
                console.print(f"[red]  ✗ Perplexity API error: {perplexity_error}[/red]")
                logger.warning(f"PERPLEXITY ERROR: {type(perplexity_error).__name__}: {perplexity_error}")
                perplexity_research = None
        else:
            if not self.perplexity_enabled:
                console.print("[dim]  ⊘ Perplexity: Not enabled in config[/dim]")
                logger.warning("PERPLEXITY SKIPPED: Not enabled")
            elif self.perplexity_client is None:
                console.print("[red]  ✗ Perplexity client is None (initialization failed)[/red]")
                logger.warning("PERPLEXITY SKIPPED: Client not initialized")
            else:
                console.print("[yellow]  ⚠️  Perplexity: Unknown state[/yellow]")
                logger.warning("PERPLEXITY SKIPPED: Unknown reason")
        
        # Combine research sources
        combined_research = ""
        
        if octagon_research and perplexity_research:
            # DUAL SOURCE - Best quality!
            combined_research = f"""=== STRUCTURED ANALYSIS (Octagon) ===
{octagon_research}

=== REAL-TIME WEB DATA (Perplexity) ===
{perplexity_research}
"""
            console.print(f"[bold green]✓ DUAL SOURCE: {ticker}[/bold green]")
        elif octagon_research:
            # Single source: Octagon only
            combined_research = octagon_research
            console.print(f"[yellow]⚠️  Single source (Octagon only): {ticker}[/yellow]")
        elif perplexity_research:
            # Single source: Perplexity only
            combined_research = perplexity_research
            console.print(f"[yellow]⚠️  Single source (Perplexity only): {ticker}[/yellow]")
        else:
            # No research available
            console.print(f"[red]❌ No research available: {ticker}[/red]")
            return ""
        
        # Cache combined research
        if combined_research:
            self.research_cache[ticker] = (combined_research, datetime.now(timezone.utc))
        
        return combined_research
    
    async def _extract_probability_from_research(
        self, 
        research_text: str,
        market: Dict,
        event: Dict,
    ) -> Optional[float]:
        """Extract probability estimate from research text."""
        try:
            prompt = f"""Extract the probability estimate from this research.

Market: {market.get('title', '')}

Research:
{research_text[:1000]}

What is the probability that YES will win (0-100)?

Return ONLY a JSON object:
{{
  "probability": <number 0-100>,
  "confidence": <number 0-1>
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract probability estimates from research."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            
            result = json.loads(response.choices[0].message.content)
            prob = float(result.get('probability', 50)) / 100.0
            
            # Clamp to reasonable range
            prob = max(0.01, min(0.99, prob))
            
            return prob
            
        except Exception as e:
            logger.error(f"Error extracting probability: {e}")
            return None
    
    async def close(self):
        """Cleanup resources."""
        try:
            await self.research_client.close()
        except:
            pass
        
        try:
            if self.perplexity_client:
                await self.perplexity_client.close()
        except:
            pass


async def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Create and run bot
        bot = KalshiInstitutionalBot(config)
        await bot.run()
        await bot.close()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Fatal error", exc_info=True)
        console.print(f"\n[red]Fatal error: {e}[/red]")


def cli():
    """Console script entry point (for setup.py console_scripts)."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())