"""
Simple Kalshi trading bot with Octagon research and OpenAI decision making.
"""
import asyncio
import argparse
import json
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger
import re

from kalshi_client import KalshiClient
from research_client import OctagonClient
from betting_models import BettingDecision, MarketAnalysis, ProbabilityExtraction
from config import load_config
import openai


class SimpleTradingBot:
    """Simple trading bot that follows a clear workflow."""

    def __init__(self, live_trading: bool = False, max_close_ts: Optional[int] = None):
        self.config = load_config()
        # Override dry_run based on CLI parameter
        self.config.dry_run = not live_trading
        self.console = Console()
        self.kalshi_client = None
        self.research_client = None
        self.openai_client = None
        self.max_close_ts = max_close_ts

    async def initialize(self):
        """Initialize all API clients."""
        self.console.print("[bold blue]Initializing trading bot...[/bold blue]")

        # Initialize clients
        self.kalshi_client = KalshiClient(
            self.config.kalshi,
            self.config.minimum_time_remaining_hours,
            self.config.max_markets_per_event,
            max_close_ts=self.max_close_ts,
        )
        self.research_client = OctagonClient(self.config.octagon)
        self.openai_client = openai.AsyncOpenAI(api_key=self.config.openai.api_key)

        # Test connections
        await self.kalshi_client.login()
        self.console.print("[green]✓ Kalshi API connected[/green]")
        self.console.print("[green]✓ Octagon API ready[/green]")
        self.console.print("[green]✓ OpenAI API ready[/green]")

        # Show environment info
        env_color = "green" if self.config.kalshi.use_demo else "yellow"
        env_name = "DEMO" if self.config.kalshi.use_demo else "PRODUCTION"
        mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"

        self.console.print(f"\n[{env_color}]Environment: {env_name}[/{env_color}]")
        self.console.print(f"[blue]Mode: {mode}[/blue]")
        self.console.print(f"[blue]Max events to analyze: {self.config.max_events_to_analyze}[/blue]")
        self.console.print(f"[blue]Research batch size: {self.config.research_batch_size}[/blue]")
        self.console.print(f"[blue]Skip existing positions: {self.config.skip_existing_positions}[/blue]")
        self.console.print(
            f"[blue]Minimum time to event strike: "
            f"{self.config.minimum_time_remaining_hours} hours (for events with strike_date)[/blue]"
        )
        self.console.print(f"[blue]Max markets per event: {self.config.max_markets_per_event}[/blue]")
        self.console.print(f"[blue]Max bet amount: ${self.config.max_bet_amount}[/blue]")
        hedging_status = "Enabled" if self.config.enable_hedging else "Disabled"
        self.console.print(
            f"[blue]Risk hedging: {hedging_status} "
            f"(ratio: {self.config.hedge_ratio}, "
            f"min confidence: {self.config.min_confidence_for_hedging})[/blue]"
        )

        # Show risk-adjusted trading settings
        z_threshold = float(getattr(self.config, "z_threshold", 1.5))
        self.console.print(f"[blue]R-score filtering: Enabled (z-threshold: {z_threshold})[/blue]")
        if self.config.enable_kelly_sizing:
            self.console.print(
                f"[blue]Kelly sizing: Enabled (fraction: {self.config.kelly_fraction}, "
                f"bankroll: ${self.config.bankroll})[/blue]"
            )
        self.console.print(
            f"[blue]Portfolio selection: {self.config.portfolio_selection_method} "
            f"(max positions: {self.config.max_portfolio_positions})[/blue]\n"
        )
        if self.max_close_ts is not None:
            hours_from_now = (self.max_close_ts - int(time.time())) / 3600
            self.console.print(
                f"[blue]Market expiration filter: close before ~{hours_from_now:.1f} hours from now[/blue]"
            )

    # ---------- Risk metrics helpers ----------

    def calculate_risk_adjusted_metrics(self, research_prob: float, market_price: float, action: str) -> dict:
        """
        Calculate hedge-fund style risk-adjusted metrics.

        Args:
            research_prob: Research probability (0-1)
            market_price: Market price (0-1)
            action: "buy_yes" or "buy_no"

        Returns:
            dict with expected_return, r_score, kelly_fraction
        """
        try:
            if action == "buy_yes":
                p = research_prob
                y = market_price
            elif action == "buy_no":
                p = 1 - research_prob
                y = market_price
            else:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}

            if y <= 0 or y >= 1 or p <= 0 or p >= 1:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}

            # Expected return on capital
            expected_return = (p - y) / y

            # R-score (z-score style)
            variance = p * (1 - p)
            if variance <= 0:
                return {"expected_return": expected_return, "r_score": 0.0, "kelly_fraction": 0.0}
            r_score = (p - y) / math.sqrt(variance)

            # Kelly fraction
            if y >= 1:
                kelly_fraction = 0.0
            else:
                kelly_fraction = (p - y) / (1 - y)
                kelly_fraction = max(0.0, min(1.0, kelly_fraction))

            return {
                "expected_return": expected_return,
                "r_score": r_score,
                "kelly_fraction": kelly_fraction,
            }
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}

    def calculate_kelly_position_size(self, kelly_fraction: float) -> float:
        """
        Calculate position size using fractional Kelly criterion.

        Args:
            kelly_fraction: Optimal Kelly fraction (0-1)

        Returns:
            Position size in dollars
        """
        max_bet_amount = float(self.config.max_bet_amount)

        if not self.config.enable_kelly_sizing or kelly_fraction <= 0:
            return max_bet_amount

        adjusted_kelly = kelly_fraction * float(self.config.kelly_fraction)
        bankroll = float(self.config.bankroll)
        max_kelly_bet_fraction = float(self.config.max_kelly_bet_fraction)

        kelly_bet_size = bankroll * adjusted_kelly

        max_allowed = bankroll * max_kelly_bet_fraction
        kelly_bet_size = min(kelly_bet_size, max_allowed)
        kelly_bet_size = min(kelly_bet_size, max_bet_amount)
        kelly_bet_size = max(kelly_bet_size, 1.0)

        return float(kelly_bet_size)

    # ---------- Portfolio selection ----------

    def apply_portfolio_selection(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """
        Apply portfolio selection to hold only the N highest R-scores.
        Step 4: Portfolio view - hold the N highest R-scores subject to limits.
        """
        if self.config.portfolio_selection_method == "legacy":
            return analysis

        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        skip_decisions = [d for d in analysis.decisions if d.action == "skip"]

        if not actionable_decisions:
            return analysis

        if self.config.portfolio_selection_method == "top_r_scores":
            actionable_decisions.sort(
                key=lambda d: (d.r_score if d.r_score is not None else -999.0),
                reverse=True,
            )
            max_positions = int(self.config.max_portfolio_positions)
            selected_decisions = actionable_decisions[:max_positions]

            rejected_decisions = []
            rank_counter = len(selected_decisions) + 1
            for decision in actionable_decisions[max_positions:]:
                r_val = decision.r_score if decision.r_score is not None else float("nan")
                reason = (
                    f"Portfolio limit: R-score {r_val:.2f} "
                    f"ranked #{rank_counter}"
                )
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=reason,
                    event_name=decision.event_name,
                    market_name=decision.market_name,
                    expected_return=decision.expected_return,
                    r_score=decision.r_score,
                    kelly_fraction=decision.kelly_fraction,
                    market_price=decision.market_price,
                    research_probability=decision.research_probability,
                    is_hedge=decision.is_hedge,
                    hedge_for=decision.hedge_for,
                    hedge_ratio=decision.hedge_ratio,
                )
                rejected_decisions.append(skip_decision)
                rank_counter += 1

            if rejected_decisions:
                logger.info(
                    f"Portfolio selection for {event_ticker}: kept top {len(selected_decisions)} "
                    f"positions, rejected {len(rejected_decisions)} lower R-score positions"
                )

            analysis.decisions = selected_decisions + skip_decisions + rejected_decisions

        elif self.config.portfolio_selection_method == "diversified":
            original_method = self.config.portfolio_selection_method
            self.config.portfolio_selection_method = "top_r_scores"
            analysis = self.apply_portfolio_selection(analysis, event_ticker)
            self.config.portfolio_selection_method = original_method

        # Recompute totals after selection
        actionable = [d for d in analysis.decisions if d.action != "skip"]
        analysis.total_recommended_bet = float(sum(d.amount for d in actionable))
        analysis.high_confidence_bets = len([d for d in actionable if d.confidence > 0.7])
        return analysis

    # ---------- Event + market discovery ----------

    async def get_top_events(self) -> List[Dict[str, Any]]:
        """
        Get top non-sports events, preferring political ones, sorted by 24-hour volume.
        """
        self.console.print("[bold]Step 1: Fetching top political / non-sports events...[/bold]")

        political_keywords = [
            "politic",
            "election",
            "primary",
            "president",
            "presidential",
            "senate",
            "senator",
            "house",
            "congress",
            "governor",
            "mayor",
            "parliament",
            "ballot",
            "referendum",
            "white house",
            "supreme court",
            "scotus",
        ]

        sports_keywords = [
            "nfl",
            "nba",
            "mlb",
            "nhl",
            "ncaa",
            "football",
            "basketball",
            "baseball",
            "hockey",
            "soccer",
            "tennis",
            "golf",
            "sports",
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching events...", total=None)

            try:
                fetch_limit = self.config.max_events_to_analyze * 3
                events = await self.kalshi_client.get_events(limit=fetch_limit)
                progress.update(task, completed=1)

                if not events:
                    self.console.print("[red]No events returned from Kalshi[/red]")
                    return []

                # 1) Exclude obvious sports events up front
                non_sports_events: List[Dict[str, Any]] = []
                for ev in events:
                    cat = str(ev.get("category", "")).lower()
                    ticker = str(ev.get("event_ticker", "")).lower()
                    title = str(ev.get("title", "")).lower()
                    blob = " ".join([cat, ticker, title])
                    if "sport" in cat or any(sk in blob for sk in sports_keywords):
                        continue
                    non_sports_events.append(ev)

                if non_sports_events:
                    self.console.print(
                        f"[blue]• Fetched {len(events)} events, "
                        f"{len(non_sports_events)} are non-sports[/blue]"
                    )
                else:
                    # If everything looks like sports, fall back but warn
                    non_sports_events = events
                    self.console.print(
                        "[yellow]Warning: all events look like sports; keeping all for now[/yellow]"
                    )

                # 2) Among non-sports, try to find political-looking ones
                political_events: List[Dict[str, Any]] = []
                for ev in non_sports_events:
                    text_parts = [
                        str(ev.get("category", "")),
                        str(ev.get("title", "")),
                        str(ev.get("subtitle", "")),
                        str(ev.get("event_ticker", "")),
                    ]
                    blob = " ".join(text_parts).lower()
                    if any(kw in blob for kw in political_keywords):
                        political_events.append(ev)

                if political_events:
                    filtered_events = political_events
                    self.console.print(
                        f"[blue]• Found {len(political_events)} political-looking events "
                        f"within non-sports set[/blue]"
                    )
                else:
                    filtered_events = non_sports_events
                    self.console.print(
                        f"[yellow]Warning: did not find clear political events, "
                        f"using all {len(filtered_events)} non-sports events[/yellow]"
                    )

                # 3) Optional: light liquidity filter
                min_volume_24h = float(getattr(self.config, "min_volume_24h", 0) or 0)
                if min_volume_24h > 0:
                    before = len(filtered_events)
                    filtered_events = [
                        ev
                        for ev in filtered_events
                        if float(ev.get("volume_24h") or 0) >= min_volume_24h
                    ]
                    self.console.print(
                        f"[blue]• Liquidity filter: {before} → {len(filtered_events)} "
                        f"events with volume_24h ≥ {min_volume_24h}[/blue]"
                    )

                # 4) Sort by 24h volume
                filtered_events.sort(
                    key=lambda e: e.get("volume_24h", 0) or 0,
                    reverse=True,
                )

                # 5) Truncate to configured max
                if len(filtered_events) > self.config.max_events_to_analyze:
                    filtered_events = filtered_events[: self.config.max_events_to_analyze]

                events = filtered_events
                self.console.print(
                    f"[green]✓ Selected {len(events)} events for research[/green]"
                )

                # Preview table
                table = Table(title="Top Target Events by 24h Volume")
                table.add_column("Event Ticker", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("24h Volume", style="magenta", justify="right")
                table.add_column("Time Remaining", style="blue", justify="right")
                table.add_column("Category", style="green")
                table.add_column("Mutually Exclusive", style="red", justify="center")

                now_ts = int(time.time())

                for event in events[:10]:
                    time_remaining = event.get("time_remaining_hours")
                    if time_remaining is None:
                        close_ts = event.get("close_ts")
                        if isinstance(close_ts, (int, float)):
                            time_remaining = max(0, (close_ts - now_ts) / 3600)
                        else:
                            time_remaining = None

                    if time_remaining is None:
                        time_str = "No date set"
                    elif time_remaining > 24:
                        time_str = f"{time_remaining / 24:.1f} days"
                    else:
                        time_str = f"{time_remaining:.1f} hours"

                    title = event.get("title", "N/A")
                    truncated_title = title[:35] + "..." if len(title) > 35 else title

                    table.add_row(
                        event.get("event_ticker", "N/A"),
                        truncated_title,
                        f"{event.get('volume_24h', 0):,}",
                        time_str,
                        event.get("category", "N/A"),
                        "YES" if event.get("mutually_exclusive", False) else "NO",
                    )

                self.console.print(table)
                return events

            except Exception as e:
                self.console.print(f"[red]Error fetching events: {e}[/red]")
                return []

    async def get_markets_for_events(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get markets for each event (uses pre-loaded markets from events)."""
        self.console.print(
            f"\n[bold]Step 2: Processing markets for {len(events)} events...[/bold]"
        )

        event_markets: Dict[str, Dict[str, Any]] = {}

        for event in events:
            event_ticker = event.get("event_ticker", "")
            if not event_ticker:
                continue

            markets = event.get("markets", [])
            total_markets = event.get("total_markets", len(markets))

            if markets:
                simple_markets = []
                for market in markets:
                    simple_markets.append(
                        {
                            "ticker": market.get("ticker", ""),
                            "title": market.get("title", ""),
                            "subtitle": market.get("subtitle", ""),
                            "volume": market.get("volume", 0),
                            "open_time": market.get("open_time", ""),
                            "close_time": market.get("close_time", ""),
                        }
                    )

                event_markets[event_ticker] = {"event": event, "markets": simple_markets}

                if total_markets > len(markets):
                    self.console.print(
                        f"[green]✓ Using top {len(markets)} markets for "
                        f"{event_ticker} (from {total_markets} total)[/green]"
                    )
                else:
                    self.console.print(
                        f"[green]✓ Using {len(markets)} markets for {event_ticker}[/green]"
                    )
            else:
                self.console.print(
                    f"[yellow]⚠ No markets found for {event_ticker}[/yellow]"
                )

        total_markets = sum(len(data["markets"]) for data in event_markets.values())
        self.console.print(
            f"[green]✓ Processing {total_markets} total markets across "
            f"{len(event_markets)} events[/green]"
        )
        return event_markets

    async def filter_markets_by_positions(
        self, event_markets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter out markets where we already have positions to save research time."""
        if self.config.dry_run or not self.config.skip_existing_positions:
            return event_markets

        self.console.print(
            f"\n[bold]Step 2.5: Filtering markets by existing positions...[/bold]"
        )

        filtered_event_markets: Dict[str, Dict[str, Any]] = {}
        total_markets_after = 0
        skipped_markets = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            total_markets = sum(len(data["markets"]) for data in event_markets.values())
            task = progress.add_task(
                "Checking existing positions...", total=total_markets
            )

            for event_ticker, data in event_markets.items():
                event = data["event"]
                markets = data["markets"]

                event_has_positions = False
                markets_checked = 0

                for market in markets:
                    ticker = market.get("ticker", "")
                    if not ticker:
                        progress.update(task, advance=1)
                        markets_checked += 1
                        continue

                    try:
                        has_position = await self.kalshi_client.has_position_in_market(
                            ticker
                        )
                        if has_position:
                            self.console.print(
                                f"[yellow]⚠ Found position in {ticker}[/yellow]"
                            )
                            event_has_positions = True
                            remaining = len(markets) - markets_checked - 1
                            progress.update(task, advance=remaining + 1)
                            break
                    except Exception as e:
                        logger.warning(
                            f"Could not check position for {ticker}: {e}"
                        )

                    progress.update(task, advance=1)
                    markets_checked += 1

                if event_has_positions:
                    skipped_markets += len(markets)
                    self.console.print(
                        f"[yellow]⚠ Skipping entire event {event_ticker}: "
                        f"Has existing positions[/yellow]"
                    )
                else:
                    filtered_event_markets[event_ticker] = {
                        "event": event,
                        "markets": markets,
                    }
                    total_markets_after += len(markets)
                    self.console.print(
                        f"[green]✓ Keeping entire event {event_ticker}: "
                        f"No existing positions[/green]"
                    )

        events_skipped = len(event_markets) - len(filtered_event_markets)
        self.console.print(f"\n[blue]Position filtering summary:[/blue]")
        self.console.print(
            f"[blue]• Events before filtering: {len(event_markets)}[/blue]"
        )
        self.console.print(
            f"[blue]• Events after filtering: {len(filtered_event_markets)}[/blue]"
        )
        self.console.print(
            f"[blue]• Events skipped (existing positions): {events_skipped}[/blue]"
        )
        self.console.print(f"[blue]• Markets in skipped events: {skipped_markets}[/blue]")
        self.console.print(
            f"[blue]• Markets remaining for research: {total_markets_after}[/blue]"
        )

        if len(filtered_event_markets) == 0:
            self.console.print(
                "[yellow]⚠ No events remaining after position filtering[/yellow]"
            )
        elif events_skipped > 0:
            time_saved_estimate = events_skipped * 3
            self.console.print(
                f"[green]✓ Estimated time saved by skipping research: "
                f"~{time_saved_estimate} minutes[/green]"
            )

        return filtered_event_markets

    # ---------- Probability extraction ----------

    def _parse_probabilities_from_research(
        self, research_text: str, markets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Parse probability predictions from research text."""
        probabilities: Dict[str, float] = {}

        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            if not ticker:
                continue

            search_terms = [ticker]
            if title:
                search_terms.append(title)
                title_words = [
                    w
                    for w in title.split()
                    if len(w) > 3
                    and w.lower()
                    not in {
                        "will",
                        "the",
                        "win",
                        "this",
                        "that",
                        "with",
                        "have",
                        "from",
                        "into",
                        "about",
                        "over",
                        "under",
                        "again",
                        "then",
                        "once",
                        "they",
                        "them",
                        "what",
                        "when",
                        "where",
                        "which",
                        "your",
                        "their",
                        "there",
                    }
                ]
                search_terms.extend(title_words)

            found_probability: Optional[float] = None
            for term in search_terms:
                if not term:
                    continue

                patterns = [
                    rf"{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}[:\s]*(\d+)%",
                    rf"(\d+\.?\d*)%[:\s]*{re.escape(term)}",
                    rf"(\d+)%[:\s]*{re.escape(term)}",
                    rf"probability.*{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*probability.*?(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*(\d+\.?\d*)%.*probability",
                    rf"probability.*(\d+\.?\d*)%.*{re.escape(term)}",
                    rf"{re.escape(term)}.*?(\d+\.?\d*)%",
                    rf"(\d+\.?\d*)%.*?{re.escape(term)}",
                ]

                for pattern in patterns:
                    matches = re.findall(
                        pattern, research_text, re.IGNORECASE | re.DOTALL
                    )
                    if matches:
                        try:
                            prob = float(matches[0])
                            if 0 <= prob <= 100:
                                found_probability = prob
                                break
                        except ValueError:
                            continue

                if found_probability is not None:
                    break

            if found_probability is not None:
                probabilities[ticker] = found_probability
                logger.info(f"Found probability for {ticker}: {found_probability}%")
            else:
                logger.warning(
                    f"No probability found for {ticker} (title: {title})"
                )

        return probabilities

    def _is_prob_extraction_trustworthy(
        self, extraction: ProbabilityExtraction, is_mutually_exclusive: bool
    ) -> bool:
        """Heuristic sanity checks on LLM / parsed probabilities."""
        try:
            probs = [float(mp.research_probability) for mp in extraction.markets]
            if not probs:
                return False

            if is_mutually_exclusive:
                s = sum(probs)
                if s < 50 or s > 200:
                    logger.warning(
                        f"Probability sum {s:.1f} out of range for mutually "
                        f"exclusive event"
                    )
                    return False

            mean = sum(probs) / len(probs)
            variance = sum((p - mean) ** 2 for p in probs) / len(probs)
            if variance < 5.0:
                logger.warning(
                    f"Probability extraction variance {variance:.2f} too low; "
                    f"treating as uninformative"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error in probability trust check: {e}")
            return False

    def _postprocess_probability_extraction(
        self,
        event_ticker: str,
        extraction: ProbabilityExtraction,
        event_markets: Dict[str, Dict[str, Any]],
    ) -> Optional[ProbabilityExtraction]:
        """
        Clamp probabilities, renormalize mutually exclusive events, and
        discard obviously bad extractions.
        """
        try:
            event_data = event_markets.get(event_ticker, {})
            event_info = event_data.get("event", {})
            is_mutually_exclusive = bool(event_info.get("mutually_exclusive", False))

            # Clamp probabilities to [1, 99] and fix NaNs
            for mp in extraction.markets:
                p = float(mp.research_probability)
                if math.isnan(p) or math.isinf(p):
                    p = 50.0
                mp.research_probability = max(1.0, min(99.0, p))

            # Renormalize for mutually exclusive events
            if is_mutually_exclusive and extraction.markets:
                total = sum(mp.research_probability for mp in extraction.markets)
                if total > 0:
                    for mp in extraction.markets:
                        mp.research_probability = mp.research_probability * 100.0 / total

            # Trust check
            if not self._is_prob_extraction_trustworthy(
                extraction, is_mutually_exclusive
            ):
                logger.warning(
                    f"Probability extraction for {event_ticker} failed trust checks; "
                    f"skipping event."
                )
                return None

            return extraction
        except Exception as e:
            logger.error(
                f"Error post-processing probabilities for {event_ticker}: {e}"
            )
            return None

    async def research_events(
        self, event_markets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Research each event and its markets using Octagon Deep Research."""
        self.console.print(
            f"\n[bold]Step 3: Researching {len(event_markets)} events...[/bold]"
        )

        events_with_markets: Dict[str, Dict[str, Any]] = {}
        events_skipped_empty = 0
        for event_ticker, data in event_markets.items():
            markets = data.get("markets", [])
            if markets:
                events_with_markets[event_ticker] = data
            else:
                events_skipped_empty += 1
                self.console.print(
                    f"[yellow]⚠ Skipping {event_ticker}: No markets to research[/yellow]"
                )

        if events_skipped_empty > 0:
            self.console.print(
                f"[yellow]⚠ Skipped {events_skipped_empty} events with no markets[/yellow]"
            )

        if not events_with_markets:
            self.console.print("[red]✗ No events with markets to research[/red]")
            return {}

        research_results: Dict[str, str] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Researching events...", total=len(events_with_markets)
            )

            batch_size = self.config.research_batch_size
            event_items = list(events_with_markets.items())

            for i in range(0, len(event_items), batch_size):
                batch = event_items[i : i + batch_size]
                self.console.print(
                    f"[blue]Processing research batch {i // batch_size + 1} "
                    f"with {len(batch)} events[/blue]"
                )

                tasks = []
                for event_ticker, data in batch:
                    event = data["event"]
                    markets = data["markets"]
                    if event and markets:
                        coro = self.research_client.research_event(event, markets)
                        tasks.append(
                            asyncio.wait_for(
                                coro,
                                timeout=self.config.research_timeout_seconds,
                            )
                        )
                    else:
                        tasks.append(asyncio.sleep(0, result=None))

                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for (event_ticker, _), result in zip(batch, results):
                        if not isinstance(result, Exception) and result:
                            research_results[event_ticker] = result
                            progress.update(task, advance=1)
                            self.console.print(
                                f"[green]✓ Researched {event_ticker}[/green]"
                            )
                        else:
                            if isinstance(result, asyncio.TimeoutError):
                                err = (
                                    f"Timeout after "
                                    f"{self.config.research_timeout_seconds}s"
                                )
                            elif result is None:
                                err = "No result returned"
                            elif isinstance(result, Exception):
                                err = str(result)
                            else:
                                err = "Unknown error"

                            self.console.print(
                                f"[red]✗ Failed to research {event_ticker}: "
                                f"{err}[/red]"
                            )
                            progress.update(task, advance=1)
                except Exception as e:
                    self.console.print(f"[red]Batch research error: {e}[/red]")
                    progress.update(task, advance=len(batch))

                await asyncio.sleep(1)

        self.console.print(
            f"[green]✓ Completed research on {len(research_results)} events[/green]"
        )
        return research_results

    async def _extract_probabilities_for_event(
        self,
        event_ticker: str,
        research_text: str,
        event_markets: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Optional[ProbabilityExtraction]]:
        """
        Extract probabilities for a single event by parsing the research text.
        """
        try:
            event_data = event_markets.get(event_ticker, {})
            markets = event_data.get("markets", [])
            event_info = event_data.get("event", {})

            if not markets:
                logger.warning(
                    f"No markets found for event {event_ticker} when extracting "
                    f"probabilities."
                )
                return event_ticker, None

            parsed_probs = self._parse_probabilities_from_research(research_text, markets)

            if not parsed_probs:
                logger.warning(
                    f"No probabilities could be parsed from research for "
                    f"event {event_ticker}."
                )
                return event_ticker, None

            markets_payload = []
            for market in markets:
                ticker = market.get("ticker", "")
                if not ticker:
                    continue
                prob = parsed_probs.get(ticker)
                if prob is None:
                    continue

                confidence = 0.7
                markets_payload.append(
                    {
                        "ticker": ticker,
                        "title": market.get("title", ""),
                        "research_probability": float(prob),
                        "reasoning": (
                            f"Probability {prob:.1f}% inferred from research text "
                            f"for {ticker}."
                        ),
                        "confidence": float(confidence),
                    }
                )

            if not markets_payload:
                logger.warning(
                    f"Parsed probabilities for event {event_ticker}, "
                    f"but none matched specific markets."
                )
                return event_ticker, None

            overall_summary = (
                f"Probabilities parsed from research text for event "
                f"{event_info.get('title', event_ticker)}."
            )

            extraction = ProbabilityExtraction(
                markets=markets_payload,
                overall_summary=overall_summary,
            )

            extraction = self._postprocess_probability_extraction(
                event_ticker, extraction, event_markets
            )
            return event_ticker, extraction
        except Exception as e:
            logger.error(f"Error extracting probabilities for {event_ticker}: {e}")
            return event_ticker, None

    async def extract_probabilities(
        self,
        research_results: Dict[str, str],
        event_markets: Dict[str, Dict[str, Any]],
    ) -> Dict[str, ProbabilityExtraction]:
        """
        Extract structured probabilities from research results.
        """
        self.console.print(
            f"\n[bold]Step 3.5: Extracting probabilities from research.[/bold]"
        )

        probability_extractions: Dict[str, ProbabilityExtraction] = {}

        if not research_results:
            self.console.print(
                "[yellow]No research results to extract probabilities from[/yellow]"
            )
            return probability_extractions

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Extracting probabilities...", total=len(research_results)
            )

            tasks = []
            for event_ticker, research_text in research_results.items():
                task_coroutine = self._extract_probabilities_for_event(
                    event_ticker, research_text, event_markets
                )
                tasks.append(task_coroutine)

            for coro in asyncio.as_completed(tasks):
                try:
                    event_ticker, extraction = await coro
                    if extraction is not None:
                        probability_extractions[event_ticker] = extraction
                    progress.update(task, advance=1)
                except Exception as e:
                    logger.error(f"Batch probability extraction error: {e}")
                    progress.update(task, advance=1)

        self.console.print(
            f"[green]✓ Extracted probabilities for "
            f"{len(probability_extractions)} events[/green]"
        )

        if not probability_extractions:
            self.console.print(
                "[yellow]No probability extractions succeeded. "
                "Check research output formatting and parsing.[/yellow]"
            )

        return probability_extractions

    # ---------- Market odds ----------

    async def get_market_odds(
        self, event_markets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch current market odds for all markets."""
        self.console.print(
            f"\n[bold]Step 4: Fetching current market odds...[/bold]"
        )

        market_odds: Dict[str, Dict[str, Any]] = {}
        all_tickers: List[str] = []

        for _, data in event_markets.items():
            for market in data["markets"]:
                ticker = market.get("ticker", "")
                if ticker:
                    all_tickers.append(ticker)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Fetching market odds...", total=len(all_tickers)
            )

            batch_size = 20
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i : i + batch_size]

                tasks = [
                    self.kalshi_client.get_market_with_odds(ticker) for ticker in batch
                ]

                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for ticker, result in zip(batch, results):
                        if not isinstance(result, Exception) and result:
                            # Pre-compute mid prices to simplify later logic
                            yes_bid = result.get("yes_bid", 0)
                            yes_ask = result.get("yes_ask", 0)
                            no_bid = result.get("no_bid", 0)
                            no_ask = result.get("no_ask", 0)

                            def _mid(b: float, a: float) -> Optional[float]:
                                if b > 0 and a > 0:
                                    return (b + a) / 2.0
                                if a > 0:
                                    return float(a)
                                if b > 0:
                                    return float(b)
                                return None

                            yes_mid = _mid(yes_bid, yes_ask)
                            no_mid = _mid(no_bid, no_ask)

                            result["yes_mid_price"] = yes_mid
                            result["no_mid_price"] = no_mid

                            market_odds[ticker] = result
                            progress.update(task, advance=1)
                        else:
                            self.console.print(
                                f"[red]✗ Failed to get odds for {ticker}[/red]"
                            )
                            progress.update(task, advance=1)
                except Exception as e:
                    self.console.print(f"[red]Batch odds fetch error: {e}[/red]")
                    progress.update(task, advance=len(batch))

                await asyncio.sleep(0.2)

        self.console.print(
            f"[green]✓ Fetched odds for {len(market_odds)} markets[/green]"
        )
        return market_odds

    # ---------- Betting decision engine ----------

    async def _get_event_betting_decisions(
        self,
        event_ticker: str,
        event_data: Dict[str, Any],
        probability_extraction: ProbabilityExtraction,
        market_odds: Dict[str, Dict[str, Any]],
    ) -> MarketAnalysis:
        """
        Deterministic decision engine that converts probabilities + odds into
        a MarketAnalysis, with configurable risk controls.
        """
        try:
            event_info = event_data["event"]
            markets = event_data["markets"]

            markets_by_ticker = {m.get("ticker", ""): m for m in markets}
            z_threshold = float(getattr(self.config, "z_threshold", 1.5))
            min_conf_to_bet = float(getattr(self.config, "min_confidence_to_bet", 0.7))
            min_edge_points = float(getattr(self.config, "min_edge_points", 10.0))

            decisions: List[BettingDecision] = []
            total_bet = 0.0
            high_conf_count = 0

            for mp in probability_extraction.markets:
                ticker = mp.ticker
                if ticker not in markets_by_ticker:
                    continue

                odds = market_odds.get(ticker, {})
                yes_mid_cents = odds.get("yes_mid_price")
                if yes_mid_cents is None or yes_mid_cents <= 0:
                    logger.info(
                        f"Skipping {ticker}: invalid yes_mid_price={yes_mid_cents}"
                    )
                    continue

                market_prob = yes_mid_cents / 100.0
                research_prob = float(mp.research_probability) / 100.0

                edge_points = (research_prob - market_prob) * 100.0
                if edge_points < min_edge_points:
                    logger.info(
                        f"Skipping {ticker}: edge {edge_points:.1f} < "
                        f"min_edge_points {min_edge_points:.1f}"
                    )
                    continue

                metrics = self.calculate_risk_adjusted_metrics(
                    research_prob, market_prob, "buy_yes"
                )
                r_score = metrics["r_score"]
                if r_score < z_threshold:
                    logger.info(
                        f"Skipping {ticker}: R-score {r_score:.2f} < "
                        f"z_threshold {z_threshold:.2f}"
                    )
                    continue

                # Base confidence from extraction or derived from R-score
                if mp.confidence is not None:
                    confidence = float(mp.confidence)
                else:
                    confidence = 0.6 + max(0.0, min(0.35, (r_score - z_threshold) / 5.0))

                if confidence < min_conf_to_bet:
                    logger.info(
                        f"Skipping {ticker}: confidence {confidence:.2f} < "
                        f"min_conf_to_bet {min_conf_to_bet:.2f}"
                    )
                    continue

                # Position sizing
                if self.config.enable_kelly_sizing:
                    kelly_size = self.calculate_kelly_position_size(
                        metrics["kelly_fraction"]
                    )
                    amount = min(kelly_size, float(self.config.max_bet_amount))
                else:
                    amount = float(self.config.max_bet_amount)

                amount = max(1.0, amount)

                market_info = markets_by_ticker[ticker]
                event_title = event_info.get("title", "")
                market_title = market_info.get("title", "")

                reasoning = (
                    f"Positive edge: research {research_prob*100:.1f}% vs market "
                    f"{market_prob*100:.1f}% (edge {edge_points:.1f} pts, "
                    f"R-score {r_score:.2f})."
                )

                decision = BettingDecision(
                    ticker=ticker,
                    action="buy_yes",
                    confidence=confidence,
                    amount=amount,
                    reasoning=reasoning,
                    event_name=event_title,
                    market_name=market_title,
                    expected_return=metrics["expected_return"],
                    r_score=metrics["r_score"],
                    kelly_fraction=metrics["kelly_fraction"],
                    market_price=market_prob,
                    research_probability=research_prob,
                    is_hedge=False,
                    hedge_for=None,
                    hedge_ratio=None,
                )

                decisions.append(decision)
                total_bet += amount
                if confidence >= 0.8:
                    high_conf_count += 1

            if not decisions:
                return MarketAnalysis(
                    decisions=[],
                    total_recommended_bet=0.0,
                    high_confidence_bets=0,
                    summary=f"No actionable edges detected for event {event_ticker}.",
                )

            analysis = MarketAnalysis(
                decisions=decisions,
                total_recommended_bet=float(total_bet),
                high_confidence_bets=high_conf_count,
                summary=(
                    f"Generated {len(decisions)} bets based on edge logic; "
                    f"total suggested: ${total_bet:.2f}."
                ),
            )

            # Portfolio selection within event
            analysis = self.apply_portfolio_selection(analysis, event_ticker)

            return analysis
        except Exception as e:
            logger.error(f"Error generating decisions for {event_ticker}: {e}")
            return MarketAnalysis(
                decisions=[],
                total_recommended_bet=0.0,
                high_confidence_bets=0,
                summary=f"Error generating decisions for {event_ticker}",
            )

    async def _get_betting_decisions_for_event(
        self,
        event_ticker: str,
        event_data: Dict[str, Any],
        extraction: ProbabilityExtraction,
        market_odds: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Optional[MarketAnalysis]]:
        """
        Wrapper: returns (event_ticker, MarketAnalysis|None) for parallel use.
        """
        try:
            analysis = await self._get_event_betting_decisions(
                event_ticker, event_data, extraction, market_odds
            )
            return event_ticker, analysis
        except Exception as e:
            logger.error(
                f"Error in _get_betting_decisions_for_event({event_ticker}): {e}"
            )
            return event_ticker, None

    # ---------- Display helpers ----------

    def _generate_readable_market_name(self, ticker: str) -> str:
        """Generate a readable market name from ticker."""
        return ticker.replace("-", " ").replace("_", " ").title()

    def _display_event_decisions(
        self, event_ticker: str, event_analysis: MarketAnalysis
    ):
        """Display the betting decisions for a single event."""
        actionable_decisions = [
            d for d in event_analysis.decisions if d.action != "skip"
        ]

        if not actionable_decisions:
            self.console.print(f"[yellow]No actionable decisions for {event_ticker}[/yellow]")
            return

        event_name = actionable_decisions[0].event_name or event_ticker
        table = Table(title=f"Betting Decisions for {event_name}", show_lines=True)
        table.add_column("Type", style="bright_blue", justify="center", width=8)
        table.add_column("Market", style="cyan", width=40)
        table.add_column("Action", style="yellow", justify="center", width=10)
        table.add_column("Confidence", style="magenta", justify="right", width=10)
        table.add_column("Amount", style="green", justify="right", width=10)
        table.add_column("Reasoning", style="blue", width=70)

        for decision in actionable_decisions:
            market_display = (
                decision.market_name
                if decision.market_name
                else self._generate_readable_market_name(decision.ticker)
            )
            bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"

            table.add_row(
                bet_type,
                market_display,
                decision.action.upper().replace("_", " "),
                f"{decision.confidence:.2f}",
                f"${decision.amount:.2f}",
                decision.reasoning,
            )

        self.console.print(table)

        if event_analysis.total_recommended_bet > 0:
            self.console.print(
                f"[blue]Event total: ${event_analysis.total_recommended_bet:.2f} | "
                f"High confidence: {event_analysis.high_confidence_bets}[/blue]"
            )

    # ---------- Hedging + global decision aggregation ----------

    def _generate_hedge_decisions(
        self, main_decisions: List[BettingDecision]
    ) -> List[BettingDecision]:
        """Generate hedge decisions to minimize risk for main betting decisions."""
        if not self.config.enable_hedging:
            return []

        hedge_decisions: List[BettingDecision] = []
        min_conf_for_hedge = float(
            getattr(self.config, "min_confidence_for_hedging", 0.75)
        )

        for main_decision in main_decisions:
            if main_decision.is_hedge or main_decision.action == "skip":
                continue

            if main_decision.confidence >= min_conf_for_hedge:
                continue

            hedge_amount = min(
                main_decision.amount * float(self.config.hedge_ratio),
                float(self.config.max_hedge_amount),
            )
            if hedge_amount < 1.0:
                continue

            hedge_action = (
                "buy_no" if main_decision.action == "buy_yes" else "buy_yes"
            )

            hedge_decision = BettingDecision(
                ticker=main_decision.ticker,
                action=hedge_action,
                confidence=0.8,
                amount=float(hedge_amount),
                reasoning=(
                    f"Risk hedge: {float(self.config.hedge_ratio)*100:.0f}% hedge for "
                    f"{main_decision.action} (confidence "
                    f"{main_decision.confidence:.2f} < {min_conf_for_hedge:.2f})"
                ),
                event_name=main_decision.event_name,
                market_name=main_decision.market_name,
                is_hedge=True,
                hedge_for=main_decision.ticker,
                hedge_ratio=float(self.config.hedge_ratio),
            )

            hedge_decisions.append(hedge_decision)

        return hedge_decisions

    async def get_betting_decisions(
        self,
        event_markets: Dict[str, Dict[str, Any]],
        probability_extractions: Dict[str, ProbabilityExtraction],
        market_odds: Dict[str, Dict[str, Any]],
    ) -> MarketAnalysis:
        """Generate betting decisions per event in parallel."""
        self.console.print(
            f"\n[bold]Step 5: Generating betting decisions...[/bold]"
        )

        all_decisions: List[BettingDecision] = []
        total_recommended_bet = 0.0
        high_confidence_bets = 0
        event_summaries: List[str] = []

        processable_events = [
            (event_ticker, data)
            for event_ticker, data in event_markets.items()
            if event_ticker in probability_extractions and data["markets"]
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Generating betting decisions...", total=len(processable_events)
            )

            tasks = []
            for event_ticker, data in processable_events:
                tasks.append(
                    self._get_betting_decisions_for_event(
                        event_ticker,
                        data,
                        probability_extractions[event_ticker],
                        market_odds,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(
                        f"Exception in betting decisions generation: {result}"
                    )
                    progress.update(task, advance=1)
                    continue

                event_ticker, event_analysis = result
                if event_analysis is not None:
                    self._display_event_decisions(event_ticker, event_analysis)

                    all_decisions.extend(event_analysis.decisions)
                    total_recommended_bet += event_analysis.total_recommended_bet
                    high_confidence_bets += event_analysis.high_confidence_bets
                    event_summaries.append(
                        f"{event_ticker}: {event_analysis.summary}"
                    )

                    self.console.print(
                        f"[green]✓ Generated {len(event_analysis.decisions)} "
                        f"decisions for {event_ticker}[/green]"
                    )
                else:
                    self.console.print(
                        f"[red]✗ Failed to generate decisions for "
                        f"{event_ticker}[/red]"
                    )

                progress.update(task, advance=1)

        hedge_decisions = self._generate_hedge_decisions(all_decisions)
        if hedge_decisions:
            all_decisions.extend(hedge_decisions)
            hedge_bet_total = sum(d.amount for d in hedge_decisions)
            total_recommended_bet += hedge_bet_total
            self.console.print(
                f"[blue]💡 Generated {len(hedge_decisions)} hedge bets "
                f"(${hedge_bet_total:.2f}) for risk management[/blue]"
            )

        analysis = MarketAnalysis(
            decisions=all_decisions,
            total_recommended_bet=float(total_recommended_bet),
            high_confidence_bets=high_confidence_bets,
            summary=(
                f"Analyzed {len(processable_events)} events. "
                + " | ".join(event_summaries[:3])
                + (
                    f" and {len(event_summaries) - 3} more..."
                    if len(event_summaries) > 3
                    else ""
                )
            ),
        )

        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        self.console.print(
            f"\n[green]✓ Generated {len(analysis.decisions)} total decisions "
            f"({len(actionable_decisions)} actionable)[/green]"
        )

        if actionable_decisions:
            table = Table(title="📊 All Betting Decisions Summary", show_lines=True)
            table.add_column("Type", style="bright_blue", justify="center", width=8)
            table.add_column("Event", style="bright_blue", width=22)
            table.add_column("Market", style="cyan", width=32)
            table.add_column("Action", style="yellow", justify="center", width=10)
            table.add_column("Confidence", style="magenta", justify="right", width=10)
            table.add_column("Amount", style="green", justify="right", width=10)
            table.add_column("Reasoning", style="blue", width=65)

            for decision in actionable_decisions:
                event_name = decision.event_name or "Unknown Event"
                market_name = decision.market_name or decision.ticker
                bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"

                table.add_row(
                    bet_type,
                    event_name,
                    market_name,
                    decision.action.upper().replace("_", " "),
                    f"{decision.confidence:.2f}",
                    f"${decision.amount:.2f}",
                    decision.reasoning,
                )

            self.console.print(table)
        else:
            self.console.print(
                "[yellow]No actionable betting decisions generated[/yellow]"
            )

        self.console.print(
            f"\n[blue]Total recommended bet: ${analysis.total_recommended_bet:.2f}[/blue]"
        )
        self.console.print(
            f"[blue]High confidence bets: {analysis.high_confidence_bets}[/blue]"
        )
        self.console.print(f"[blue]Strategy: {analysis.summary}[/blue]")

        return analysis

    # ---------- Execution + CSV logging ----------

    async def place_bets(
        self,
        analysis: MarketAnalysis,
        market_odds: Dict[str, Dict[str, Any]],
        probability_extractions: Dict[str, ProbabilityExtraction],
    ):
        """Place bets based on the analysis."""
        self.console.print(f"\n[bold]Step 6: Placing bets...[/bold]")

        if not analysis.decisions:
            self.console.print("[yellow]No betting decisions to execute[/yellow]")
            return

        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        if not actionable_decisions:
            self.console.print(
                "[yellow]No actionable betting decisions to execute[/yellow]"
            )
            return

        self.console.print(
            f"Found {len(actionable_decisions)} actionable decisions"
        )

        for decision in actionable_decisions:
            if self.config.dry_run:
                self.console.print(
                    f"[blue]DRY RUN: Would place {decision.action} bet of "
                    f"${decision.amount} on {decision.ticker}[/blue]"
                )
            else:
                side = "yes" if decision.action == "buy_yes" else "no"
                result = await self.kalshi_client.place_order(
                    decision.ticker, side, decision.amount
                )

                if result.get("success"):
                    self.console.print(
                        f"[green]✓ Placed {decision.action} bet of "
                        f"${decision.amount} on {decision.ticker}[/green]"
                    )
                else:
                    self.console.print(
                        f"[red]✗ Failed to place bet on {decision.ticker}: "
                        f"{result.get('error', 'Unknown error')}[/red]"
                    )

        if self.config.dry_run:
            self.console.print(
                "\n[yellow]DRY RUN MODE: No actual bets were placed[/yellow]"
            )
        else:
            self.console.print(f"\n[green]✓ Completed bet placement[/green]")

    def save_betting_decisions_to_csv(
        self,
        analysis: MarketAnalysis,
        research_results: Dict[str, str],
        probability_extractions: Dict[str, ProbabilityExtraction],
        market_odds: Dict[str, Dict[str, Any]],
        event_markets: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Save betting decisions to a timestamped CSV file including raw research data.
        """
        output_dir = Path("betting_decisions")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"betting_decisions_{timestamp}.csv"
        filepath = output_dir / filename

        csv_data: List[Dict[str, Any]] = []

        for decision in analysis.decisions:
            event_ticker = None
            raw_research = ""
            research_summary = ""
            research_probability = None
            research_reasoning = ""
            market_yes_price = None
            market_no_price = None
            market_yes_mid = None
            market_no_mid = None
            event_title = ""
            market_title = ""

            for evt_ticker, data in event_markets.items():
                for market in data["markets"]:
                    if market.get("ticker") == decision.ticker:
                        event_ticker = evt_ticker
                        event_title = data["event"].get("title", "")
                        market_title = market.get("title", "")
                        break
                if event_ticker:
                    break

            if event_ticker and event_ticker in research_results:
                raw_research = research_results[event_ticker]

            if event_ticker and event_ticker in probability_extractions:
                extraction = probability_extractions[event_ticker]
                research_summary = extraction.overall_summary
                for market_prob in extraction.markets:
                    if market_prob.ticker == decision.ticker:
                        research_probability = market_prob.research_probability
                        research_reasoning = market_prob.reasoning
                        break

            if decision.ticker in market_odds:
                odds = market_odds[decision.ticker]
                yes_bid = odds.get("yes_bid", 0)
                no_bid = odds.get("no_bid", 0)
                yes_ask = odds.get("yes_ask", 0)
                no_ask = odds.get("no_ask", 0)

                has_yes_side = (yes_bid > 0) or (yes_ask > 0)
                has_no_side = (no_bid > 0) or (no_ask > 0)
                if not (has_yes_side and has_no_side):
                    continue

                market_yes_price = yes_ask if yes_ask > 0 else None
                market_no_price = no_ask if no_ask > 0 else None

                if yes_bid > 0 and yes_ask > 0:
                    market_yes_mid = (yes_bid + yes_ask) / 2.0
                if no_bid > 0 and no_ask > 0:
                    market_no_mid = (no_bid + no_ask) / 2.0

            if (market_yes_price is None or market_yes_price == 0) or (
                market_no_price is None or market_no_price == 0
            ):
                continue

            csv_row: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "event_ticker": event_ticker or "",
                "event_title": event_title,
                "market_ticker": decision.ticker,
                "market_title": market_title,
                "action": decision.action,
                "bet_amount": decision.amount,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "research_probability": research_probability,
                "research_reasoning": research_reasoning,
                "market_yes_price": market_yes_price,
                "market_no_price": market_no_price,
                "market_yes_mid": market_yes_mid,
                "market_no_mid": market_no_mid,
                "expected_return": getattr(decision, "expected_return", None),
                "r_score": getattr(decision, "r_score", None),
                "kelly_fraction": getattr(decision, "kelly_fraction", None),
                "calc_market_prob": getattr(decision, "market_price", None),
                "calc_research_prob": getattr(decision, "research_probability", None),
                "is_hedge": getattr(decision, "is_hedge", False),
                "hedge_for": getattr(decision, "hedge_for", "") or "",
                "research_summary": research_summary,
                "raw_research": raw_research.replace("\n", " ").replace("\r", " ")
                if raw_research
                else "",
            }

            market_enriched: Dict[str, Any] = {}
            if decision.ticker in market_odds:
                m = market_odds[decision.ticker]
                market_enriched["market_title_full"] = m.get("title")
                market_enriched["market_subtitle"] = m.get("subtitle")
                market_enriched["market_yes_sub_title"] = m.get("yes_sub_title")
                market_enriched["market_no_sub_title"] = m.get("no_sub_title")

                market_enriched.update(
                    {
                        "market_event_ticker": m.get("event_ticker"),
                        "market_market_type": m.get("market_type"),
                        "market_open_time": m.get("open_time"),
                        "market_close_time": m.get("close_time"),
                        "market_expiration_time": m.get("expiration_time"),
                        "market_latest_expiration_time": m.get("latest_expiration_time"),
                        "market_settlement_timer_seconds": m.get(
                            "settlement_timer_seconds"
                        ),
                        "market_status": m.get("status"),
                        "market_response_price_units": m.get("response_price_units"),
                        "market_notional_value": m.get("notional_value"),
                        "market_tick_size": m.get("tick_size"),
                        "market_yes_bid": m.get("yes_bid"),
                        "market_yes_ask": m.get("yes_ask"),
                        "market_no_bid": m.get("no_bid"),
                        "market_no_ask": m.get("no_ask"),
                        "market_last_price": m.get("last_price"),
                        "market_previous_yes_bid": m.get("previous_yes_bid"),
                        "market_previous_yes_ask": m.get("previous_yes_ask"),
                        "market_previous_price": m.get("previous_price"),
                        "market_volume": m.get("volume"),
                        "market_volume_24h": m.get("volume_24h"),
                        "market_liquidity": m.get("liquidity"),
                        "market_open_interest": m.get("open_interest"),
                        "market_result": m.get("result"),
                        "market_can_close_early": m.get("can_close_early"),
                        "market_expiration_value": m.get("expiration_value"),
                        "market_category": m.get("category"),
                        "market_risk_limit_cents": m.get("risk_limit_cents"),
                        "market_rules_primary": m.get("rules_primary"),
                        "market_rules_secondary": m.get("rules_secondary"),
                        "market_settlement_value": m.get("settlement_value"),
                        "market_settlement_value_dollars": m.get(
                            "settlement_value_dollars"
                        ),
                    }
                )

            csv_row.update(market_enriched)
            csv_data.append(csv_row)

        if csv_data:
            fieldnames = [
                "timestamp",
                "event_ticker",
                "event_title",
                "market_ticker",
                "market_title",
                "action",
                "bet_amount",
                "confidence",
                "reasoning",
                "research_probability",
                "research_reasoning",
                "market_yes_price",
                "market_no_price",
                "market_yes_mid",
                "market_no_mid",
                "expected_return",
                "r_score",
                "kelly_fraction",
                "calc_market_prob",
                "calc_research_prob",
                "is_hedge",
                "hedge_for",
                "research_summary",
                "raw_research",
            ]

            additional_market_fields = [
                "market_title_full",
                "market_subtitle",
                "market_yes_sub_title",
                "market_no_sub_title",
                "market_event_ticker",
                "market_market_type",
                "market_open_time",
                "market_close_time",
                "market_expiration_time",
                "market_latest_expiration_time",
                "market_settlement_timer_seconds",
                "market_status",
                "market_response_price_units",
                "market_notional_value",
                "market_tick_size",
                "market_yes_bid",
                "market_yes_ask",
                "market_no_bid",
                "market_no_ask",
                "market_last_price",
                "market_previous_yes_bid",
                "market_previous_yes_ask",
                "market_previous_price",
                "market_volume",
                "market_volume_24h",
                "market_liquidity",
                "market_open_interest",
                "market_result",
                "market_can_close_early",
                "market_expiration_value",
                "market_category",
                "market_risk_limit_cents",
                "market_rules_primary",
                "market_rules_secondary",
                "market_settlement_value",
                "market_settlement_value_dollars",
            ]

            base_set = set(fieldnames)
            present_keys = set()
            for row in csv_data:
                for k in row.keys():
                    if k not in base_set:
                        present_keys.add(k)

            ordered_extras = [f for f in additional_market_fields if f in present_keys]
            remaining_extras = sorted(
                k for k in present_keys if k not in set(ordered_extras)
            )
            fieldnames.extend(ordered_extras + remaining_extras)

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(f"Saved {len(csv_data)} betting decisions to {filepath}")
            self.console.print(
                f"[bold green]✓[/bold green] Betting decisions saved to: [blue]{filepath}[/blue]"
            )
        else:
            logger.warning("No betting decisions to save")
            self.console.print("[yellow]No betting decisions to save[/yellow]")

        return str(filepath)

    # ---------- Main run loop ----------

    async def run(self):
        """Main bot execution."""
        try:
            await self.initialize()

            events = await self.get_top_events()
            if not events:
                self.console.print("[red]No events found. Exiting.[/red]")
                return

            event_markets = await self.get_markets_for_events(events)
            if not event_markets:
                self.console.print("[red]No markets found. Exiting.[/red]")
                return

            event_markets = await self.filter_markets_by_positions(event_markets)
            if not event_markets:
                self.console.print(
                    "[red]No markets remaining after position filtering. Exiting.[/red]"
                )
                return

            if len(event_markets) > self.config.max_events_to_analyze:
                filtered_events_list = []
                for event_ticker, data in event_markets.items():
                    event = data["event"]
                    volume_24h = event.get("volume_24h", 0)
                    filtered_events_list.append((event_ticker, data, volume_24h))

                filtered_events_list.sort(key=lambda x: x[2], reverse=True)
                top_events = filtered_events_list[: self.config.max_events_to_analyze]
                event_markets = {et: d for et, d, _ in top_events}

                self.console.print(
                    f"[blue]• Limited to top {len(event_markets)} events by volume "
                    f"after position filtering[/blue]"
                )

            research_results = await self.research_events(event_markets)
            if not research_results:
                self.console.print("[red]No research results. Exiting.[/red]")
                return

            probability_extractions = await self.extract_probabilities(
                research_results, event_markets
            )
            if not probability_extractions:
                self.console.print(
                    "[red]No probability extractions. Exiting.[/red]"
                )
                return

            market_odds = await self.get_market_odds(event_markets)
            if not market_odds:
                self.console.print("[red]No market odds found. Exiting.[/red]")
                return

            analysis = await self.get_betting_decisions(
                event_markets, probability_extractions, market_odds
            )

            self.save_betting_decisions_to_csv(
                analysis=analysis,
                research_results=research_results,
                probability_extractions=probability_extractions,
                market_odds=market_odds,
                event_markets=event_markets,
            )

            await self.place_bets(analysis, market_odds, probability_extractions)

            self.console.print("\n[bold green]Bot execution completed![/bold green]")
        except Exception as e:
            self.console.print(f"[red]Bot execution error: {e}[/red]")
            logger.exception("Bot execution failed")
        finally:
            if self.research_client:
                await self.research_client.close()
            if self.kalshi_client:
                await self.kalshi_client.close()


async def main(live_trading: bool = False, max_close_ts: Optional[int] = None):
    """Main entry point."""
    bot = SimpleTradingBot(live_trading=live_trading, max_close_ts=max_close_ts)
    await bot.run()


def cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Simple Kalshi trading bot with Octagon research and "
            "OpenAI decision making"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run trading-bot                    # Run bot in dry run mode (default)
  uv run trading-bot --live             # Run bot with live trading enabled
  uv run trading-bot --help             # Show this help message

Configuration:
  Create a .env file with your API keys:
    KALSHI_API_KEY=your_kalshi_api_key
    KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----"
    OCTAGON_API_KEY=your_octagon_api_key
    OPENAI_API_KEY=your_openai_api_key

  Optional settings:
    KALSHI_USE_DEMO=true               # Use demo environment (default: true)
    MAX_EVENTS_TO_ANALYZE=50           # Max events to analyze (default: 50)
    MAX_BET_AMOUNT=2.0                 # Max bet per market (default tightened)
    RESEARCH_BATCH_SIZE=10             # Parallel research requests (default: 10)
    SKIP_EXISTING_POSITIONS=true       # Skip markets with existing positions
    Z_THRESHOLD=2.0                    # Minimum R-score (z-score) for betting
    MIN_CONFIDENCE_TO_BET=0.7          # Minimum confidence for any bet
    MIN_EDGE_POINTS=10.0               # Min edge (percentage points) vs market
    KELLY_FRACTION=0.25                # Fraction of Kelly for sizing
    BANKROLL=500.0                     # Total bankroll for Kelly calculations

  Trading modes:
    Default: Dry run mode - shows what trades would be made without placing real bets
    --live: Live trading mode - actually places bets (use with caution!)
        """,
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry run mode)",
    )
    parser.add_argument(
        "--max-expiration-hours",
        type=int,
        default=None,
        dest="max_expiration_hours",
        help=(
            "Only include markets that close within this many hours from now "
            "(minimum 1 hour)."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Kalshi Trading Bot 1.0.0",
    )

    args = parser.parse_args()

    try:
        max_close_ts = None
        if args.max_expiration_hours is not None:
            hours = max(1, args.max_expiration_hours)
            max_close_ts = int(time.time()) + (hours * 3600)
        asyncio.run(main(live_trading=args.live, max_close_ts=max_close_ts))
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Please check your .env file configuration.[/yellow]")
        console.print("[yellow]Run with --help for more information.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
