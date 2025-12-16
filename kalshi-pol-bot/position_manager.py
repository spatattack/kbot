"""
Position management for Kalshi events markets - Institutional Grade.
Continuous evaluation, smart exits, only sell when edge is truly gone.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    ticker: str
    reason: str
    urgency: str  # "high", "medium", "low"
    current_edge: float  # Current edge in percentage points
    current_pnl_pct: float  # Current P&L percentage
    recommendation: str  # "exit_now", "exit_when_convenient", "hold"
    thesis_check: str  # "thesis_intact", "thesis_broken", "thesis_uncertain"


class PositionManager:
    """
    Manage position lifecycle with institutional-grade logic.
    
    Key principles:
    1. Only exit when thesis is clearly broken
    2. If probability moving in our favor, hold
    3. Continuous re-evaluation of all positions
    4. Exit losers quickly, let winners run
    """
    
    def __init__(
        self,
        kalshi_client,
        research_client,
        openai_client,
        config,
        # Exit thresholds
        edge_disappeared_threshold: float = 3.0,  # Exit if edge < 3pp
        adverse_move_threshold: float = 20.0,     # Exit if moved 20pp against us
        time_to_close_hours: float = 6.0,         # Re-evaluate 6h before expiry
        major_loss_threshold: float = 0.70,       # Exit if down >70% (clearly wrong)
        minor_loss_threshold: float = 0.30,       # Monitor if down >30%
    ):
        self.kalshi_client = kalshi_client
        self.research_client = research_client
        self.openai_client = openai_client
        self.config = config
        
        self.edge_disappeared_threshold = edge_disappeared_threshold
        self.adverse_move_threshold = adverse_move_threshold
        self.time_to_close_hours = time_to_close_hours
        self.major_loss_threshold = major_loss_threshold
        self.minor_loss_threshold = minor_loss_threshold
        
        # Cache for re-research results
        self._research_cache: Dict[str, Tuple[float, datetime, str]] = {}
        
    async def evaluate_all_positions(self) -> List[ExitSignal]:
        """
        Continuously evaluate all positions for exit signals.
        
        Returns:
            List of exit signals (may be empty if all should hold)
        """
        try:
            positions = await self.kalshi_client.get_user_positions()
            
            if not positions:
                logger.info("No positions to evaluate")
                return []
            
            logger.info(f"Evaluating {len(positions)} positions")
            
            exit_signals = []
            hold_signals = []
            
            for position in positions:
                contracts = abs(position.get('position', 0))
                if contracts == 0:
                    continue
                
                signal = await self._evaluate_single_position(position)
                
                if signal:
                    if signal.recommendation in ["exit_now", "exit_when_convenient"]:
                        exit_signals.append(signal)
                    elif signal.recommendation == "hold":
                        hold_signals.append(signal)
            
            logger.info(
                f"Position evaluation: {len(exit_signals)} exits, "
                f"{len(hold_signals)} holds"
            )
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"Error evaluating positions: {e}")
            return []
    
    async def _evaluate_single_position(self, position: Dict) -> Optional[ExitSignal]:
        """
        Evaluate a single position with institutional rigor.
        
        Decision tree:
        1. Check for major loss (>70%) -> EXIT NOW
        2. Check if thesis still intact -> If broken, EXIT
        3. Check if edge disappeared -> If gone, EXIT
        4. Check if approaching expiry -> Re-evaluate
        5. Check if adverse movement -> Monitor closely
        6. Otherwise -> HOLD
        """
        try:
            ticker = position['ticker']
            event_ticker = position.get('event_ticker', '')
            contracts = abs(position.get('position', 0))
            is_yes = position.get('position', 0) > 0
            avg_entry_price = position.get('avg_entry_price', 50)  # Cents
            
            # Get current market
            market = self.kalshi_client.get_market_with_odds(ticker)
            
            if not market:
                logger.warning(f"Could not get market data for {ticker}")
                return None
            
            # Current prices
            current_market_price = market.get('yes_ask', 50) if is_yes else market.get('no_ask', 50)
            current_market_prob = current_market_price / 100.0
            
            # Calculate P&L
            current_value_price = market.get('yes_bid', avg_entry_price) if is_yes else market.get('no_bid', avg_entry_price)
            cost = avg_entry_price
            pnl_cents = current_value_price - cost
            pnl_pct = pnl_cents / cost if cost > 0 else 0
            
            # Time to expiration
            time_to_expiry_hours = self._get_time_to_expiry(market)
            
            # === EXIT CONDITION 1: Major Loss (Thesis Clearly Broken) ===
            if pnl_pct < -self.major_loss_threshold:
                return ExitSignal(
                    ticker=ticker,
                    reason=f"Major loss: {pnl_pct*100:.1f}% - thesis clearly wrong",
                    urgency="high",
                    current_edge=0.0,
                    current_pnl_pct=pnl_pct,
                    recommendation="exit_now",
                    thesis_check="thesis_broken",
                )
            
            # === CONDITION 2: Re-research Position ===
            # Get updated probability and thesis check
            research_prob, thesis_status, research_explanation = await self._reevaluate_thesis(
                ticker, event_ticker, market, is_yes, avg_entry_price
            )
            
            if research_prob is None:
                # Can't re-research, use conservative logic
                logger.warning(f"Could not re-research {ticker}")
                
                # If losing badly and close to expiry, exit
                if pnl_pct < -0.30 and time_to_expiry_hours and time_to_expiry_hours < self.time_to_close_hours:
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Down {pnl_pct*100:.1f}% near expiry, cannot re-research",
                        urgency="high",
                        current_edge=0.0,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_now",
                        thesis_check="thesis_uncertain",
                    )
                
                # Otherwise hold conservatively
                return ExitSignal(
                    ticker=ticker,
                    reason="Cannot re-research, monitoring",
                    urgency="low",
                    current_edge=0.0,
                    current_pnl_pct=pnl_pct,
                    recommendation="hold",
                    thesis_check="thesis_uncertain",
                )
            
            # === CONDITION 3: Thesis Check ===
            if thesis_status == "thesis_broken":
                return ExitSignal(
                    ticker=ticker,
                    reason=f"Thesis broken: {research_explanation}",
                    urgency="high",
                    current_edge=0.0,
                    current_pnl_pct=pnl_pct,
                    recommendation="exit_now",
                    thesis_check=thesis_status,
                )
            
            # === CONDITION 4: Edge Calculation ===
            # Calculate current edge
            if is_yes:
                # We own YES, compare research_prob vs market
                edge = research_prob - current_market_prob
            else:
                # We own NO, compare (1-research_prob) vs (1-market)
                edge = (1 - research_prob) - (1 - current_market_prob)
            
            edge_pct = edge * 100
            
            # If edge disappeared, consider exiting
            if edge_pct < self.edge_disappeared_threshold:
                # But check if we're profitable
                if pnl_pct > 0.10:  # If up >10%, take profit
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Edge gone ({edge_pct:.1f}pp), take profit (+{pnl_pct*100:.1f}%)",
                        urgency="medium",
                        current_edge=edge_pct,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_when_convenient",
                        thesis_check=thesis_status,
                    )
                elif pnl_pct < -self.minor_loss_threshold:  # If losing, exit
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Edge gone ({edge_pct:.1f}pp), losing {pnl_pct*100:.1f}%",
                        urgency="high",
                        current_edge=edge_pct,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_now",
                        thesis_check=thesis_status,
                    )
                else:
                    # Small loss, edge gone, but not urgent
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Edge diminished ({edge_pct:.1f}pp)",
                        urgency="low",
                        current_edge=edge_pct,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_when_convenient",
                        thesis_check=thesis_status,
                    )
            
            # === CONDITION 5: Approaching Expiry ===
            if time_to_expiry_hours and time_to_expiry_hours < self.time_to_close_hours:
                # Close to expiry, take action based on position
                if pnl_pct < -0.20:  # Losing, exit
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Near expiry ({time_to_expiry_hours:.1f}h), losing {pnl_pct*100:.1f}%",
                        urgency="high",
                        current_edge=edge_pct,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_now",
                        thesis_check=thesis_status,
                    )
                elif pnl_pct > 0.15:  # Winning well, take profit
                    return ExitSignal(
                        ticker=ticker,
                        reason=f"Near expiry ({time_to_expiry_hours:.1f}h), take profit (+{pnl_pct*100:.1f}%)",
                        urgency="medium",
                        current_edge=edge_pct,
                        current_pnl_pct=pnl_pct,
                        recommendation="exit_when_convenient",
                        thesis_check=thesis_status,
                    )
                # Otherwise, let it ride to expiry if thesis intact
            
            # === CONDITION 6: Adverse Movement ===
            entry_prob = avg_entry_price / 100.0
            movement = (current_market_prob - entry_prob) * 100
            
            if is_yes:
                # For YES position, negative movement is bad
                if movement < -self.adverse_move_threshold:
                    if pnl_pct < -0.40:  # Big loss
                        return ExitSignal(
                            ticker=ticker,
                            reason=f"Adverse move {movement:.1f}pp, down {pnl_pct*100:.1f}%",
                            urgency="high",
                            current_edge=edge_pct,
                            current_pnl_pct=pnl_pct,
                            recommendation="exit_now",
                            thesis_check=thesis_status,
                        )
            else:
                # For NO position, positive movement is bad
                if movement > self.adverse_move_threshold:
                    if pnl_pct < -0.40:  # Big loss
                        return ExitSignal(
                            ticker=ticker,
                            reason=f"Adverse move +{movement:.1f}pp, down {pnl_pct*100:.1f}%",
                            urgency="high",
                            current_edge=edge_pct,
                            current_pnl_pct=pnl_pct,
                            recommendation="exit_now",
                            thesis_check=thesis_status,
                        )
            
            # === DEFAULT: HOLD ===
            # Edge still exists, thesis intact, not at extremes
            return ExitSignal(
                ticker=ticker,
                reason=f"Thesis intact, edge {edge_pct:.1f}pp, P&L {pnl_pct*100:.+.1f}%",
                urgency="low",
                current_edge=edge_pct,
                current_pnl_pct=pnl_pct,
                recommendation="hold",
                thesis_check=thesis_status,
            )
            
        except Exception as e:
            logger.error(f"Error evaluating position {position.get('ticker')}: {e}")
            return None
    
    async def _reevaluate_thesis(
        self,
        ticker: str,
        event_ticker: str,
        market: Dict,
        is_yes: bool,
        entry_price: float,
    ) -> Tuple[Optional[float], str, str]:
        """
        Re-evaluate the thesis for a position.
        
        Returns:
            (research_prob, thesis_status, explanation)
            thesis_status: "thesis_intact", "thesis_broken", "thesis_uncertain"
        """
        # Check cache first
        cache_key = ticker
        if cache_key in self._research_cache:
            cached_prob, cached_time, cached_status = self._research_cache[cache_key]
            age = datetime.now(timezone.utc) - cached_time
            if age.total_seconds() < 1800:  # 30 min cache
                return cached_prob, cached_status, "Cached"
        
        try:
            # Get event and market info
            event = self.kalshi_client.get_event(event_ticker)
            if not event:
                return None, "thesis_uncertain", "Could not fetch event"
            
            markets = [market]
            
            # Re-research
            research_text = await self.research_client.research_event(event, markets)
            
            if not research_text or len(research_text) < 50:
                return None, "thesis_uncertain", "Insufficient research"
            
            # Extract new probability
            import json
            prompt = f"""Re-evaluate this position we're holding.

Market: {market.get('title', '')}
Our Position: {"YES" if is_yes else "NO"} at {entry_price}¢
Current Market: {market.get('yes_ask', 50)}¢ / {market.get('yes_bid', 50)}¢

Latest Research:
{research_text[:800]}

Questions:
1. What is current fair probability for YES? (0-100)
2. Is our original thesis still valid?
3. What changed since we entered?

Return JSON:
{{
  "probability": <0-100>,
  "thesis_status": "intact" | "broken" | "uncertain",
  "explanation": "brief 1-2 sentence explanation"
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You re-evaluate trading theses critically."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            prob = float(result.get('probability', 50)) / 100.0
            prob = max(0.01, min(0.99, prob))
            
            status_map = {
                "intact": "thesis_intact",
                "broken": "thesis_broken",
                "uncertain": "thesis_uncertain",
            }
            thesis_status = status_map.get(result.get('thesis_status', 'uncertain'), "thesis_uncertain")
            
            explanation = result.get('explanation', 'No explanation')
            
            # Cache result
            self._research_cache[cache_key] = (prob, datetime.now(timezone.utc), thesis_status)
            
            logger.info(
                f"Re-evaluated {ticker}: prob={prob*100:.1f}%, "
                f"thesis={thesis_status}, {explanation}"
            )
            
            return prob, thesis_status, explanation
            
        except Exception as e:
            logger.error(f"Error re-evaluating thesis for {ticker}: {e}")
            return None, "thesis_uncertain", f"Error: {e}"
    
    def _get_time_to_expiry(self, market: Dict) -> Optional[float]:
        """Get time to expiry in hours."""
        try:
            close_time_str = market.get('close_time')
            if not close_time_str:
                return None
            
            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            time_to_expiry = close_time - now
            hours = time_to_expiry.total_seconds() / 3600
            
            return hours
            
        except Exception:
            return None
    
    async def exit_position(self, ticker: str) -> bool:
        """
        Exit a position by selling all contracts.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current position
            positions = await self.kalshi_client.get_user_positions()
            position = next((p for p in positions if p.get('ticker') == ticker), None)
            
            if not position:
                logger.warning(f"No position found for {ticker}")
                return False
            
            contracts = abs(position.get('position', 0))
            if contracts == 0:
                logger.warning(f"Position for {ticker} is already 0")
                return False
            
            is_yes = position.get('position', 0) > 0
            
            # Get current market for limit price
            market = self.kalshi_client.get_market_with_odds(ticker)
            if not market:
                logger.error(f"Could not get market for {ticker}")
                return False
            
            # Sell at bid (immediate execution)
            side = "yes" if is_yes else "no"
            limit_price = market.get(f'{side}_bid', 50)
            
            # Place sell order
            result = self.kalshi_client.place_order(
                ticker=ticker,
                action="sell",
                side=side,
                count=contracts,
                order_type="limit",
                limit_price=limit_price,
            )
            
            if result.get('status') == 'success':
                logger.info(f"Successfully exited {ticker}: {contracts} contracts @ {limit_price}¢")
                return True
            else:
                logger.error(f"Failed to exit {ticker}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error exiting position {ticker}: {e}")
            return False


class CorrelationAnalyzer:
    """Analyze correlations between positions."""
    
    def __init__(self):
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
    
    def calculate_correlation(
        self,
        market1: Dict,
        market2: Dict,
    ) -> float:
        """
        Estimate correlation between two markets.
        Returns -1 to 1.
        """
        # Simple heuristic: same event = high correlation
        if market1.get('event_ticker') == market2.get('event_ticker'):
            return 0.8
        
        # Same category = medium correlation
        if market1.get('category') == market2.get('category'):
            return 0.4
        
        # Otherwise assume low correlation
        return 0.1


class AdverseSelectionFilter:
    """Filter for adverse selection (when market knows something we don't)."""
    
    def check_adverse_selection(
        self,
        market: Dict,
        research_prob: float,
        volume_24h: float,
    ) -> Tuple[bool, str]:
        """
        Check if we might be falling for adverse selection.
        
        Returns:
            (is_adverse, reason)
        """
        market_prob = market.get('yes_ask', 50) / 100.0
        edge_pct = abs(research_prob - market_prob) * 100
        
        # Very high volume + large edge = suspicious
        if volume_24h > 200_000 and edge_pct > 20:
            return True, "High volume market with large edge - suspicious"
        
        # Very tight spread + large edge = market is confident
        spread = market.get('yes_ask', 50) - market.get('yes_bid', 50)
        if spread <= 1 and edge_pct > 15:
            return True, "Tight spread suggests market confidence"
        
        return False, ""