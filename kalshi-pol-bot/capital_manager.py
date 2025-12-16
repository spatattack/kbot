"""
Capital management for Kalshi trading bot.
Handles account balance, position sizing, and capital allocation limits.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CapitalState:
    """Current capital state."""
    total_balance: float
    available_cash: float
    deployed_capital: float
    unrealized_pnl: float
    positions_count: int
    events_exposed: Dict[str, float]  # event_ticker -> capital deployed


class CapitalManager:
    """Manage capital allocation and position limits."""
    
    def __init__(
        self,
        kalshi_client,
        max_position_pct: float = 0.10,    # Max 10% per position
        max_event_pct: float = 0.20,       # Max 20% per event
        max_total_deployed_pct: float = 0.80,  # Max 80% deployed
        reserve_pct: float = 0.20,         # Keep 20% in reserve
    ):
        self.kalshi_client = kalshi_client
        self.max_position_pct = max_position_pct
        self.max_event_pct = max_event_pct
        self.max_total_deployed_pct = max_total_deployed_pct
        self.reserve_pct = reserve_pct
        
        self._capital_state: Optional[CapitalState] = None
        
    async def refresh_capital_state(self) -> CapitalState:
        """Refresh capital state from Kalshi API."""
        try:
            # Get account balance
            balance_data = await self.kalshi_client.get_balance()
            total_balance = float(balance_data.get('balance', 0)) / 100  # Convert cents to dollars
            
            # Get current positions
            positions = await self.kalshi_client.get_user_positions()
            
            # Calculate deployed capital and P&L
            deployed_capital = 0
            unrealized_pnl = 0
            events_exposed: Dict[str, float] = {}
            
            for position in positions:
                contracts = abs(position.get('position', 0))
                if contracts == 0:
                    continue
                
                # Cost basis (what we paid)
                avg_price = position.get('avg_entry_price', 50)  # Cents per contract
                cost = (contracts * avg_price) / 100  # Convert to dollars
                deployed_capital += cost
                
                # Get current market value
                try:
                    ticker = position['ticker']
                    market = await self.kalshi_client.get_market_with_odds(ticker)
                    
                    # Current value (what we could sell for)
                    if position['position'] > 0:  # Long YES
                        current_price = market.get('yes_bid', avg_price)
                    else:  # Long NO
                        current_price = market.get('no_bid', avg_price)
                    
                    current_value = (contracts * current_price) / 100
                    unrealized_pnl += (current_value - cost)
                    
                    # Track exposure by event
                    event_ticker = position.get('event_ticker', 'UNKNOWN')
                    events_exposed[event_ticker] = events_exposed.get(event_ticker, 0) + cost
                    
                except Exception as e:
                    logger.warning(f"Could not get market value for {position.get('ticker')}: {e}")
            
            # Available cash = balance - deployed - reserve
            reserve = total_balance * self.reserve_pct
            available_cash = total_balance - deployed_capital - reserve
            available_cash = max(0, available_cash)  # Can't be negative
            
            self._capital_state = CapitalState(
                total_balance=total_balance,
                available_cash=available_cash,
                deployed_capital=deployed_capital,
                unrealized_pnl=unrealized_pnl,
                positions_count=len([p for p in positions if p.get('position', 0) != 0]),
                events_exposed=events_exposed,
            )
            
            logger.info(
                f"Capital: ${total_balance:.2f} total, ${available_cash:.2f} available, "
                f"${deployed_capital:.2f} deployed ({self.positions_count} positions), "
                f"P&L: ${unrealized_pnl:.2f}"
            )
            
            return self._capital_state
            
        except Exception as e:
            logger.error(f"Error refreshing capital state: {e}")
            # Return conservative estimate if we can't get real data
            return CapitalState(
                total_balance=100.0,  # Conservative default
                available_cash=20.0,
                deployed_capital=0.0,
                unrealized_pnl=0.0,
                positions_count=0,
                events_exposed={},
            )
    
    @property
    def capital_state(self) -> Optional[CapitalState]:
        """Get current capital state (cached)."""
        return self._capital_state
    
    def calculate_safe_position_size(
        self,
        kelly_size: float,
        event_ticker: str,
        ticker: str,
    ) -> float:
        """
        Calculate safe position size within capital limits.
        
        Args:
            kelly_size: Suggested Kelly size in dollars
            event_ticker: Event ticker for event-level limits
            ticker: Market ticker for logging
            
        Returns:
            Safe position size in dollars
        """
        if not self._capital_state:
            logger.warning("Capital state not initialized, using conservative sizing")
            return min(kelly_size, 10.0)  # Conservative default
        
        # Start with Kelly size
        size = kelly_size
        original_size = size
        
        # Limit 1: Available cash
        if size > self._capital_state.available_cash:
            logger.info(
                f"{ticker}: Limiting size from ${size:.2f} to ${self._capital_state.available_cash:.2f} "
                f"(available cash)"
            )
            size = self._capital_state.available_cash
        
        # Limit 2: Max % of total balance per position
        max_by_position = self._capital_state.total_balance * self.max_position_pct
        if size > max_by_position:
            logger.info(
                f"{ticker}: Limiting size from ${size:.2f} to ${max_by_position:.2f} "
                f"({self.max_position_pct*100:.0f}% position limit)"
            )
            size = max_by_position
        
        # Limit 3: Max % of total balance per event
        current_event_exposure = self._capital_state.events_exposed.get(event_ticker, 0)
        max_event_capital = self._capital_state.total_balance * self.max_event_pct
        remaining_event_capacity = max_event_capital - current_event_exposure
        
        if size > remaining_event_capacity:
            logger.info(
                f"{ticker}: Limiting size from ${size:.2f} to ${remaining_event_capacity:.2f} "
                f"({self.max_event_pct*100:.0f}% event limit, ${current_event_exposure:.2f} already deployed)"
            )
            size = remaining_event_capacity
        
        # Limit 4: Max total deployed
        max_deployed = self._capital_state.total_balance * self.max_total_deployed_pct
        remaining_capacity = max_deployed - self._capital_state.deployed_capital
        
        if size > remaining_capacity:
            logger.info(
                f"{ticker}: Limiting size from ${size:.2f} to ${remaining_capacity:.2f} "
                f"(max {self.max_total_deployed_pct*100:.0f}% deployed limit)"
            )
            size = remaining_capacity
        
        # Ensure positive
        size = max(0, size)
        
        # Log if significantly reduced
        if size < original_size * 0.5:
            logger.warning(
                f"{ticker}: Position size reduced by {(1 - size/original_size)*100:.0f}% "
                f"due to capital limits (${original_size:.2f} -> ${size:.2f})"
            )
        
        return size
    
    def check_drawdown_limit(self, max_drawdown_pct: float = 0.20) -> bool:
        """
        Check if current drawdown exceeds limit.
        
        Args:
            max_drawdown_pct: Maximum allowed drawdown (default 20%)
            
        Returns:
            True if within limits, False if exceeded
        """
        if not self._capital_state:
            return True  # Unknown state, allow trading
        
        # Calculate current drawdown
        current_value = self._capital_state.deployed_capital + self._capital_state.unrealized_pnl
        peak_value = self._capital_state.deployed_capital  # Initial deployment was the "peak"
        
        if peak_value <= 0:
            return True  # No positions, no drawdown
        
        drawdown = (peak_value - current_value) / peak_value
        
        if drawdown > max_drawdown_pct:
            logger.error(
                f"DRAWDOWN LIMIT EXCEEDED: {drawdown*100:.1f}% (limit: {max_drawdown_pct*100:.1f}%)"
            )
            return False
        
        return True
    
    def get_position_summary(self) -> str:
        """Get human-readable position summary."""
        if not self._capital_state:
            return "Capital state not initialized"
        
        deployed_pct = (self._capital_state.deployed_capital / self._capital_state.total_balance * 100) if self._capital_state.total_balance > 0 else 0
        
        summary = f"""
Capital Summary:
  Total Balance: ${self._capital_state.total_balance:.2f}
  Available Cash: ${self._capital_state.available_cash:.2f}
  Deployed Capital: ${self._capital_state.deployed_capital:.2f} ({deployed_pct:.1f}%)
  Unrealized P&L: ${self._capital_state.unrealized_pnl:.2f}
  Active Positions: {self._capital_state.positions_count}
  Events Exposed: {len(self._capital_state.events_exposed)}
"""
        
        if self._capital_state.events_exposed:
            summary += "\nEvent Exposure:\n"
            for event_ticker, capital in sorted(
                self._capital_state.events_exposed.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]:  # Top 5
                pct = (capital / self._capital_state.total_balance * 100) if self._capital_state.total_balance > 0 else 0
                summary += f"  {event_ticker}: ${capital:.2f} ({pct:.1f}%)\n"
        
        return summary