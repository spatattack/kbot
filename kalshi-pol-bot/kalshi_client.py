"""
Kalshi API client - async version with correct authentication.
Based on official Kalshi API documentation.
"""

import base64
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


class KalshiClient:
    """
    Async client for interacting with Kalshi API.
    Supports both demo and production environments.
    """

    def __init__(self, config):
        """
        Initialize Kalshi client.
        
        Args:
            config: KalshiConfig object with API credentials
        """
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.use_demo = config.use_demo
        
        # Load private key
        self.private_key = serialization.load_pem_private_key(
            config.private_key.encode(),
            password=None,
        )
        
        # HTTP client (will be initialized in async context)
        self.client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"Initialized KalshiClient: {self.base_url}")

    async def login(self):
        """Initialize async HTTP client and test connection."""
        self.client = httpx.AsyncClient(timeout=30.0)
        try:
            # Test connection by fetching events
            events = await self.get_events(limit=1)
            logger.info(f"Connected to Kalshi API at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Kalshi API: {e}")
            raise

    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        return str(int(time.time() * 1000))

    def _create_signature(self, timestamp: str, method: str, path: str) -> str:
        """
        Create request signature using RSA-PSS (per Kalshi docs).
        
        Args:
            timestamp: Current time in milliseconds
            method: HTTP method (GET, POST, etc.)
            path: Request path (without query parameters)
            
        Returns:
            Base64 encoded signature
        """
        # Strip query parameters from path before signing
        path_without_query = path.split('?')[0]
        
        # Create message to sign: timestamp + method + path
        message = f"{timestamp}{method}{path_without_query}".encode('utf-8')
        
        # Sign with RSA-PSS (as specified in Kalshi docs)
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Return base64 encoded
        return base64.b64encode(signature).decode('utf-8')

    async def _make_request(
        self, 
        method: str, 
        path: str,
        authenticated: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make async HTTP request to Kalshi API.
        
        Args:
            method: HTTP method
            path: API path
            authenticated: Whether to include auth headers
            **kwargs: Additional arguments for httpx
            
        Returns:
            Response JSON as dict
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Call login() first.")
            
        url = f"{self.base_url}{path}"
        
        # Add authentication headers if needed
        if authenticated:
            timestamp = self._get_timestamp()
            signature = self._create_signature(timestamp, method, path)
            
            headers = kwargs.get('headers', {})
            headers.update({
                'KALSHI-ACCESS-KEY': self.api_key,
                'KALSHI-ACCESS-SIGNATURE': signature,
                'KALSHI-ACCESS-TIMESTAMP': timestamp,
            })
            kwargs['headers'] = headers
        
        # Make request
        try:
            response = await self.client.request(method, url, **kwargs)
            
            # Check for errors
            if response.status_code >= 400:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', error_msg)
                except:
                    pass
                
                logger.error(f"Kalshi API error: {response.status_code} - {error_msg}")
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response
                )
            
            return response.json()
            
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # Public endpoints (no auth required)
    
    async def get_events(self, limit: int = 100, status: str = "open") -> List[Dict]:
        """Get events from Kalshi."""
        path = f"/trade-api/v2/events?limit={limit}&status={status}"
        response = await self._make_request("GET", path, authenticated=False)
        return response.get('events', [])
    
    async def get_markets_for_event(self, event_ticker: str, limit: int = 200) -> List[Dict]:
        """Get markets for a specific event."""
        path = f"/trade-api/v2/markets?event_ticker={event_ticker}&limit={limit}"
        response = await self._make_request("GET", path, authenticated=False)
        return response.get('markets', [])
    
    async def get_market_with_odds(self, ticker: str) -> Dict:
        """Get market details including current odds."""
        path = f"/trade-api/v2/markets/{ticker}"
        market = await self._make_request("GET", path, authenticated=False)
        
        # Get orderbook for odds
        try:
            orderbook_path = f"/trade-api/v2/markets/{ticker}/orderbook?depth=1"
            orderbook = await self._make_request("GET", orderbook_path, authenticated=False)
            
            # Add yes/no bid/ask to market dict
            if 'orderbook' in orderbook:
                ob = orderbook['orderbook']
                market['yes_bid'] = ob.get('yes', [{}])[0].get('price', 50) if ob.get('yes') else 50
                market['yes_ask'] = ob.get('yes', [{}])[0].get('price', 50) if ob.get('yes') else 50
                market['no_bid'] = ob.get('no', [{}])[0].get('price', 50) if ob.get('no') else 50
                market['no_ask'] = ob.get('no', [{}])[0].get('price', 50) if ob.get('no') else 50
        except Exception as e:
            logger.warning(f"Could not get orderbook for {ticker}: {e}")
            # Set defaults
            market['yes_bid'] = 50
            market['yes_ask'] = 50
            market['no_bid'] = 50
            market['no_ask'] = 50
        
        return market
    
    # Authenticated endpoints
    
    async def get_balance(self) -> Dict:
        """Get account balance (requires auth)."""
        path = "/trade-api/v2/portfolio/balance"
        return await self._make_request("GET", path, authenticated=True)
    
    async def get_user_positions(self, limit: int = 1000) -> List[Dict]:
        """Get current positions (requires auth)."""
        path = f"/trade-api/v2/portfolio/positions?limit={limit}"
        response = await self._make_request("GET", path, authenticated=True)
        return response.get('market_positions', [])

    async def has_position_in_market(self, ticker: str) -> bool:
        """Check if user has a position in a market."""
        try:
            positions = await self.get_user_positions()
            for pos in positions:
                if pos.get('ticker') == ticker:
                    return True
            return False
        except Exception:
            return False
    
    async def place_order(
        self,
        ticker: str,
        side: str,
        amount: float,
    ) -> Dict:
        """
        Place an order (simplified interface).
        
        Args:
            ticker: Market ticker
            side: "yes" or "no"
            amount: Amount to bet in dollars
            
        Returns:
            Order result
        """
        # For dry run, just return success
        if self.use_demo:
            logger.info(f"DRY RUN: Would place order {side} ${amount} for {ticker}")
            return {"status": "success", "dry_run": True}
        
        # Convert dollars to cents and contracts
        cents = int(amount * 100)
        
        path = "/trade-api/v2/portfolio/orders"
        
        payload = {
            "ticker": ticker,
            "side": side,
            "count": 1,
            "type": "market",
            "cost_limit": cents,
        }
        
        return await self._make_request(
            "POST",
            path,
            authenticated=True,
            json=payload
        )
    
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
