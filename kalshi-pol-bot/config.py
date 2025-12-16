"""
Configuration management for Kalshi trading bot.
Clean version with correct API URLs.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class KalshiConfig(BaseModel):
    """Kalshi API configuration."""
    api_key: str
    private_key: str  # PEM format private key as string
    use_demo: bool = True
    
    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on demo/prod setting."""
        if self.use_demo:
            return "https://demo-api.kalshi.co"
        return "https://api.kalshi.com"  # Production uses .com


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str
    model: str = "gpt-4o"


class OctagonConfig(BaseModel):
    """Octagon research client configuration (OpenAI-backed internal research)."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class PerplexityConfig(BaseModel):
    """Perplexity API configuration for real-time web research."""
    api_key: str
    model: str = "sonar"
    enabled: bool = True


class BotConfig(BaseModel):
    """Main bot configuration."""
    
    # API configurations
    kalshi: KalshiConfig
    openai: OpenAIConfig
    octagon: OctagonConfig
    perplexity: PerplexityConfig
    
    # Trading parameters
    dry_run: bool = True
    max_bet_amount: float = 100.0
    max_events_to_analyze: int = 50
    
    # Research settings
    research_enabled: bool = True
    research_cache_ttl: int = 3600  # 1 hour in seconds


def load_config() -> BotConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        BotConfig instance with all settings
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Kalshi configuration
    kalshi_api_key = os.getenv("KALSHI_API_KEY")
    if not kalshi_api_key:
        raise ValueError("KALSHI_API_KEY environment variable is required")
    
    # Load private key from file or environment variable
    kalshi_private_key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE")
    kalshi_private_key = os.getenv("KALSHI_PRIVATE_KEY")
    
    if kalshi_private_key_file:
        # Load from file (recommended for security)
        try:
            with open(kalshi_private_key_file, 'r') as f:
                kalshi_private_key = f.read()
        except FileNotFoundError:
            raise ValueError(f"KALSHI_PRIVATE_KEY_FILE '{kalshi_private_key_file}' not found")
        except Exception as e:
            raise ValueError(f"Error reading KALSHI_PRIVATE_KEY_FILE: {e}")
    
    if not kalshi_private_key:
        raise ValueError("Either KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_FILE environment variable is required")
    
    kalshi_use_demo = os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"
    
    # OpenAI configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Octagon configuration (uses OpenAI)
    octagon_api_key = os.getenv("OCTAGON_API_KEY", openai_api_key)
    octagon_base_url = os.getenv("OCTAGON_BASE_URL")
    
    # Perplexity configuration
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")
    perplexity_model = os.getenv("PERPLEXITY_MODEL", "sonar")
    perplexity_enabled = os.getenv("PERPLEXITY_ENABLED", "true").lower() == "true" and bool(perplexity_api_key)
    
    # Trading parameters
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    max_bet_amount = float(os.getenv("MAX_BET_AMOUNT", "100.0"))
    max_events_to_analyze = int(os.getenv("MAX_EVENTS_TO_ANALYZE", "50"))
    
    return BotConfig(
        kalshi=KalshiConfig(
            api_key=kalshi_api_key,
            private_key=kalshi_private_key,
            use_demo=kalshi_use_demo,
        ),
        openai=OpenAIConfig(
            api_key=openai_api_key,
            model=openai_model,
        ),
        octagon=OctagonConfig(
            api_key=octagon_api_key,
            base_url=octagon_base_url,
        ),
        perplexity=PerplexityConfig(
            api_key=perplexity_api_key,
            model=perplexity_model,
            enabled=perplexity_enabled,
        ),
        dry_run=dry_run,
        max_bet_amount=max_bet_amount,
        max_events_to_analyze=max_events_to_analyze,
    )