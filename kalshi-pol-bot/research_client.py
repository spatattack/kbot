import os
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OctagonClient:
    """
    Drop-in replacement for the old Octagon research client.

    - Keeps the same class name so trading_bot.py doesn't need changes.
    - Uses OpenAI directly via OPENAI_API_KEY / OPENAI_MODEL.
    - research_event(...) returns a plain text string, which the bot can
      later parse or do .replace(...) on.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

        # Prefer OPENAI_API_KEY, fall back to OCTAGON_API_KEY if needed
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OCTAGON_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY (or OCTAGON_API_KEY) in your .env."
            )

        # Allow overriding the model via env
        # e.g. OPENAI_MODEL=gpt-5-mini or gpt-4.1-mini
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        self.client = AsyncOpenAI(api_key=api_key)

    async def research_event(self, event: Dict[str, Any], markets: List[Dict[str, Any]]) -> str:
        """
        Run research on an event + markets and return a textual analysis.

        trading_bot.py expects a STRING from this method.
        """

        try:
            event_ticker = event.get("event_ticker", "UNKNOWN")
            title = event.get("title", "")
            subtitle = event.get("subtitle", "")
            category = event.get("category", "")

            markets_summary = []
            for m in markets:
                ticker = m.get("ticker", "")
                m_title = m.get("title", "")
                m_subtitle = m.get("subtitle", "")
                line = f"- {ticker}: {m_title}"
                if m_subtitle:
                    line += f" | {m_subtitle}"
                markets_summary.append(line)

            markets_text = "\n".join(markets_summary)

            prompt = f"""
You are an expert political prediction market analyst.

Event:
  Ticker: {event_ticker}
  Title: {title}
  Subtitle: {subtitle}
  Category: {category}

Markets for this event:
{markets_text}

For each market, do the following:
1. Briefly summarize the key considerations (polls, fundamentals, news, etc.).
2. Estimate the probability that the YES side will resolve as true (0–100%).
3. Give a short recommendation (e.g. 'YES is underpriced', 'NO is underpriced', or 'fairly priced').

Return your answer as clear, readable text, with each market on its own section,
including the market TICKER, estimated probability, and recommendation.
"""

            logger.info(
                f"Researching event {event_ticker} with OpenAI model {self.model}"
            )

            # Minimal, compatible call for current OpenAI chat API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=800,
            )

            content = response.choices[0].message.content or ""
            if not content.strip():
                logger.error(f"Empty content returned for event {event_ticker}")
                return f"Error: empty research content for event {event_ticker}"

            logger.info(f"Completed research for event {event_ticker}")
            return content

        except Exception as e:
            # Catch EVERYTHING to avoid bubbling exceptions into trading_bot
            logger.exception(f"Error researching event {event.get('event_ticker', 'UNKNOWN')}: {e}")
            return f"Error researching event {event.get('event_ticker', 'UNKNOWN')}: {e}"

    async def close(self) -> None:
        """
        Kept for API compatibility; AsyncOpenAI doesn't require explicit close.
        """
        return None

