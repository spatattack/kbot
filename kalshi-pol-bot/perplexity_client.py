"""
Perplexity API client for real-time web research on Kalshi political markets.
Uses OpenAI compatibility for simplicity.
"""

import logging
from typing import Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Client for Perplexity API using OpenAI compatibility."""
    
    def __init__(self, api_key: str, model: str = "sonar-small-online"):
        """
        Initialize Perplexity client using OpenAI compatibility.
        
        Args:
            api_key: Perplexity API key
            model: Model to use (default: sonar-small-online for real-time web search)
        """
        self.api_key = api_key
        self.model = model
        
        # Use OpenAI client with Perplexity base URL
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.close()
    
    async def fetch_event_research(
        self,
        event_ticker: str,
        event_title: str,
        event_description: str,
        category: str,
        sub_category: str,
    ) -> Optional[str]:
        """
        Fetch real-time research for a political event using Perplexity.
        
        Args:
            event_ticker: Kalshi event ticker
            event_title: Human-readable event title
            event_description: Event description
            category: Event category
            sub_category: Event sub-category
            
        Returns:
            Research text from Perplexity, or None if failed
        """
        try:
            # Craft a focused research query
            prompt = self._build_research_prompt(
                event_title, event_description, category, sub_category
            )
            
            logger.warning(f"PERPLEXITY: Making API call for event_ticker={event_ticker}, model={self.model}")
            logger.warning(f"PERPLEXITY: Event title: {event_title}")
            logger.warning(f"PERPLEXITY: Prompt length: {len(prompt)} chars")
            
            # Call Perplexity using OpenAI-compatible interface
            logger.warning(f"PERPLEXITY: Calling chat.completions.create()...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a political analyst researching prediction market events. "
                            "Provide factual, objective research with the latest information from the web. "
                            "Focus on recent polls, news, expert opinions, and concrete data points. "
                            "Cite specific sources when available."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Low temperature for factual research
                max_tokens=1500,
                # Perplexity-specific parameters
                extra_body={
                    "return_related_questions": False,
                    "return_images": False,
                    "search_recency_filter": "week",  # Prioritize recent content
                }
            )
            
            # Extract the research text
            research_text = response.choices[0].message.content
            
            logger.warning(f"PERPLEXITY: API call completed successfully")
            logger.warning(f"PERPLEXITY: Response length: {len(research_text) if research_text else 0} chars")
            
            if research_text:
                logger.info(f"Perplexity research completed for {event_ticker} ({len(research_text)} chars)")
                
                # Log sources if available
                if hasattr(response, 'search_results') and response.search_results:
                    logger.info(f"Perplexity sources: {len(response.search_results)} articles")
                
                logger.warning(f"PERPLEXITY: Returning {len(research_text)} chars of research for {event_ticker}")
                return research_text
            else:
                logger.warning(f"Perplexity returned empty content for {event_ticker}")
                logger.warning(f"PERPLEXITY: Response object: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Perplexity API error for {event_ticker}: {e}")
            logger.warning(f"PERPLEXITY ERROR: {type(e).__name__}: {e}")
            import traceback
            logger.warning(f"PERPLEXITY TRACEBACK: {traceback.format_exc()}")
            return None
    
    def _build_research_prompt(
        self,
        event_title: str,
        event_description: str,
        category: str,
        sub_category: str,
    ) -> str:
        """Build a focused research prompt for Perplexity."""
        title = event_title.strip()
        description = event_description.strip() if event_description else ""
        
        prompt = f"""Research this political prediction market event using the latest web information:

EVENT: {title}

CATEGORY: {category} / {sub_category}

{f"DESCRIPTION: {description}" if description else ""}

Please provide:
1. LATEST NEWS: Most recent relevant news, developments, or announcements (within last 7 days if available)
2. POLLING DATA: Latest polls, surveys, or expert predictions with specific numbers
3. EXPERT OPINIONS: What political analysts, journalists, or insiders are saying
4. HISTORICAL CONTEXT: Relevant precedents or trends
5. KEY FACTORS: Critical variables that will likely determine the outcome

Focus on:
- Concrete, verifiable facts with sources
- Specific numbers, percentages, and dates
- Recent developments (prioritize last 1-7 days)
- Signal over noise - only highly relevant information

Format: Provide clear, structured analysis in paragraph form. Start with the most time-sensitive information."""

        return prompt
    
    async def fetch_market_research(
        self,
        market_ticker: str,
        market_title: str,
        event_context: str,
    ) -> Optional[str]:
        """
        Fetch research for a specific market within an event.
        
        Args:
            market_ticker: Market ticker
            market_title: Market title/question
            event_context: Context from the parent event
            
        Returns:
            Research text, or None if failed
        """
        try:
            prompt = f"""Research this specific prediction market question using latest web information:

MARKET QUESTION: {market_title}

EVENT CONTEXT: {event_context}

Please provide:
1. Latest relevant news or developments
2. Specific data points (polls, odds, expert predictions)
3. Key factors affecting this outcome
4. Recent trends or momentum

Focus on factual, recent information with sources."""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a political analyst. Provide factual research with latest web data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=800,
                extra_body={
                    "return_related_questions": False,
                    "return_images": False,
                    "search_recency_filter": "week",
                }
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Perplexity market research error for {market_ticker}: {e}")
            return None
"""
Perplexity API client for real-time web research on Kalshi political markets.
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Client for Perplexity API to get real-time web research."""
    
    def __init__(self, api_key: str, model: str = "sonar-small-online"):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key
            model: Model to use (default: sonar-small-online for real-time web search)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def fetch_event_research(
        self,
        event_ticker: str,
        event_title: str,
        event_description: str,
        category: str,
        sub_category: str,
    ) -> Optional[str]:
        """
        Fetch real-time research for a political event using Perplexity.
        
        Args:
            event_ticker: Kalshi event ticker
            event_title: Human-readable event title
            event_description: Event description
            category: Event category
            sub_category: Event sub-category
            
        Returns:
            Research text from Perplexity, or None if failed
        """
        try:
            # Craft a focused research query
            prompt = self._build_research_prompt(
                event_title, event_description, category, sub_category
            )
            
            # Call Perplexity API
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a political analyst researching prediction market events. "
                                "Provide factual, objective research with the latest information from the web. "
                                "Focus on recent polls, news, expert opinions, and concrete data points. "
                                "Cite specific sources when available."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,  # Low temperature for factual research
                    "max_tokens": 1500,
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract the research text
            if "choices" in data and len(data["choices"]) > 0:
                research_text = data["choices"][0]["message"]["content"]
                logger.info(f"Perplexity research completed for {event_ticker} ({len(research_text)} chars)")
                return research_text
            else:
                logger.warning(f"Perplexity returned no choices for {event_ticker}")
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Perplexity API HTTP error for {event_ticker}: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Perplexity API timeout for {event_ticker}")
            return None
        except Exception as e:
            logger.error(f"Perplexity API error for {event_ticker}: {e}")
            return None
    
    def _build_research_prompt(
        self,
        event_title: str,
        event_description: str,
        category: str,
        sub_category: str,
    ) -> str:
        """
        Build a focused research prompt for Perplexity.
        
        The prompt should extract real-time information that helps predict the outcome.
        """
        # Clean up the inputs
        title = event_title.strip()
        description = event_description.strip() if event_description else ""
        
        prompt = f"""Research this political prediction market event using the latest web information:

EVENT: {title}

CATEGORY: {category} / {sub_category}

{f"DESCRIPTION: {description}" if description else ""}

Please provide:
1. LATEST NEWS: Most recent relevant news, developments, or announcements (within last 7 days if available)
2. POLLING DATA: Latest polls, surveys, or expert predictions with specific numbers
3. EXPERT OPINIONS: What political analysts, journalists, or insiders are saying
4. HISTORICAL CONTEXT: Relevant precedents or trends
5. KEY FACTORS: Critical variables that will likely determine the outcome

Focus on:
- Concrete, verifiable facts with sources
- Specific numbers, percentages, and dates
- Recent developments (prioritize last 1-7 days)
- Signal over noise - only highly relevant information

Format: Provide clear, structured analysis in paragraph form. Start with the most time-sensitive information."""

        return prompt
    
    async def fetch_market_research(
        self,
        market_ticker: str,
        market_title: str,
        event_context: str,
    ) -> Optional[str]:
        """
        Fetch research for a specific market within an event.
        
        This is more granular than event research - useful for markets with specific questions.
        
        Args:
            market_ticker: Market ticker
            market_title: Market title/question
            event_context: Context from the parent event
            
        Returns:
            Research text, or None if failed
        """
        try:
            prompt = f"""Research this specific prediction market question using latest web information:

MARKET QUESTION: {market_title}

EVENT CONTEXT: {event_context}

Please provide:
1. Latest relevant news or developments
2. Specific data points (polls, odds, expert predictions)
3. Key factors affecting this outcome
4. Recent trends or momentum

Focus on factual, recent information with sources."""

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a political analyst. Provide factual research with latest web data."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Perplexity market research error for {market_ticker}: {e}")
            return None