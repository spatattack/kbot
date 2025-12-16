import asyncio
import json
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import httpx


class OpenAIResearcher:
    """
    Wrapper around OpenAI models that produces structured event probability output.
    """

    def __init__(self, config):
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.model = config.model

    async def research_event(self, event: dict) -> dict:
        """
        Research a single Kalshi event using OpenAI. Produces structured JSON:

        {
            "ticker": "...",
            "probabilities": { "YES": float, "NO": float },
            "confidence": float,
            "rationale": "..."
        }
        """

        system_prompt = (
            "You are a quantitative event researcher. "
            "Given a Kalshi market event, estimate fair-value probabilities "
            "and provide reasoning. Return ONLY valid JSON."
        )

        user_prompt = (
            f"Event:\n{json.dumps(event, indent=2)}\n\n"
            "Return JSON with fields: probabilities, confidence, rationale."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content
            return json.loads(raw)

        except Exception as e:
            return {
                "ticker": event.get("event_ticker"),
                "error": str(e),
                "probabilities": {},
                "confidence": 0.0,
                "rationale": ""
            }


class PerplexityResearcher:
    """
    Wrapper for Perplexity API. Only used if enabled.
    """

    def __init__(self, config):
        self.api_key = config.api_key
        self.model = config.model
        self.enabled = config.enabled

    async def research_event(self, event: dict) -> dict:
        if not self.enabled:
            return {
                "ticker": event.get("event_ticker"),
                "probabilities": {},
                "confidence": 0.0,
                "rationale": "Perplexity disabled"
            }

        url = "https://api.perplexity.ai/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a quant researcher. Estimate fair probability "
                        "distribution for the following Kalshi event and return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(event),
                },
            ],
            "max_tokens": 256,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                return json.loads(raw)

            except Exception as e:
                return {
                    "ticker": event.get("event_ticker"),
                    "error": str(e),
                    "probabilities": {},
                    "confidence": 0.0,
                    "rationale": ""
                }


class ResearchAggregator:
    """
    Combines OpenAI and optional Perplexity research into a single summary.
    """

    def __init__(self, openai_researcher, perplexity_researcher=None):
        self.openai_researcher = openai_researcher
        self.perplexity_researcher = perplexity_researcher

    async def research_event(self, event: dict, markets: List[dict]) -> str:
        """
        Research an event using available researchers and return a combined text summary.
        """
        summaries = []

        try:
            openai_result = await self.openai_researcher.research_event(event)
            if openai_result and "rationale" in openai_result:
                summaries.append(f"OpenAI Research: {openai_result.get('rationale', '')}")
        except Exception as e:
            pass

        if self.perplexity_researcher:
            try:
                perplexity_result = await self.perplexity_researcher.research_event(event)
                if perplexity_result and "rationale" in perplexity_result:
                    summaries.append(f"Perplexity Research: {perplexity_result.get('rationale', '')}")
            except Exception as e:
                pass

        return "\n\n".join(summaries) if summaries else "No research available"
