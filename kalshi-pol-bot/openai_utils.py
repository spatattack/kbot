"""
Utilities for working with OpenAI Chat Completions API with structured outputs.
"""

from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar
from pydantic import BaseModel
import openai

T = TypeVar("T", bound=BaseModel)


async def responses_parse_pydantic(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    response_format: Type[T],
    reasoning_effort: str = "low",
    text_verbosity: str = "medium",
) -> T:
    """
    Parse structured output using Chat Completions API with structured outputs.
    
    Uses OpenAI's built-in structured output feature to ensure JSON matches the Pydantic schema.
    """
    try:
        # Use standard chat completions with structured outputs
        completion = await client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        
        # Extract the parsed object
        parsed = completion.choices[0].message.parsed
        
        if parsed is None:
            raise RuntimeError("OpenAI returned None for parsed output")
        
        return parsed
        
    except Exception as e:
        # Log the error with more context
        raise RuntimeError(f"Failed to validate response against {response_format.__name__}: {e}")