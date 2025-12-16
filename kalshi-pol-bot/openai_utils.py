"""
Utility helpers for working with OpenAI in the trading bot.

We intentionally implement `responses_parse_pydantic` using the Chat Completions
API instead of the Responses API so that it works with models that do not
support `reasoning.effort` or other Responses-only parameters.
"""

from __future__ import annotations

import json
from typing import Type, TypeVar, Sequence, Mapping, Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


async def responses_parse_pydantic(
    client: AsyncOpenAI,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    response_format: Type[T],
    **_: Any,
) -> T:
    """
    Call OpenAI and parse the result directly into a Pydantic model.

    Notes:
    - We ignore extra keyword arguments like `reasoning_effort` or `text_verbosity`
      so callers can pass them without breaking models that don't support them.
    - To use `response_format={"type": "json_object"}`, OpenAI requires that the
      messages mention "json" somewhere. We enforce that here.
    """

    # Prepend a system message that explicitly asks for JSON
    messages_with_json = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON generator. "
                "Always respond with a single valid JSON object that matches the "
                "expected schema. Do not include any text outside the JSON."
            ),
        },
        *list(messages),
    ]

    completion = await client.chat.completions.create(
        model=model,
        messages=messages_with_json,
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI returned an empty response")

    # Some models may already return a dict-like object, but usually this is JSON text.
    if isinstance(content, dict):
        data = content
    else:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode JSON from OpenAI response: {e}: {content!r}"
            ) from e

    try:
        return response_format.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Failed to validate response against {response_format.__name__}: {e}"
        ) from e
