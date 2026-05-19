from __future__ import annotations

import threading
from typing import Optional

from groq import Groq
from loguru import logger

from neural_search.config import get_settings

settings = get_settings()

_CLIENT: Optional[Groq] = None
_CLIENT_LOCK = threading.Lock()


def _get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = Groq(api_key=settings.groq_api_key)
                logger.debug("Groq client initialised (expander)")
    return _CLIENT

_EXPANSION_PROMPT = (
    "You are a search query expansion assistant.\n"
    "Given a user query, generate {n} alternative phrasings that express the same intent "
    "using different vocabulary. These will improve semantic search recall.\n\n"
    "Rules:\n"
    "- Each alternative must use clearly different vocabulary from the original\n"
    "- Do not change the core intent\n"
    "- Output ONLY the alternatives, one per line, no numbering, no explanation\n\n"
    "Query: {query}"
)


def expand_query(query: str, n: int = 2) -> list[str]:
    """
    Return [original_query] + n alternative phrasings.
    Falls back to [original_query] silently on any error.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": _EXPANSION_PROMPT.format(query=query, n=n),
                }
            ],
            max_tokens=200,
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
        expansions = [line.strip() for line in raw.splitlines() if line.strip()][:n]
        result = [query] + expansions
        logger.debug(f"Query expanded to: {result}")
        return result
    except Exception as e:
        logger.warning(f"Query expansion failed, using original only: {e}")
        return [query]
