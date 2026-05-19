from __future__ import annotations

import random

from groq import Groq
from loguru import logger

from neural_search.config import get_settings
from neural_search.synthesis.prompt import build_prompt

settings = get_settings()

_MAX_BACKOFF = 32
_DEFAULT_MODEL = "llama-3.1-8b-instant"
_DEPRECATED_MODELS = {"llama3-8b-8192"}

_FALLBACK_RESPONSE = {
    "answer": "Unable to generate an answer at this time.",
    "sources_used": [],
    "model": _DEFAULT_MODEL,
}


def _get(chunk, key: str):
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


class GroqSynthesizer:
    def __init__(self, settings_obj=None):
        s = settings_obj or settings
        s.assert_groq_configured()
        model = s.groq_model
        if model in _DEPRECATED_MODELS:
            logger.warning(f"Groq model '{model}' is deprecated — using {_DEFAULT_MODEL}")
            model = _DEFAULT_MODEL
        self._client = Groq(api_key=s.groq_api_key)
        self._model = model

    def synthesize(self, query: str, chunks: list, retries: int = 3) -> dict | None:
        """
        Synthesize an answer from retrieved chunks.
        Returns None if synthesis should not proceed (handled upstream by confidence gate).
        """
        if not settings.synthesis_enabled:
            return None

        context_chunks = chunks[:5]
        prompt = build_prompt(query, context_chunks)

        sources = [
            {
                "chunk_id": _get(c, "chunk_id"),
                "source_file": _get(c, "source_file"),
                "source_url": _get(c, "source_url"),
                "page": _get(c, "page"),
            }
            for c in context_chunks
        ]

        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    max_tokens=512,
                    temperature=0.2,
                )
                return {
                    "answer": response.choices[0].message.content.strip(),
                    "sources_used": sources,
                    "model": self._model,
                }
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Groq synthesis failed after {retries} attempts: {e}")
                    return _FALLBACK_RESPONSE
                wait = min(2 ** attempt + random.uniform(0, 1), _MAX_BACKOFF)
                logger.warning(f"Groq attempt {attempt + 1} failed: {e} — retrying in {wait:.1f}s")
                import time
                time.sleep(wait)

        return _FALLBACK_RESPONSE
