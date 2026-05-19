from __future__ import annotations

import os


def _get(chunk, key: str, default=None):
    """Safely access a chunk field whether it's a dict or a dataclass/object."""
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _clean_source(source_file: str | None) -> str:
    """Return a readable source label from a raw file path."""
    if not source_file:
        return "Unknown source"
    name = os.path.basename(source_file)
    name = os.path.splitext(name)[0]          # strip .pdf / .docx etc.
    name = name.replace("_", " ").replace("-", " ")
    return name


def _reframe_query(query: str) -> str:
    """
    If the query is a bare keyword/phrase (no question mark, < 5 words,
    doesn't start with a question word), reframe it so the LLM can
    answer properly instead of refusing.

    Examples:
        'agentic'               → 'What do the documents say about agentic?'
        'attention mechanism'   → 'What do the documents say about attention mechanism?'
        'How do agents work?'   → unchanged (already a question)
    """
    query = query.strip()
    word_count = len(query.split())
    _question_starts = (
        "what", "how", "why", "when", "who", "where", "which",
        "explain", "describe", "list", "summarize", "summarise",
    )
    is_question = (
        query.endswith("?")
        or query.lower().split()[0] in _question_starts
    )
    if not is_question and word_count < 5:
        return f"What do the documents say about {query}?"
    return query


def build_prompt(query: str, chunks: list) -> dict:
    """
    Build the system + user prompt for Groq synthesis.

    Handles keyword queries by reframing them as topic questions.
    Cleans source filenames for readable citations.
    """
    effective_query = _reframe_query(query)

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = _clean_source(_get(chunk, "source_file"))
        page = _get(chunk, "page", "?")
        text = (_get(chunk, "text") or "").strip()
        context_blocks.append(f"[Source {i}: {source}, Page {page}]\n{text}")

    context = "\n\n".join(context_blocks)

    system = (
        "You are a knowledgeable document assistant. "
        "Your job is to answer the user's question using the provided document excerpts.\n\n"
        "Rules:\n"
        "- Base your answer on the provided context.\n"
        "- Cite the source name and page number for each key point, "
        "e.g. (Source 2, Page 5).\n"
        "- If the query is a keyword or topic, summarise what the documents "
        "say about it — do not refuse.\n"
        "- If the context contains only partial information, share what you "
        "found and note what is missing.\n"
        "- Only if the context contains absolutely no relevant information, "
        "respond with: 'The provided documents do not contain information "
        "about this topic.'\n"
        "- Keep the answer concise and well-structured."
    )

    user = f"Question: {effective_query}\n\nContext:\n{context}"

    return {"system": system, "user": user}
