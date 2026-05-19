from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from .parser import ParsedPage

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _make_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    chunk_index: int
    text: str
    token_count: int          # accurate — tiktoken cl100k_base
    metadata: dict = field(default_factory=dict)


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # character-based splitter; token count stored separately
    )

    all_chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        if not page.text.strip():
            continue
        for text in splitter.split_text(page.text):
            clean_text = text.strip()
            if not clean_text or len(clean_text.split()) < 50:
                continue
            all_chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(page.source_file, global_index),
                    doc_id=page.doc_id,
                    source_file=page.source_file,
                    page=page.page,
                    chunk_index=global_index,
                    text=clean_text,
                    token_count=_count_tokens(clean_text),  # accurate
                    metadata=page.metadata,
                )
            )
            global_index += 1

    logger.debug(f"Chunked {len(pages)} pages → {len(all_chunks)} chunks")
    return all_chunks
