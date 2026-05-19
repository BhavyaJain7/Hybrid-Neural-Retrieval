import json
from pathlib import Path
from loguru import logger
from neural_search.config import settings as global_settings
from neural_search.ingestion.parser import parse_document, parse_directory
from neural_search.ingestion.chunker import chunk_pages, Chunk


def _export_jsonl(chunks: list[Chunk], path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Exported {len(chunks)} chunks to {path}")


def run_ingestion(
    source: Path,
    sparse_retriever=None,
    dense_retriever=None,
    reset: bool = False,
    export_snapshot: bool = True,
    collection_slug: str = "default",
    settings_obj=None,
) -> list[Chunk]:
    """
    Full ingestion pipeline: parse → chunk → index → (optional) JSONL snapshot.

    When `source` is a single file, chunks are *added* to the existing BM25
    corpus (incremental) rather than replacing it.  When `source` is a
    directory (or `reset=True`), the index is rebuilt from scratch.
    """
    settings = settings_obj or global_settings

    if reset:
        logger.warning("Reset flag set — wiping existing indexes")
        if sparse_retriever:
            sparse_retriever.reset()
        if dense_retriever:
            dense_retriever.reset()

    if source.is_dir():
        pages = parse_directory(source)
    elif source.is_file():
        pages = parse_document(source)
    else:
        logger.error(f"Source path does not exist: {source}")
        return []

    if not pages:
        logger.warning("No pages parsed — check document content and format")
        return []

    chunks = chunk_pages(pages)
    if not chunks:
        logger.warning("No chunks produced — check chunking config")
        return []

    # Determine whether this is a full re-index or incremental add
    is_incremental = source.is_file() and not reset

    if export_snapshot:
        snapshot_path = settings.snapshot_path_for(collection_slug)
        # Append in incremental mode so previous chunks aren't clobbered
        write_mode = "a" if is_incremental else "w"
        _export_jsonl(chunks, snapshot_path, mode=write_mode)

    if sparse_retriever:
        if is_incremental:
            logger.info("Adding chunks to BM25 sparse retriever (incremental)...")
            sparse_retriever.add(chunks)
        else:
            logger.info("Rebuilding BM25 sparse retriever (full index)...")
            sparse_retriever.index(chunks)

    if dense_retriever:
        logger.info("Upserting into Qdrant dense retriever...")
        dense_retriever.upsert(chunks)

    logger.info(f"Ingestion complete — {len(chunks)} chunks from {source}")
    return chunks
