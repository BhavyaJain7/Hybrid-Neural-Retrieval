"""
CLI: Ingest a directory or single file into Neural Search indexes.

Usage:
    python scripts/ingest_documents.py --input-dir ./data/documents
    python scripts/ingest_documents.py --input-dir ./data/documents --reset
"""
import argparse
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.retrieval.sparse import BM25sRetriever
from neural_search.retrieval.dense import QdrantRetriever


def main():
    parser = argparse.ArgumentParser(description="Neural Search — Document Ingestion CLI")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.documents_path_for("default"),
        help="Path to directory or single file to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing indexes and rebuild from scratch",
    )
    args = parser.parse_args()

    settings.ensure_dirs()

    logger.info(f"Starting ingestion from: {args.input_dir}")
    logger.info(f"Reset: {args.reset}")

    sparse = BM25sRetriever(collection_slug="default")
    dense = QdrantRetriever(collection_slug="default")

    chunks = run_ingestion(
        source=args.input_dir,
        sparse_retriever=sparse,
        dense_retriever=dense,
        reset=args.reset,
    )

    logger.success(f"Done — {len(chunks)} chunks indexed")


if __name__ == "__main__":
    main()
