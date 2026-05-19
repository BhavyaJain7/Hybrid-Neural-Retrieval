from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

try:
    import jsonlines
except ImportError as e:
    raise ImportError("jsonlines is required. Install with: uv add jsonlines") from e

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Resolve src layout regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural_search.lexical_search.bm25_index import BM25Index

BASE = Path(__file__).resolve().parent.parent
os.makedirs(BASE / "data" / "indices", exist_ok=True)
DEFAULT_CORPUS = BASE / "data" / "raw" / "corpus.jsonl"
DEFAULT_OUTPUT = BASE / "data" / "indices" / "bm25.pkl"

def load_corpus(
    path: Path,
    limit: int | None = None,
) -> list[tuple[str, str]]:

    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    passages: list[tuple[str, str]] = []
    skipped = 0

    logger.info(f"Streaming corpus from {path} …")

    try:
        from tqdm import tqdm
        _wrap = lambda it, **kw: tqdm(it, **kw)  # noqa: E731
    except ImportError:
        _wrap = lambda it, **kw: it  # noqa: E731

    with jsonlines.open(path) as reader:
        wrapped = _wrap(reader, desc="Loading corpus", unit="line", total=None)
        for idx, record in enumerate(wrapped):
            if limit is not None and idx >= limit:
                break

            # Robustness: handle non-dict records
            if not isinstance(record, dict):
                logger.warning(f"Line {idx}: expected dict, got {type(record).__name__} — skipping.")
                skipped += 1
                continue

            # Passage ID — fall back to line index if missing
            pid = record.get("id") or record.get("_id") or record.get("docid") or str(idx)

            # Text field
            text = record.get("text") or record.get("passage") or record.get("contents") or ""
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()

            if not text:
                logger.debug(f"Line {idx} (pid={pid}): empty text — skipping.")
                skipped += 1
                continue

            passages.append((str(pid), text))

    logger.info(
        f"Loaded {len(passages):,} passages "
        f"({skipped:,} skipped) from {path.name}"
    )
    return passages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(
    corpus_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> None:
    t_total = time.perf_counter()

    # --- Load corpus ---
    passages = load_corpus(corpus_path, limit=limit)

    if not passages:
        logger.error("No passages loaded — aborting.")
        sys.exit(1)

    # --- Build index ---
    index = BM25Index()
    index.build(passages, show_progress=True)

    # --- Persist ---
    index.save(output_path)

    elapsed = time.perf_counter() - t_total
    logger.info(
        f"Done. Total wall time: {elapsed:.1f}s | "
        f"Index: {output_path} | "
        f"Passages: {index.corpus_size:,}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25 index over MS MARCO corpus.")
    p.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help=f"Path to corpus.jsonl (default: {DEFAULT_CORPUS})",
    )
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output path for bm25.pkl (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of passages loaded (smoke test). Omit for full corpus.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        corpus_path=args.corpus,
        output_path=args.out,
        limit=args.limit,
    )
