"""
run_lexical_eval.py — Step 4

Reads:  data/indices/bm25.pkl
        data/processed/eval_queries.jsonl
Writes: artifacts/results/bm25_run.json  (ranx-compatible run format)

Output format:
    {
        "9652": {"0_3": 12.4, "1024_1": 10.8, ...},
        ...
    }

Usage:
    python scripts/run_lexical_eval.py
    python scripts/run_lexical_eval.py --index data/indices/bm25.pkl --queries data/processed/eval_queries.jsonl
    python scripts/run_lexical_eval.py --top-k 100   # retrieve deeper for recall@100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural_search.lexical_search.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = BASE / "data" / "indices" / "bm25.pkl"
DEFAULT_QUERIES = BASE / "data" / "processed" / "eval_queries.jsonl"
DEFAULT_OUTPUT = BASE / "artifacts" / "results" / "bm25_run.json"
DEFAULT_TOP_K   = 10


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[tuple[str, str]]:
    """
    Load eval queries from JSONL.

    Expected schema: {"id": "9652", "text": "what is …"}
    Tolerates alternate field names: qid/query_id for ID, query/question for text.

    Returns:
        List of (query_id, query_text) tuples — duplicates are deduplicated,
        keeping the first occurrence.
    """
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    skipped = 0

    with jsonlines.open(path) as reader:
        for idx, record in enumerate(reader):
            if not isinstance(record, dict):
                logger.warning(f"Line {idx}: expected dict — skipping.")
                skipped += 1
                continue

            qid = (
                record.get("id")
                or record.get("qid")
                or record.get("query_id")
                or str(idx)
            )
            qid = str(qid)

            text = (
                record.get("text")
                or record.get("query")
                or record.get("question")
                or ""
            )
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()

            if not text:
                logger.debug(f"Line {idx} (qid={qid}): empty query — skipping.")
                skipped += 1
                continue

            if qid in seen_ids:
                logger.warning(f"Duplicate query_id {qid!r} at line {idx} — skipping.")
                skipped += 1
                continue

            seen_ids.add(qid)
            queries.append((qid, text))

    logger.info(
        f"Loaded {len(queries):,} queries "
        f"({skipped:,} skipped) from {path.name}"
    )
    return queries


# ---------------------------------------------------------------------------
# Eval run
# ---------------------------------------------------------------------------

def run_eval(
    index_path: Path,
    queries_path: Path,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    t_total = time.perf_counter()

    # --- Load ---
    index = BM25Index.load(index_path)
    queries = load_queries(queries_path)

    if not queries:
        logger.error("No queries loaded — aborting.")
        sys.exit(1)

    # --- Search ---
    logger.info(f"Running {len(queries):,} queries (top_k={top_k}) …")
    run = index.search_batch(queries, top_k=top_k)

    # --- Integrity checks before writing ---
    n_results      = len(run)
    empty_queries  = [qid for qid, res in run.items() if not res]
    zero_score     = [
        qid for qid, res in run.items()
        if res and all(s == 0.0 for s in res.values())
    ]

    if empty_queries:
        logger.warning(
            f"{len(empty_queries)} queries returned zero results "
            f"(all-stopword or very short queries). "
            f"First 5: {empty_queries[:5]}"
        )

    if zero_score:
        logger.warning(
            f"{len(zero_score)} queries returned only zero-score results "
            f"(terms not in corpus). First 5: {zero_score[:5]}"
        )

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    file_size_kb = output_path.stat().st_size / 1024
    elapsed = time.perf_counter() - t_total

    logger.info(
        f"Wrote {n_results:,} query results → {output_path} "
        f"({file_size_kb:.0f} KB) in {elapsed:.1f}s"
    )

    # Summary
    avg_results = (
        sum(len(v) for v in run.values()) / n_results if n_results else 0
    )
    logger.info(
        f"Summary | queries: {n_results:,} | avg results/query: {avg_results:.1f} | "
        f"empty: {len(empty_queries)} | zero-score: {len(zero_score)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BM25 eval over 500 MS MARCO queries.")
    p.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX,
        help=f"Path to bm25.pkl (default: {DEFAULT_INDEX})",
    )
    p.add_argument(
        "--queries", type=Path, default=DEFAULT_QUERIES,
        help=f"Path to eval_queries.jsonl (default: {DEFAULT_QUERIES})",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output path for bm25_run.json (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Results per query (default: {DEFAULT_TOP_K})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        index_path=args.index,
        queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
    )
