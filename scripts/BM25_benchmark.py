"""
benchmark_latency.py — Step 5

Reads:  data/indices/bm25.pkl
        data/processed/eval_queries.jsonl
Writes: artifacts/results/bm25_latency.json

Methodology:
- Warm-up pass (5 queries) to prime Python internals and OS file cache
- Timed pass over N queries (default: all 500, or --n-queries to cap)
- Reports p50, p95, p99, mean, min, max latencies in milliseconds
- Flags any query that breaches the p95 SLA (200ms)

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --n-queries 100
    python scripts/benchmark_latency.py --top-k 100 --n-queries 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

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
DEFAULT_OUTPUT = BASE / "artifacts" / "results" / "bm25_latency.json"
DEFAULT_TOP_K    = 10
DEFAULT_N        = None       # None = use all available queries
WARMUP_N         = 5

# SLA thresholds (ms) — from project success criteria
SLA_P50  = 50.0
SLA_P95  = 200.0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    """Compute percentile without numpy (fallback)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo)


def compute_stats(latencies_ms: list[float]) -> dict:
    """Compute latency statistics. Uses numpy if available, else pure Python."""
    if not latencies_ms:
        return {}

    if _NUMPY:
        arr = np.array(latencies_ms)
        stats = {
            "n_queries":  int(len(arr)),
            "mean_ms":    float(np.mean(arr)),
            "std_ms":     float(np.std(arr)),
            "min_ms":     float(np.min(arr)),
            "p25_ms":     float(np.percentile(arr, 25)),
            "p50_ms":     float(np.percentile(arr, 50)),
            "p75_ms":     float(np.percentile(arr, 75)),
            "p95_ms":     float(np.percentile(arr, 95)),
            "p99_ms":     float(np.percentile(arr, 99)),
            "max_ms":     float(np.max(arr)),
        }
    else:
        stats = {
            "n_queries":  len(latencies_ms),
            "mean_ms":    sum(latencies_ms) / len(latencies_ms),
            "std_ms":     0.0,   # omit without numpy
            "min_ms":     min(latencies_ms),
            "p25_ms":     _percentile(latencies_ms, 25),
            "p50_ms":     _percentile(latencies_ms, 50),
            "p75_ms":     _percentile(latencies_ms, 75),
            "p95_ms":     _percentile(latencies_ms, 95),
            "p99_ms":     _percentile(latencies_ms, 99),
            "max_ms":     max(latencies_ms),
        }

    # Round to 2dp for readability
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Query loader (minimal, re-used from run_lexical_eval pattern)
# ---------------------------------------------------------------------------

def load_queries(path: Path, limit: Optional[int] = None) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: list[tuple[str, str]] = []
    with jsonlines.open(path) as reader:
        for idx, record in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            if not isinstance(record, dict):
                continue
            qid  = str(record.get("id") or record.get("qid") or idx)
            text = str(record.get("text") or record.get("query") or "").strip()
            if text:
                queries.append((qid, text))
    return queries


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    index_path: Path,
    queries_path: Path,
    output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    n_queries: Optional[int] = DEFAULT_N,
) -> None:

    # --- Load ---
    index = BM25Index.load(index_path)
    queries = load_queries(queries_path, limit=n_queries)

    if not queries:
        logger.error("No queries loaded — aborting.")
        sys.exit(1)

    target_n = len(queries)
    logger.info(f"Benchmarking {target_n} queries (top_k={top_k}) …")

    # --- Warm-up ---
    warmup_queries = queries[:WARMUP_N]
    logger.info(f"Warm-up pass ({len(warmup_queries)} queries) …")
    for _, qtext in warmup_queries:
        index.search(qtext, top_k=top_k)

    # --- Timed pass ---
    latencies_ms: list[float] = []
    slow_queries: list[dict] = []   # queries breaching p95 SLA

    try:
        from tqdm import tqdm
        iterator = tqdm(queries, desc="Benchmarking", unit="query")
    except ImportError:
        iterator = queries  # type: ignore[assignment]

    t_wall_start = time.perf_counter()

    for qid, qtext in iterator:
        _, _, lat_ms = index.search(qtext, top_k=top_k)
        latencies_ms.append(lat_ms)

        if lat_ms > SLA_P95:
            slow_queries.append({"query_id": qid, "query": qtext, "latency_ms": round(lat_ms, 2)})

    total_wall_ms = (time.perf_counter() - t_wall_start) * 1000

    # --- Stats ---
    stats = compute_stats(latencies_ms)
    stats["total_wall_ms"]      = round(total_wall_ms, 1)
    stats["throughput_qps"]     = round(target_n / (total_wall_ms / 1000), 1)
    stats["top_k"]              = top_k
    stats["corpus_size"]        = index.corpus_size
    stats["sla_p50_ms"]         = SLA_P50
    stats["sla_p95_ms"]         = SLA_P95
    stats["p50_sla_pass"]       = stats["p50_ms"] <= SLA_P50
    stats["p95_sla_pass"]       = stats["p95_ms"] <= SLA_P95
    stats["n_slow_queries"]     = len(slow_queries)

    if slow_queries:
        # Store up to 10 slow query examples for inspection
        stats["slow_query_examples"] = slow_queries[:10]

    # --- Write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- Report ---
    _print_report(stats)
    logger.info(f"Latency report written → {output_path}")


def _print_report(stats: dict) -> None:
    """Print a human-readable summary to stdout."""
    p50_ok = "✅" if stats.get("p50_sla_pass") else "❌"
    p95_ok = "✅" if stats.get("p95_sla_pass") else "❌"

    lines = [
        "",
        "┌─────────────────────────────────────────┐",
        "│          BM25 Latency Benchmark          │",
        "├─────────────────────────────────────────┤",
        f"│  Queries benchmarked : {stats.get('n_queries', 0):>6,}            │",
        f"│  Corpus size         : {stats.get('corpus_size', 0):>6,}            │",
        f"│  Top-k               : {stats.get('top_k', '?'):>6}            │",
        f"│  Throughput          : {stats.get('throughput_qps', 0):>6.1f} qps          │",
        "├─────────────────────────────────────────┤",
        f"│  Mean                : {stats.get('mean_ms', 0):>6.1f} ms           │",
        f"│  Std dev             : {stats.get('std_ms', 0):>6.1f} ms           │",
        f"│  Min                 : {stats.get('min_ms', 0):>6.1f} ms           │",
        f"│  p50  (SLA <50ms)  {p50_ok}: {stats.get('p50_ms', 0):>6.1f} ms           │",
        f"│  p75                 : {stats.get('p75_ms', 0):>6.1f} ms           │",
        f"│  p95  (SLA <200ms) {p95_ok}: {stats.get('p95_ms', 0):>6.1f} ms           │",
        f"│  p99                 : {stats.get('p99_ms', 0):>6.1f} ms           │",
        f"│  Max                 : {stats.get('max_ms', 0):>6.1f} ms           │",
        f"│  Slow queries (>{SLA_P95:.0f}ms): {stats.get('n_slow_queries', 0):>3}             │",
        "└─────────────────────────────────────────┘",
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark BM25 query latency.")
    p.add_argument("--index",     type=Path, default=DEFAULT_INDEX)
    p.add_argument("--queries",   type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--top-k",     type=int,  default=DEFAULT_TOP_K)
    p.add_argument(
        "--n-queries", type=int, default=DEFAULT_N,
        help="Number of queries to benchmark. Default: all.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(
        index_path=args.index,
        queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
        n_queries=args.n_queries,
    )
