"""
scripts/eval_mrr_latency.py

Production-grade MRR@K vs Latency evaluation.

Key features:
- Statistically correct latency percentiles (P50, P95, P99, std)
- Warmup phase to remove cold-start bias
- Failure-safe execution
- Reranker loaded from committed benchmark JSON by default (no runtime reranking)
- --rerank-live flag to force live reranker evaluation
- Configurable reranker candidate pool via --rerank-k
- Optional query shuffling
- Dual table view: sorted by MRR and by efficiency

Usage:
    python scripts/eval_mrr_latency.py --eval-dir evaluation --collection base
    python scripts/eval_mrr_latency.py --eval-dir evaluation --collection base --rerank-live --rerank-k 50
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

from neural_search.evaluation.dataset import load_dataset
from neural_search.evaluation.metrics import mrr
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.learned import LearnedHybridFusion
from neural_search.retrieval.sparse import BM25sRetriever

RERANKER_MODE_KEY = "Hybrid+Reranker"


# ── Benchmark Loader (reranker default path) ──────────────────────────────────

def load_reranker_from_benchmark(eval_dir: Path) -> dict | None:
    """
    Reads pre-committed phase*.json for Hybrid RRF + Reranker stats.
    Scans results/ newest-first. Returns formatted stats dict or None.

    Note: avg_latency_ms is derived from total eval_latency_ms / n_queries.
    This reflects amortized batch cost, not per-query production latency.
    P50/P95/P99/std are unavailable from phase JSONs.
    """
    results_dir = eval_dir / "results"
    if not results_dir.exists():
        return None

    benchmark_keys = ["Hybrid RRF + Reranker", "Hybrid+Reranker", "Hybrid RRF + Rerank"]

    for path in sorted(results_dir.glob("phase*.json"), reverse=True):
        data = json.loads(path.read_text())
        overall = data.get("overall", {})

        entry = None
        for key in benchmark_keys:
            if key in overall:
                entry = overall[key]
                break

        if not entry:
            continue

        n = entry.get("queries_evaluated", 0)
        total_latency_ms = entry.get("eval_latency_ms", 0.0)
        avg_lat = round(total_latency_ms / n, 2) if n else 0.0
        mrr_val = round(entry.get("MRR", 0.0), 4)
        efficiency = round((mrr_val / avg_lat) * 1000, 4) if avg_lat > 0 else 0.0

        return {
            "mrr_at_k": mrr_val,
            "avg_latency_ms": avg_lat,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "std_latency_ms": None,
            "mrr_per_sec": efficiency,
            "queries_evaluated": n,
            "failures": 0,
            "source": path.name,
            "latency_note": "amortized batch — not per-query production latency",
        }

    return None


# ── Live Evaluation ───────────────────────────────────────────────────────────

def evaluate_mrr_latency(dataset, retrieve_fn, k: int, warmup: int = 5) -> dict:
    """
    Runs retrieve_fn over all labeled queries with warmup.
    Returns MRR@k and statistically correct latency percentiles.
    """
    queries = dataset.labeled_queries()

    for q in queries[:warmup]:
        try:
            retrieve_fn(q.text, k)
        except Exception:
            pass

    mrr_scores = []
    latencies_ms = []
    failures = 0

    for query in queries:
        relevant = dataset.get_relevant(query.id)
        if not relevant:
            continue
        try:
            t0 = time.perf_counter()
            results = retrieve_fn(query.text, k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            failures += 1
            continue

        result_ids = [r["chunk_id"] for r in results]
        mrr_scores.append(mrr(result_ids, relevant))
        latencies_ms.append(elapsed_ms)

    n = len(mrr_scores)

    if n == 0:
        return {
            "mrr_at_k": 0.0, "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0, "std_latency_ms": 0.0,
            "mrr_per_sec": 0.0, "queries_evaluated": 0, "failures": failures,
        }

    avg_mrr = round(sum(mrr_scores) / n, 4)
    avg_lat = round(sum(latencies_ms) / n, 2)
    p50 = round(np.percentile(latencies_ms, 50), 2)
    p95 = round(np.percentile(latencies_ms, 95), 2)
    p99 = round(np.percentile(latencies_ms, 99), 2)
    std = round(statistics.stdev(latencies_ms), 2) if n > 1 else 0.0
    efficiency = round((avg_mrr / avg_lat) * 1000, 4) if avg_lat > 0 else 0.0

    return {
        "mrr_at_k": avg_mrr,
        "avg_latency_ms": avg_lat,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "std_latency_ms": std,
        "mrr_per_sec": efficiency,
        "queries_evaluated": n,
        "failures": failures,
    }


# ── Table Rendering ───────────────────────────────────────────────────────────

def _fmt(val, fmt) -> str:
    return f"{val:{fmt}}" if val is not None else f"{'n/a':>10}"


def print_table(results: dict, k: int, sort_by: str = "mrr") -> None:
    cols = ["MRR@K", "Avg", "P50", "P95", "P99", "Std", "MRR/s", "Q", "Fail"]
    col_w = 26
    metric_w = 10

    header = f"{'Mode':<{col_w}}" + "".join(f"{c:>{metric_w}}" for c in cols)
    sep = "-" * len(header)
    sort_label = "sorted by MRR" if sort_by == "mrr" else "sorted by Efficiency"

    print(f"\n{'MRR@' + str(k) + ' vs Latency — ' + sort_label:^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    key = "mrr_at_k" if sort_by == "mrr" else "mrr_per_sec"
    for mode, m in sorted(results.items(), key=lambda x: x[1][key], reverse=True):
        tag = " *" if "source" in m else ""
        print(
            f"{mode + tag:<{col_w}}"
            + _fmt(m["mrr_at_k"],          f">{metric_w}.4f")
            + _fmt(m["avg_latency_ms"],    f">{metric_w}.1f")
            + _fmt(m["p50_latency_ms"],    f">{metric_w}.1f")
            + _fmt(m["p95_latency_ms"],    f">{metric_w}.1f")
            + _fmt(m["p99_latency_ms"],    f">{metric_w}.1f")
            + _fmt(m["std_latency_ms"],    f">{metric_w}.1f")
            + _fmt(m["mrr_per_sec"],       f">{metric_w}.4f")
            + _fmt(m["queries_evaluated"], f">{metric_w}")
            + _fmt(m["failures"],          f">{metric_w}")
        )

    print(sep)
    print("Latency columns in ms. MRR/s = (MRR / avg_latency_ms) × 1000.")
    if any("source" in v for v in results.values()):
        print("* Latency from committed benchmark JSON (amortized batch) — not per-query production latency.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MRR@K vs Latency evaluation — all retrieval modes.")
    parser.add_argument("--eval-dir",    default="evaluation")
    parser.add_argument("--collection",  default="base")
    parser.add_argument("--k",           type=int, default=10)
    parser.add_argument("--rerank-k",    type=int, default=50, help="Candidate pool size for live reranking")
    parser.add_argument("--rerank-live", action="store_true",  help="Force live reranker eval instead of loading from benchmark")
    parser.add_argument("--shuffle",     action="store_true",  help="Shuffle query order before eval")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    dataset = load_dataset(
        queries_path=eval_dir / "queries.json",
        relevance_path=eval_dir / "relevance.json",
    )

    if args.shuffle:
        random.shuffle(dataset.labeled_queries())

    print(f"Loaded {dataset.coverage}")

    # ── Retrievers ────────────────────────────────────────────────────────────
    sparse  = BM25sRetriever(collection_slug=args.collection)
    dense   = QdrantRetriever(collection_slug=args.collection)
    hybrid  = HybridRetriever(sparse=sparse, dense=dense)
    learned = LearnedHybridFusion(sparse=sparse, dense=dense)

    modes = {
        "BM25":    lambda q, k: sparse.search(q, k=k),
        "Dense":   lambda q, k: dense.search(q, k=k),
        "Hybrid":  lambda q, k: hybrid.search(q, k=k),
        "Learned": lambda q, k: learned.search(q, k=k),
    }

    # ── Live eval for all non-reranker modes ──────────────────────────────────
    results = {}
    for name, fn in modes.items():
        print(f"Evaluating: {name} ...", flush=True)
        results[name] = evaluate_mrr_latency(dataset, fn, k=args.k)

    # ── Reranker: benchmark first, live only if --rerank-live ─────────────────
    print(f"Evaluating: {RERANKER_MODE_KEY} ...", flush=True)

    if not args.rerank_live:
        reranker_stats = load_reranker_from_benchmark(eval_dir)
        if reranker_stats:
            print(f"  Loaded from {reranker_stats['source']} — use --rerank-live to force runtime eval.")
            results[RERANKER_MODE_KEY] = reranker_stats
        else:
            print("  No committed benchmark found. Falling back to live eval (this will be slow).")
            args.rerank_live = True

    if args.rerank_live:
        from neural_search.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

        def hybrid_rerank_fn(q, k):
            candidates = hybrid.search(q, k=args.rerank_k)
            return reranker.rerank(q, candidates, top_k=k)

        print(f"  Running live reranker (candidate pool={args.rerank_k}) — expected ~150s+")
        results[RERANKER_MODE_KEY] = evaluate_mrr_latency(dataset, hybrid_rerank_fn, k=args.k)

    # ── Print both views ──────────────────────────────────────────────────────
    print_table(results, args.k, sort_by="mrr")
    print_table(results, args.k, sort_by="efficiency")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else eval_dir / "results" / "mrr_latency_prod.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"k": args.k, "rerank_k": args.rerank_k, "results": results}, indent=2))
    print(f"\nSaved → {out_path}")

    try:
        dense._client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
