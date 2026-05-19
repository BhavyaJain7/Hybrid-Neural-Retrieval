#!/usr/bin/env python3
"""
eval_expansion.py — Phase 4 Evaluation

Measures query expansion lift on the VAGUE query subset and verifies
no regression on keyword queries.

Compares:
  - Hybrid RRF           (baseline — no expansion)
  - Hybrid RRF + Expand  (expansion enabled)

Usage:
    python scripts/eval_expansion.py --collection base --k 5
    python scripts/eval_expansion.py --collection base --k 5 --output evaluation/results/phase4_expansion.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever


def load_eval_data(eval_dir: Path) -> tuple[list[dict], dict]:
    queries = json.loads((eval_dir / "queries.json").read_text())
    relevance_raw = json.loads((eval_dir / "relevance.json").read_text())
    labeled = {qid: set(ids) for qid, ids in relevance_raw.items() if ids}
    print(f"Loaded {len(queries)} queries, {len(labeled)} with labels")
    return queries, labeled


def evaluate_mode(
    queries: list[dict],
    relevance: dict[str, set],
    retrieve_fn,
    k: int,
    query_type: str | None = None,
) -> dict:
    p_scores, recall_scores, mrr_scores, ndcg_scores = [], [], [], []
    skipped = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue
        if query_type and query.get("type") != query_type:
            continue

        relevant = relevance[qid]
        try:
            results = retrieve_fn(query["text"], k)
        except Exception as e:
            print(f"  [WARN] {qid}: {e}")
            skipped += 1
            continue

        result_ids = [r["chunk_id"] for r in results]
        p_scores.append(precision_at_k(result_ids, relevant, k))
        recall_scores.append(recall_at_k(result_ids, relevant, k))
        mrr_scores.append(mrr(result_ids, relevant))
        ndcg_scores.append(ndcg_at_k(result_ids, relevant, k))

    n = len(p_scores)
    return {
        f"P@{k}": round(sum(p_scores) / n, 4) if n else 0.0,
        f"Recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        "MRR": round(sum(mrr_scores) / n, 4) if n else 0.0,
        f"nDCG@{k}": round(sum(ndcg_scores) / n, 4) if n else 0.0,
        "queries_evaluated": n,
        "queries_skipped": skipped,
    }


def print_comparison(baseline: dict, expanded: dict, k: int, title: str) -> None:
    metrics = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    print(f"\n{title}")
    print(f"  {'Metric':<14} {'Baseline':>10} {'Expanded':>10} {'Delta':>10}")
    print("  " + "-" * 46)
    for m in metrics:
        b = baseline.get(m, 0.0)
        e = expanded.get(m, 0.0)
        delta = e - b
        sign = "+" if delta >= 0 else ""
        flag = "  ✅" if delta >= 0 else "  ❌"
        print(f"  {m:<14} {b:>10.4f} {e:>10.4f} {sign}{delta:>9.4f}{flag}")
    n_b = baseline.get("queries_evaluated", 0)
    n_e = expanded.get("queries_evaluated", 0)
    print(f"  Queries evaluated: baseline={n_b}, expanded={n_e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Query Expansion Evaluation")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-dir", default="evaluation")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    queries, relevance = load_eval_data(eval_dir)

    print(f"\nLoading index for collection: {args.collection}")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print(f"ERROR: BM25 index not found for '{args.collection}'")
        sys.exit(1)
    dense = QdrantRetriever(collection_slug=args.collection)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    # ── Retrieval functions ───────────────────────────────────────────────────
    def baseline_fn(q: str, k: int):
        return hybrid.search_full(
            q, k=k, expand=False, query_type="vague", web_search=False, rerank=False
        )["results"]

    def expand_fn(q: str, k: int):
        return hybrid.search_full(
            q, k=k, expand=True, query_type="vague", web_search=False, rerank=False
        )["results"]

    def keyword_baseline_fn(q: str, k: int):
        return hybrid.search_full(
            q, k=k, expand=False, query_type="keyword", web_search=False, rerank=False
        )["results"]

    def keyword_expand_fn(q: str, k: int):
        # expand=True but query_type="keyword" → expansion suppressed by design
        return hybrid.search_full(
            q, k=k, expand=True, query_type="keyword", web_search=False, rerank=False
        )["results"]

    print(f"\n=== Phase 4: Query Expansion Evaluation (k={args.k}) ===")

    # ── Vague queries (where expansion should win) ────────────────────────────
    print("\n[1/4] Vague baseline (no expansion)...")
    t0 = time.perf_counter()
    vague_baseline = evaluate_mode(queries, relevance, baseline_fn, args.k, query_type="vague")
    vague_baseline["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print("[2/4] Vague + expansion (Groq rephrasing)...")
    t0 = time.perf_counter()
    vague_expanded = evaluate_mode(queries, relevance, expand_fn, args.k, query_type="vague")
    vague_expanded["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print_comparison(vague_baseline, vague_expanded, args.k, "── Vague Queries ──")

    # ── Keyword queries (regression check — expansion should NOT fire) ────────
    print("\n[3/4] Keyword baseline (no expansion)...")
    t0 = time.perf_counter()
    kw_baseline = evaluate_mode(queries, relevance, keyword_baseline_fn, args.k, query_type="keyword")
    kw_baseline["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print("[4/4] Keyword + expand flag (expansion suppressed by query_type)...")
    t0 = time.perf_counter()
    kw_expanded = evaluate_mode(queries, relevance, keyword_expand_fn, args.k, query_type="keyword")
    kw_expanded["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print_comparison(kw_baseline, kw_expanded, args.k, "── Keyword Queries (regression check — should be identical) ──")

    # ── Pass/fail summary ─────────────────────────────────────────────────────
    ndcg_key = f"nDCG@{args.k}"
    vague_lift = vague_expanded.get(ndcg_key, 0) - vague_baseline.get(ndcg_key, 0)
    kw_regression = kw_expanded.get(ndcg_key, 0) - kw_baseline.get(ndcg_key, 0)

    print("\n=== Exit Criteria ===")
    print(f"  Vague nDCG@{args.k} lift:     {vague_lift:+.4f}  {'✅ PASS' if vague_lift > 0 else '❌ FAIL (no lift on vague queries)'}")
    print(f"  Keyword nDCG@{args.k} delta:   {kw_regression:+.4f}  {'✅ PASS (no regression)' if kw_regression >= -0.001 else '❌ FAIL (keyword regression detected)'}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else Path(args.eval_dir) / "results" / "phase4_expansion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 4,
        "collection": args.collection,
        "k": args.k,
        "description": "Query expansion lift on vague queries; regression check on keyword queries",
        "vague_queries": {
            "Hybrid RRF (no expansion)": vague_baseline,
            "Hybrid RRF + Expansion": vague_expanded,
            "ndcg_lift": round(vague_lift, 4),
            "exit_criteria_met": vague_lift > 0,
        },
        "keyword_queries": {
            "Hybrid RRF (no expansion)": kw_baseline,
            "Hybrid RRF + Expansion (suppressed)": kw_expanded,
            "ndcg_delta": round(kw_regression, 4),
            "no_regression": kw_regression >= -0.001,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_path}")

    try:
        dense._client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
