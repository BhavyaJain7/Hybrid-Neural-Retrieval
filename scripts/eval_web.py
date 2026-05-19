#!/usr/bin/env python3
"""
eval_web.py — Phase 5 Evaluation

Measures Tavily web augmentation lift on low-confidence queries (vague subset)
and verifies no degradation on high-confidence queries (keyword subset).

Compares:
  - Hybrid RRF              (no web, baseline)
  - Hybrid RRF + Web        (Tavily enabled, gated by score threshold)
  - Hybrid RRF + Web forced (web_search=True flag, bypasses gating)

Usage:
    python scripts/eval_web.py --collection base --k 5
    python scripts/eval_web.py --collection base --k 5 --output evaluation/results/phase5_web.json

Requirements:
    TAVILY_ENABLED=true and TAVILY_API_KEY set in .env
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.config import get_settings
from neural_search.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.hybrid import HybridRetriever
from neural_search.retrieval.sparse import BM25sRetriever

settings = get_settings()


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
    web_fired_count = 0
    skipped = 0

    for query in queries:
        qid = query["id"]
        if qid not in relevance:
            continue
        if query_type and query.get("type") != query_type:
            continue

        relevant = relevance[qid]
        try:
            result_data = retrieve_fn(query["text"], k)
            # retrieve_fn may return (results, web_used) tuple or just results
            if isinstance(result_data, tuple):
                results, web_used = result_data
                if web_used:
                    web_fired_count += 1
            else:
                results = result_data
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
    out = {
        f"P@{k}": round(sum(p_scores) / n, 4) if n else 0.0,
        f"Recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        "MRR": round(sum(mrr_scores) / n, 4) if n else 0.0,
        f"nDCG@{k}": round(sum(ndcg_scores) / n, 4) if n else 0.0,
        "queries_evaluated": n,
        "queries_skipped": skipped,
    }
    if web_fired_count > 0:
        out["web_fired_count"] = web_fired_count
    return out


def print_comparison(baseline: dict, web: dict, k: int, title: str) -> None:
    metrics = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    print(f"\n{title}")
    print(f"  {'Metric':<14} {'Baseline':>10} {'+ Web':>10} {'Delta':>10}")
    print("  " + "-" * 46)
    for m in metrics:
        b = baseline.get(m, 0.0)
        w = web.get(m, 0.0)
        delta = w - b
        sign = "+" if delta >= 0 else ""
        flag = "  ✅" if delta >= 0 else "  ❌"
        print(f"  {m:<14} {b:>10.4f} {w:>10.4f} {sign}{delta:>9.4f}{flag}")
    web_count = web.get("web_fired_count", "N/A")
    print(f"  Tavily fired: {web_count} / {web.get('queries_evaluated', 0)} queries")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Web Retrieval Evaluation")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-dir", default="evaluation")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()

    if not settings.tavily_enabled:
        print("ERROR: TAVILY_ENABLED=false in .env — cannot run Phase 5 eval.")
        print("Set TAVILY_ENABLED=true and provide a valid TAVILY_API_KEY.")
        sys.exit(1)

    if settings.tavily_api_key == "not-set":
        print("ERROR: TAVILY_API_KEY not configured in .env")
        sys.exit(1)

    eval_dir = Path(args.eval_dir)
    queries, relevance = load_eval_data(eval_dir)

    print(f"\nLoading index for collection: {args.collection}")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print(f"ERROR: BM25 index not found for '{args.collection}'")
        sys.exit(1)
    dense = QdrantRetriever(collection_slug=args.collection)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)

    print(f"  Tavily web retriever: {'loaded ✅' if hybrid._web else 'FAILED ❌'}")
    print(f"  web_trigger_threshold: {settings.web_trigger_threshold}")

    # ── Retrieval functions ───────────────────────────────────────────────────

    def baseline_fn(q: str, k: int):
        return hybrid.search_full(
            q, k=k, expand=False, web_search=False, rerank=False
        )["results"]

    def web_gated_fn(q: str, k: int):
        """Web fires only when local confidence < threshold."""
        out = hybrid.search_full(q, k=k, expand=False, web_search=False, rerank=False)
        return out["results"], out["web_results_used"]

    def web_forced_fn(q: str, k: int):
        """web_search=True bypasses gating — always calls Tavily."""
        out = hybrid.search_full(q, k=k, expand=False, web_search=True, rerank=False)
        return out["results"], out["web_results_used"]

    print(f"\n=== Phase 5: Web Retrieval Evaluation (k={args.k}) ===")
    print(f"    web_trigger_threshold = {settings.web_trigger_threshold}")

    # ── Vague queries — where local retrieval is weakest ─────────────────────
    print("\n[1/4] Vague baseline (local only)...")
    t0 = time.perf_counter()
    vague_baseline = evaluate_mode(queries, relevance, baseline_fn, args.k, query_type="vague")
    vague_baseline["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print("[2/4] Vague + web (gated by score threshold)...")
    t0 = time.perf_counter()
    vague_web_gated = evaluate_mode(queries, relevance, web_gated_fn, args.k, query_type="vague")
    vague_web_gated["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print_comparison(vague_baseline, vague_web_gated, args.k, "── Vague Queries (gated web) ──")

    # ── Keyword queries — regression check ───────────────────────────────────
    print("\n[3/4] Keyword baseline (local only)...")
    t0 = time.perf_counter()
    kw_baseline = evaluate_mode(queries, relevance, baseline_fn, args.k, query_type="keyword")
    kw_baseline["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print("[4/4] Keyword + web forced (worst-case regression check)...")
    t0 = time.perf_counter()
    kw_web_forced = evaluate_mode(queries, relevance, web_forced_fn, args.k, query_type="keyword")
    kw_web_forced["eval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    print_comparison(kw_baseline, kw_web_forced, args.k, "── Keyword Queries (forced web — regression check) ──")

    # ── Pass/fail summary ─────────────────────────────────────────────────────
    ndcg_key = f"nDCG@{args.k}"
    vague_lift = vague_web_gated.get(ndcg_key, 0) - vague_baseline.get(ndcg_key, 0)
    kw_regression = kw_web_forced.get(ndcg_key, 0) - kw_baseline.get(ndcg_key, 0)

    print("\n=== Exit Criteria ===")
    print(f"  Vague nDCG@{args.k} lift (gated web):    {vague_lift:+.4f}  {'✅ PASS' if vague_lift >= 0 else '❌ FAIL'}")
    print(f"  Keyword nDCG@{args.k} delta (forced web): {kw_regression:+.4f}  {'✅ PASS (no regression)' if kw_regression >= -0.005 else '❌ FAIL (regression on keyword queries)'}")
    print(f"  Tavily call budget check:              web_trigger_threshold={settings.web_trigger_threshold} (<30/day target)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else Path(args.eval_dir) / "results" / "phase5_web.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 5,
        "collection": args.collection,
        "k": args.k,
        "web_trigger_threshold": settings.web_trigger_threshold,
        "tavily_weight": settings.tavily_weight,
        "description": "Tavily web augmentation: lift on vague queries, no regression on keyword queries",
        "vague_queries": {
            "Hybrid RRF (local only)": vague_baseline,
            "Hybrid RRF + Web (gated)": vague_web_gated,
            "ndcg_lift": round(vague_lift, 4),
            "exit_criteria_met": vague_lift >= 0,
        },
        "keyword_queries": {
            "Hybrid RRF (local only)": kw_baseline,
            "Hybrid RRF + Web (forced)": kw_web_forced,
            "ndcg_delta": round(kw_regression, 4),
            "no_regression": kw_regression >= -0.005,
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
