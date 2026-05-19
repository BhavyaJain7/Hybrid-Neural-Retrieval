#!/usr/bin/env python3
"""
run_eval.py

Evaluation runner. Compares BM25, Dense, Hybrid, and Learned Fusion
against the labeled relevance dataset.

Includes per-type breakdown (keyword / semantic / vague) so you can
see exactly where each mode wins or fails.

Usage:
    python scripts/run_eval.py --collection base --k 5
    python scripts/run_eval.py --collection base --k 5 --phase 3
    python scripts/run_eval.py --collection base --k 5 --output evaluation/results/phase3.json
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
from neural_search.retrieval.learned import LearnedHybridFusion
from neural_search.retrieval.sparse import BM25sRetriever


def load_eval_data(eval_dir: Path) -> tuple[list[dict], dict]:
    queries_path = eval_dir / "queries.json"
    relevance_path = eval_dir / "relevance.json"

    if not queries_path.exists():
        print(f"ERROR: queries.json not found at {queries_path}")
        sys.exit(1)
    if not relevance_path.exists():
        print(f"ERROR: relevance.json not found at {relevance_path}")
        sys.exit(1)

    queries = json.loads(queries_path.read_text())
    relevance = json.loads(relevance_path.read_text())

    for qid, chunk_ids in relevance.items():
        for cid in chunk_ids:
            if "<chunk_id" in cid:
                print(f"ERROR: Placeholder chunk_id in relevance.json for {qid}")
                print("Run build_eval_dataset.py to label real chunk IDs first.")
                sys.exit(1)

    labeled = {qid: set(ids) for qid, ids in relevance.items() if ids}
    print(f"Loaded {len(queries)} queries, {len(labeled)} with relevance labels")
    return queries, labeled


def evaluate_mode(
    queries: list[dict],
    relevance: dict[str, set],
    retrieve_fn,
    k: int,
    query_type: str | None = None,
) -> dict:
    p_scores, recall_scores, mrr_scores, ndcg_scores = [], [], [], []

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
            print(f"  [WARN] Retrieval failed for {qid}: {e}")
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
    }


def print_table(results: dict, k: int, title: str = "") -> None:
    col_w = 26
    metric_keys = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    if title:
        print(f"\n{title}")
    header = f"{'Mode':<{col_w}}" + "".join(f"{m:>12}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for mode_name, metrics in results.items():
        row = f"{mode_name:<{col_w}}" + "".join(
            f"{metrics.get(m, 0.0):>12.4f}" for m in metric_keys
        )
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eval-dir", default="evaluation")
    parser.add_argument("--output", help="Explicit path to save results JSON")
    parser.add_argument(
        "--phase",
        type=int,
        help="Phase number — auto-saves to evaluation/results/phaseN.json",
    )
    parser.add_argument(
        "--no-learned",
        action="store_true",
        help="Skip learned fusion (if model not trained yet)",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranker mode",
    )
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

    def sparse_fn(q, k):
        return sparse.search(q, k=k)

    def dense_fn(q, k):
        return dense.search(q, k=k)

    def hybrid_fn(q, k):
        return hybrid.search_full(
            q, k=k, expand=False, web_search=False, rerank=False
        )["results"]

    def hybrid_rerank_fn(q, k):
        return hybrid.search_full(
            q, k=k, expand=False, web_search=False, rerank=True, rerank_top_k=k
        )["results"]

    modes: dict[str, any] = {
        "BM25 (sparse)": sparse_fn,
        "Dense (Qdrant)": dense_fn,
        "Hybrid RRF": hybrid_fn,
    }

    if not args.no_reranker:
        modes["Hybrid RRF + Reranker"] = hybrid_rerank_fn

    # Learned fusion — skip gracefully if model not trained
    if not args.no_learned:
        learned = LearnedHybridFusion(sparse=sparse, dense=dense)
        if learned.is_trained:
            modes["Learned Fusion"] = lambda q, k: learned.search(q, k=k)
        else:
            print("[INFO] Learned fusion model not found — skipping.")
            print("       Run scripts/train_fusion.py to train it.")

    query_types = ["keyword", "semantic", "vague"]

    # ── Overall eval ──────────────────────────────────────────────────────────
    overall: dict[str, dict] = {}
    print(f"\nEvaluating at k={args.k} across {len(relevance)} labeled queries...\n")

    for mode_name, fn in modes.items():
        t0 = time.perf_counter()
        metrics = evaluate_mode(queries, relevance, fn, args.k)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        overall[mode_name] = {**metrics, "eval_latency_ms": elapsed}

    print_table(overall, args.k, title="── Overall ──")

    # ── Per-type breakdown ────────────────────────────────────────────────────
    per_type: dict[str, dict[str, dict]] = {}
    for qtype in query_types:
        type_results: dict[str, dict] = {}
        for mode_name, fn in modes.items():
            metrics = evaluate_mode(queries, relevance, fn, args.k, query_type=qtype)
            if metrics["queries_evaluated"] > 0:
                type_results[mode_name] = metrics
        if type_results:
            per_type[qtype] = type_results
            print_table(type_results, args.k, title=f"── {qtype.capitalize()} queries ──")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif args.phase is not None:
        out_path = Path(args.eval_dir) / "results" / f"phase{args.phase}.json"

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection": args.collection,
            "k": args.k,
            "queries_labeled": len(relevance),
            "overall": overall,
            "per_type": per_type,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Results saved → {out_path}")
    try:
        dense._client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
