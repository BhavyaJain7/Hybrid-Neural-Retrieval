"""
evaluation/runner.py

Per-method retrieval evaluation runner.

This module owns the core evaluation logic so it can be imported and
unit-tested independently. ``scripts/run_eval.py`` is a thin CLI wrapper
around the functions here.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from neural_search.evaluation.dataset import EvalDataset, EvalQuery
from neural_search.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


# Type alias for a retrieval function: (query, k) -> list[dict]
RetrieveFn = Callable[[str, int], list[dict]]


def evaluate_mode(
    dataset: EvalDataset,
    retrieve_fn: RetrieveFn,
    k: int = 5,
    query_type: str | None = None,
) -> dict:
    """
    Evaluate a single retrieval function against labeled queries.

    Args:
        dataset:      Loaded eval dataset (queries + relevance labels).
        retrieve_fn:  Callable ``(query_text, k) -> list[dict]`` where each
                      dict has at minimum a ``chunk_id`` key.
        k:            Cutoff for all rank-based metrics.
        query_type:   If provided, evaluate only queries of this type
                      (``"keyword"``, ``"semantic"``, or ``"vague"``).

    Returns:
        Dict with keys ``P@k``, ``Recall@k``, ``MRR``, ``nDCG@k``,
        ``queries_evaluated``, and ``eval_latency_ms``.
    """
    queries = dataset.labeled_queries()
    if query_type is not None:
        queries = [q for q in queries if q.type == query_type]

    p_scores, recall_scores, mrr_scores, ndcg_scores = [], [], [], []

    t0 = time.perf_counter()
    for query in queries:
        relevant = dataset.get_relevant(query.id)
        if not relevant:
            continue

        results = retrieve_fn(query.text, k)
        result_ids = [r["chunk_id"] for r in results]

        p_scores.append(precision_at_k(result_ids, relevant, k))
        recall_scores.append(recall_at_k(result_ids, relevant, k))
        mrr_scores.append(mrr(result_ids, relevant))
        ndcg_scores.append(ndcg_at_k(result_ids, relevant, k))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    n = len(p_scores)

    return {
        f"P@{k}": round(sum(p_scores) / n, 4) if n else 0.0,
        f"Recall@{k}": round(sum(recall_scores) / n, 4) if n else 0.0,
        "MRR": round(sum(mrr_scores) / n, 4) if n else 0.0,
        f"nDCG@{k}": round(sum(ndcg_scores) / n, 4) if n else 0.0,
        "queries_evaluated": n,
        "eval_latency_ms": elapsed_ms,
    }


def evaluate_all_modes(
    dataset: EvalDataset,
    modes: dict[str, RetrieveFn],
    k: int = 5,
    query_type: str | None = None,
) -> dict[str, dict]:
    """
    Evaluate multiple retrieval modes against the same dataset.

    Args:
        dataset:    Loaded eval dataset.
        modes:      ``{mode_name: retrieve_fn}`` mapping.
        k:          Rank cutoff.
        query_type: Optional filter — pass ``"vague"`` to measure expansion lift.

    Returns:
        ``{mode_name: metrics_dict}`` mapping.
    """
    return {
        name: evaluate_mode(dataset, fn, k=k, query_type=query_type)
        for name, fn in modes.items()
    }


def print_results_table(results: dict[str, dict], k: int) -> None:
    """Pretty-print a comparison table to stdout."""
    metric_keys = [f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}"]
    col_w = 26
    header = f"{'Mode':<{col_w}}" + "".join(f"{m:>12}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for mode_name, metrics in results.items():
        row = f"{mode_name:<{col_w}}" + "".join(
            f"{metrics.get(m, 0.0):>12.4f}" for m in metric_keys
        )
        print(row)
    print()
