"""
scripts/label_relevance.py

Bootstrap relevance labels by:
1. Running each query against the hybrid retriever
2. Printing top-K results with chunk text for human review
3. Writing accepted chunk_ids into evaluation/relevance.json

Usage:
    python scripts/label_relevance.py                     # interactive mode
    python scripts/label_relevance.py --auto --k 5        # auto-accept top-3 per query (for smoke testing only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
QUERIES_PATH = Path("evaluation/queries.json")
RELEVANCE_PATH = Path("evaluation/relevance.json")


def fetch_results(query: str, collection: str, k: int, mode: str = "hybrid") -> list[dict]:
    resp = requests.post(
        f"{API_BASE}/search",
        json={"query": query, "collection": collection, "k": k, "mode": mode},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def interactive_label(queries: list[dict], relevance: dict, collection: str, k: int) -> dict:
    unlabeled = [q for q in queries if not relevance.get(q["id"])]
    print(f"\n{len(unlabeled)} queries to label. Commands: y=relevant  n=skip  q=quit\n")

    for q in unlabeled:
        print(f"\n{'='*60}")
        print(f"[{q['id']}] ({q.get('type','?')}) {q['text']}")
        print(f"{'='*60}")

        try:
            results = fetch_results(q["text"], collection, k)
        except Exception as e:
            print(f"  ERROR fetching results: {e}")
            continue

        accepted = []
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] chunk_id: {r['chunk_id']}")
            print(f"       source: {r['source_file']}  page: {r['page']}  score: {r.get('rrf_score', r.get('score', '?')):.4f}")
            print(f"       {r['text'][:200].strip()}...")
            ans = input("  Relevant? (y/n/q): ").strip().lower()
            if ans == "q":
                print("Quitting — saving progress.")
                relevance[q["id"]] = accepted
                return relevance
            if ans == "y":
                accepted.append(r["chunk_id"])

        relevance[q["id"]] = accepted
        print(f"  Saved {len(accepted)} relevant chunks for {q['id']}")

        # Save after every query in case of interruption
        _save(relevance)

    return relevance


def auto_label(queries: list[dict], relevance: dict, collection: str, k: int, top_n: int = 3) -> dict:
    """Auto-accept top-N results per query. Use only for smoke testing."""
    print(f"AUTO MODE: accepting top-{top_n} results per query (not for real evaluation)\n")
    for q in queries:
        if relevance.get(q["id"]):
            continue
        try:
            results = fetch_results(q["text"], collection, k)
            relevance[q["id"]] = [r["chunk_id"] for r in results[:top_n]]
            print(f"  {q['id']}: auto-labeled {len(relevance[q['id']])} chunks")
        except Exception as e:
            print(f"  {q['id']}: ERROR — {e}")
    return relevance


def _save(relevance: dict) -> None:
    with RELEVANCE_PATH.open("w") as f:
        json.dump(relevance, f, indent=2)
    print(f"  [saved → {RELEVANCE_PATH}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="base", help="Collection slug to search against")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve per query")
    parser.add_argument("--auto", action="store_true", help="Auto-accept top-3 results (smoke test only)")
    parser.add_argument("--auto-top-n", type=int, default=3)
    args = parser.parse_args()

    if not QUERIES_PATH.exists():
        print(f"ERROR: {QUERIES_PATH} not found")
        sys.exit(1)

    with QUERIES_PATH.open() as f:
        queries = json.load(f)

    if RELEVANCE_PATH.exists():
        with RELEVANCE_PATH.open() as f:
            raw = json.load(f)
        relevance = {k: v for k, v in raw.items() if not k.startswith("_")}
    else:
        relevance = {}

    if args.auto:
        relevance = auto_label(queries, relevance, args.collection, args.k, args.auto_top_n)
    else:
        relevance = interactive_label(queries, relevance, args.collection, args.k)

    _save(relevance)
    labeled = sum(1 for v in relevance.values() if v)
    print(f"\nDone. {labeled}/{len(queries)} queries labeled.")


if __name__ == "__main__":
    main()
