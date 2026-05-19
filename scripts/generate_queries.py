#!/usr/bin/env python3
"""
scripts/generate_queries.py

Generates synthetic eval queries from ingested chunks using Groq.

Modes:
    --mode auto      Enforces 40% keyword / 30% semantic / 30% vague (default)
    --mode keyword   Generate only keyword queries
    --mode semantic  Generate only semantic queries
    --mode vague     Generate only vague queries

Writes to evaluation/queries.json and evaluation/relevance.json.
Resumes from existing files — safe to re-run.
Press Ctrl+C at any time to stop and save progress with a summary.

Usage:
    # Auto-distributed (default)
    python scripts/generate_queries.py --collection <slug> --target 150

    # Only semantic queries
    python scripts/generate_queries.py --collection <slug> --target 420 --mode semantic

    # Only vague queries to top up
    python scripts/generate_queries.py --collection <slug> --target 500 --mode vague
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from groq import Groq
from neural_search.config import get_settings

settings = get_settings()

# ── Type targets (fractions must sum to 1.0) ─────────────────────────────────
TYPE_TARGETS = {"keyword": 0.40, "semantic": 0.30, "vague": 0.30}

# ── Prompts per query type ────────────────────────────────────────────────────
_PROMPTS: dict[str, str] = {
    "keyword": (
        "You are building a search evaluation dataset.\n"
        "Given the passage below, write ONE short keyword search query (4-8 words) "
        "that a user would type to find this passage.\n"
        "Rules:\n"
        "- Use specific technical terms from the passage\n"
        "- Do NOT write a full question sentence\n"
        "- Do NOT use words like 'what', 'how', 'explain'\n"
        "- Output ONLY the query, nothing else\n\n"
        "Passage:\n{text}"
    ),
    "semantic": (
        "You are building a search evaluation dataset.\n"
        "Given the passage below, write ONE natural language question that this passage answers.\n"
        "Rules:\n"
        "- Paraphrase — avoid copying exact phrases from the passage\n"
        "- Use different vocabulary than the passage where possible\n"
        "- The question must be specific enough to have one clear answer\n"
        "- Output ONLY the question, nothing else\n\n"
        "Passage:\n{text}"
    ),
    "vague": (
        "You are building a search evaluation dataset.\n"
        "Given the passage below, write ONE vague or high-level question that someone "
        "might ask before knowing the details in this passage.\n"
        "Rules:\n"
        "- The question should be open-ended and not use technical jargon from the passage\n"
        "- Avoid any words that appear verbatim in the passage\n"
        "- It should require reading the passage to answer properly\n"
        "- Output ONLY the question, nothing else\n\n"
        "Passage:\n{text}"
    ),
}


def _load_snapshot(collection: str) -> list[dict]:
    snapshot = settings.snapshot_path_for(collection)
    if not snapshot.exists():
        print(f"[ERROR] Snapshot not found: {snapshot}")
        print("Run ingestion first to generate the JSONL snapshot.")
        sys.exit(1)
    chunks = []
    with open(snapshot) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[INFO] Loaded {len(chunks)} chunks from {snapshot}")
    return chunks


def _sample_chunks(chunks: list[dict], target: int, sample_every: int) -> list[dict]:
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (c.get("source_file", ""), c.get("chunk_index", 0)),
    )
    strided = sorted_chunks[::sample_every]
    budget = min(len(strided), int(target * 1.5))
    sampled = random.sample(strided, budget) if len(strided) > budget else strided
    print(f"[INFO] Sampled {len(sampled)} chunks (stride={sample_every}, budget={budget})")
    return sampled


def _assign_type(index: int, total: int) -> str:
    ratio = index / total
    if ratio < TYPE_TARGETS["keyword"]:
        return "keyword"
    elif ratio < TYPE_TARGETS["keyword"] + TYPE_TARGETS["semantic"]:
        return "semantic"
    else:
        return "vague"


def _make_query_id(chunk_id: str, query_type: str) -> str:
    raw = f"{chunk_id}::{query_type}"
    return "q_" + hashlib.md5(raw.encode()).hexdigest()[:10]


def _call_groq(client: Groq, prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=128,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            text = text.strip('"').strip("'").strip()
            return text if text else None
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [WARN] Groq error (attempt {attempt+1}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def _validate_query(query: str, chunk_text: str, query_type: str) -> bool:
    if not query or len(query.split()) < 3:
        return False
    if query_type == "keyword":
        return True
    chunk_lower = chunk_text.lower()
    query_words = query.lower().split()
    for i in range(len(query_words) - 2):
        trigram = " ".join(query_words[i : i + 3])
        if trigram in chunk_lower:
            return False
    return True


def _print_summary(
    status: str,
    all_queries: list[dict],
    new_count: int,
    skipped: int,
    failed: int,
    queries_path: Path,
    relevance_path: Path,
) -> None:
    type_counts: dict[str, int] = {}
    for q in all_queries:
        type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"Status                : {status}")
    print(f"Total queries written : {len(all_queries)}")
    print(f"New this run          : {new_count}")
    print(f"Skipped               : {skipped}")
    print(f"Failed validation     : {failed}")
    print(f"Distribution          : {type_counts}")
    print(f"Queries   -> {queries_path}")
    print(f"Relevance -> {relevance_path}")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Collection slug")
    parser.add_argument("--output", default="evaluation", help="Output directory")
    parser.add_argument(
        "--target",
        type=int,
        default=150,
        help="Target total queries across all runs",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="Stride for chunk sampling",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["auto", "keyword", "semantic", "vague"],
        default="auto",
        help="auto=40/30/30 distribution | keyword/semantic/vague=generate only that type",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    queries_path = out / "queries.json"
    relevance_path = out / "relevance.json"

    existing_queries: list[dict] = []
    existing_relevance: dict = {}
    if queries_path.exists():
        existing_queries = json.loads(queries_path.read_text())
        print(f"[INFO] Found {len(existing_queries)} existing queries — will append")
    if relevance_path.exists():
        existing_relevance = json.loads(relevance_path.read_text())

    existing_ids = {q["id"] for q in existing_queries}

    if not settings.groq_api_key or settings.groq_api_key == "not-set":
        print("[ERROR] GROQ_API_KEY not set in .env")
        sys.exit(1)

    client = Groq(api_key=settings.groq_api_key)
    chunks = _load_snapshot(args.collection)

    if not chunks:
        print("[ERROR] No chunks found. Ingest documents first.")
        sys.exit(1)

    sampled = _sample_chunks(chunks, args.target, args.sample_every)
    random.shuffle(sampled)
    total = len(sampled)

    new_queries: list[dict] = []
    new_relevance: dict = {}
    skipped = 0
    failed = 0

    mode_label = (
        args.mode if args.mode != "auto"
        else "auto (keyword=40% semantic=30% vague=30%)"
    )
    print(f"\n[INFO] Generating up to {total} queries (target: {args.target})")
    print(f"[INFO] Mode: {mode_label}")
    print("[INFO] Press Ctrl+C at any time to stop and save progress.\n")

    def _save_and_summarize(status: str) -> None:
        all_queries = existing_queries + new_queries
        all_relevance = {**existing_relevance, **new_relevance}
        out.mkdir(parents=True, exist_ok=True)
        queries_path.write_text(json.dumps(all_queries, indent=2))
        relevance_path.write_text(json.dumps(all_relevance, indent=2))
        _print_summary(
            status=status,
            all_queries=all_queries,
            new_count=len(new_queries),
            skipped=skipped,
            failed=failed,
            queries_path=queries_path,
            relevance_path=relevance_path,
        )

    try:
        for i, chunk in enumerate(sampled):
            if len(new_queries) + len(existing_queries) >= args.target:
                break

            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "").strip()

            if not chunk_id or not chunk_text or len(chunk_text.split()) < 20:
                skipped += 1
                continue

            query_type = args.mode if args.mode != "auto" else _assign_type(i, total)
            qid = _make_query_id(chunk_id, query_type)

            if qid in existing_ids:
                skipped += 1
                continue

            prompt = _PROMPTS[query_type].format(text=chunk_text[:800])
            query_text = _call_groq(client, prompt)

            if not query_text or not _validate_query(query_text, chunk_text, query_type):
                print(
                    f"  [{i+1}/{total}] FAILED validation"
                    f" — {query_type} from {chunk_id[:16]}"
                )
                failed += 1
                continue

            new_queries.append({"id": qid, "text": query_text, "type": query_type})
            new_relevance[qid] = [chunk_id]
            print(f"  [{i+1}/{total}] [{query_type:8s}] {query_text[:80]}")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping early...")
        _save_and_summarize(status="INTERRUPTED — progress saved")
        sys.exit(0)

    _save_and_summarize(status="DONE")
    print("\nNext step: run scripts/build_eval_dataset.py to fix retrieval bias")


if __name__ == "__main__":
    main()
