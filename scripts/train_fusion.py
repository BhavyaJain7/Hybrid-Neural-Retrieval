"""
scripts/train_fusion.py

Trains the Logistic Regression hybrid fusion model on labeled eval data.
Must be run with the API stopped (Qdrant lock constraint).

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --collection base --k 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_search.evaluation.dataset import load_dataset
from neural_search.retrieval.dense import QdrantRetriever
from neural_search.retrieval.learned import LearnedFusionTrainer
from neural_search.retrieval.sparse import BM25sRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="base")
    parser.add_argument("--k", type=int, default=20, help="Candidate pool size per query")
    parser.add_argument("--queries", default="evaluation/queries.json")
    parser.add_argument("--relevance", default="evaluation/relevance.json")
    args = parser.parse_args()

    print(f"\nLoading eval dataset...")
    dataset = load_dataset(queries_path=args.queries, relevance_path=args.relevance)
    print(f"Coverage: {dataset.coverage}")

    if not dataset.labeled_queries():
        print("ERROR: No labeled queries. Run label_relevance.py first.")
        sys.exit(1)

    print(f"Loading retrievers for collection: '{args.collection}'...")
    sparse = BM25sRetriever(collection_slug=args.collection)
    if not sparse.load():
        print(f"ERROR: BM25 index not found. Run ingestion first.")
        sys.exit(1)

    dense = QdrantRetriever(collection_slug=args.collection)

    print(f"Training learned fusion model (k={args.k})...")
    trainer = LearnedFusionTrainer(dataset=dataset, sparse=sparse, dense=dense)
    fusion = trainer.train(collection=args.collection, k=args.k)

    print(f"\n✓ Model trained and saved.")
    print(f"  Model ready: {fusion.is_trained}")
    print(f"\nNext: run eval to compare learned vs RRF:")
    print(f"  python scripts/run_eval.py --collection {args.collection}")

    dense._client.close()


if __name__ == "__main__":
    main()
