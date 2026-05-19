import pickle
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.chunker import Chunk

try:
    import bm25s
except ImportError as e:
    raise ImportError(
        "bm25s is required for sparse retrieval. Run: pip install bm25s"
    ) from e

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = list(stopwords.words("english"))
except Exception:
    logger.warning("NLTK stopwords unavailable — proceeding without stopword filtering")
    _STOPWORDS = []


class BM25sRetriever:
    def __init__(self, collection_slug: str):
        self._slug = collection_slug
        self._index = None
        self._chunks: list[Chunk] = []
        self._dir = settings.bm25_path_for(collection_slug)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._dir / "bm25.pkl"
        self._chunks_file = self._dir / "chunks.pkl"

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return [
            [t for t in text.lower().split() if t.isalpha() and t not in _STOPWORDS]
            for text in texts
        ]

    def index(self, chunks: list[Chunk]) -> None:
        """Rebuild the index from scratch with exactly `chunks` (used for full re-index / reset)."""
        self._chunks = chunks
        self._rebuild_and_persist()
        logger.info(f"[{self._slug}] BM25 index built — {len(chunks)} chunks")

    def add(self, new_chunks: list[Chunk]) -> None:
        """Merge `new_chunks` into the existing corpus and rebuild.
        
        Existing chunks that share a source_file with any chunk in `new_chunks` are
        replaced so re-ingesting a file doesn't create duplicate entries.
        """
        if not self._chunks:
            self.load()  # pull persisted state if not already in memory
        existing_sources = {c.source_file for c in new_chunks}
        retained = [c for c in self._chunks if c.source_file not in existing_sources]
        self._chunks = retained + new_chunks
        self._rebuild_and_persist()
        logger.info(
            f"[{self._slug}] BM25 index updated — {len(self._chunks)} chunks total "
            f"(+{len(new_chunks)} new)"
        )

    def _rebuild_and_persist(self) -> None:
        corpus_tokens = self._tokenize([c.text for c in self._chunks])
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens)
        with open(self._index_file, "wb") as f:
            pickle.dump(self._index, f)
        with open(self._chunks_file, "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> bool:
        if self._index_file.exists() and self._chunks_file.exists():
            with open(self._index_file, "rb") as f:
                self._index = pickle.load(f)
            with open(self._chunks_file, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info(f"[{self._slug}] BM25 index loaded — {len(self._chunks)} chunks")
            return True
        return False

    def search(self, query: str, k: int = None) -> list[dict]:
        if self._index is None:
            self.load()
        if not self._chunks:
            return []
        k = min(k or settings.top_k, len(self._chunks))
        results, scores = self._index.retrieve(
            bm25s.tokenize(query, stopwords=_STOPWORDS), corpus=None, k=k
        )
        raw_scores = scores[0]
        max_score = float(max(raw_scores)) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0
        return [
            {
                "chunk_id": self._chunks[idx].chunk_id,
                "score": float(score) / max_score,
                "rank": rank,
                "source": "sparse",
                "text": self._chunks[idx].text,
                "source_file": self._chunks[idx].source_file,
                "page": self._chunks[idx].page,
                "token_count": self._chunks[idx].token_count,
                "collection": self._slug,
            }
            for rank, (idx, score) in enumerate(zip(results[0], raw_scores), start=1)
        ]

    def count(self) -> int:
        if not self._chunks:
            self.load()
        return len(self._chunks)

    def reset(self) -> None:
        self._index = None
        self._chunks = []
        for f in [self._index_file, self._chunks_file]:
            if f.exists():
                f.unlink()
        logger.info(f"[{self._slug}] BM25 index wiped")
