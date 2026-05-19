"""
CollectionManager — handles lifecycle of named document collections.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from neural_search.config import settings
from neural_search.retrieval.dense import QdrantRetriever

MAX_COLLECTIONS = 10


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class CollectionManager:
    def __init__(self):
        self._base = settings.data_dir / "collections"
        self._base.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, slug: str) -> Path:
        return self._base / slug / "metadata.json"

    def _read_meta(self, slug: str) -> dict:
        # #14: catch JSONDecodeError — one corrupt file must not break all collections
        path = self._meta_path(slug)
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Corrupt metadata for collection '{slug}' — skipping")
            return {}

    def _write_meta(self, slug: str, meta: dict) -> None:
        path = self._meta_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename — atomic on POSIX, avoids partial writes
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        tmp.replace(path)

    def list_collections(self) -> list[dict]:
        collections = []
        for path in sorted(self._base.iterdir()):
            if path.is_dir():
                meta = self._read_meta(path.name)
                if meta:
                    collections.append(meta)
        return collections

    def get_collection(self, slug: str) -> dict | None:
        meta = self._read_meta(slug)
        return meta if meta else None

    def create_collection(self, name: str, description: str = "") -> dict:
        if len(self.list_collections()) >= MAX_COLLECTIONS:
            raise ValueError(f"Collection limit reached ({MAX_COLLECTIONS}). Delete one first.")
        slug = slugify(name)
        if self.get_collection(slug):
            raise ValueError(f"Collection '{name}' already exists.")
        meta = {
            "slug": slug,
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": [],
            "total_chunks": 0,
            "total_tokens": 0,
        }
        self._write_meta(slug, meta)
        logger.info(f"Collection created: '{name}' (slug={slug})")
        return meta

    def delete_collection(self, slug: str) -> None:
        if not self.get_collection(slug):
            raise ValueError(f"Collection '{slug}' not found.")
        # Wipe Qdrant vector collection (best-effort — it may not exist)
        try:
            QdrantRetriever(collection_slug=slug).reset()
        except Exception as e:
            logger.warning(f"Could not wipe Qdrant collection '{slug}': {e}")
        for path in [
            settings.bm25_path_for(slug),
            settings.documents_path_for(slug),
            self._base / slug,
        ]:
            if path.exists():
                shutil.rmtree(path)
        logger.info(f"Collection deleted: '{slug}'")

    def add_file_record(self, slug: str, record: dict) -> None:
        meta = self._read_meta(slug)
        existing = [f for f in meta["files"] if f["filename"] != record["filename"]]
        existing.append(record)
        meta["files"] = existing
        meta["total_chunks"] = sum(f["chunks"] for f in existing)
        meta["total_tokens"] = sum(f["tokens"] for f in existing)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_meta(slug, meta)

    def file_exists(self, slug: str, filename: str) -> bool:
        meta = self._read_meta(slug)
        return any(f["filename"] == filename for f in meta.get("files", []))
