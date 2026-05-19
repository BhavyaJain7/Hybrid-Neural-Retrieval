"""
Integration Test 4: CollectionManager Lifecycle
================================================

What this tests
---------------
The full CRUD lifecycle of a named collection using the real filesystem:

  create → get → list → add_file_record → file_exists → delete

Real components:
  - CollectionManager (JSON metadata written/read on real disk)
  - Real tmp filesystem (pytest tmp_path via real_settings conftest)

These tests verify:
  1. create_collection() writes a valid metadata.json
  2. get_collection() returns the correct metadata dict
  3. list_collections() includes the newly created collection
  4. duplicate create raises ValueError (not 500)
  5. add_file_record() updates total_chunks / total_tokens
  6. file_exists() returns True after add_file_record, False before
  7. delete_collection() removes the collection directory from disk
  8. get_collection() returns None after delete
  9. Collection limit (MAX_COLLECTIONS=10) is enforced
  10. Corrupt metadata.json does not crash list_collections
"""
import json
import pytest
from pathlib import Path


class TestCollectionLifecycle:
    """Full CRUD over the real filesystem with no mocks."""

    @pytest.fixture
    def manager(self, real_settings):
        from neural_search.collections.manager import CollectionManager
        # Point at real tmp dir already configured by real_settings
        return CollectionManager()

    def test_create_returns_valid_meta(self, manager):
        meta = manager.create_collection("My Docs", "test description")
        assert meta["slug"] == "my-docs"
        assert meta["name"] == "My Docs"
        assert meta["description"] == "test description"
        assert meta["total_chunks"] == 0
        assert meta["files"] == []

    def test_metadata_file_written_to_disk(self, manager, real_settings):
        manager.create_collection("Disk Test", "")
        meta_file = real_settings.data_dir / "collections" / "disk-test" / "metadata.json"
        assert meta_file.exists(), "metadata.json was not written to disk"
        data = json.loads(meta_file.read_text())
        assert data["slug"] == "disk-test"

    def test_get_returns_created_collection(self, manager):
        manager.create_collection("Get Test", "")
        col = manager.get_collection("get-test")
        assert col is not None
        assert col["slug"] == "get-test"

    def test_get_nonexistent_returns_none(self, manager):
        assert manager.get_collection("does-not-exist") is None

    def test_list_includes_created_collection(self, manager):
        manager.create_collection("Listed", "")
        cols = manager.list_collections()
        slugs = [c["slug"] for c in cols]
        assert "listed" in slugs

    def test_duplicate_create_raises_value_error(self, manager):
        manager.create_collection("Dup", "")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_collection("Dup", "")

    def test_add_file_record_updates_totals(self, manager):
        manager.create_collection("Records", "")
        manager.add_file_record("records", {
            "filename": "report.pdf",
            "pages": 5,
            "chunks": 20,
            "tokens": 1500,
            "ingested_at": "2026-01-01T00:00:00+00:00",
            "status": "ok",
        })
        col = manager.get_collection("records")
        assert col["total_chunks"] == 20
        assert col["total_tokens"] == 1500
        assert len(col["files"]) == 1

    def test_file_exists_true_after_add(self, manager):
        manager.create_collection("Existence", "")
        manager.add_file_record("existence", {
            "filename": "doc.pdf", "pages": 1,
            "chunks": 1, "tokens": 100,
            "ingested_at": "2026-01-01T00:00:00+00:00", "status": "ok",
        })
        assert manager.file_exists("existence", "doc.pdf") is True

    def test_file_exists_false_before_add(self, manager):
        manager.create_collection("NoFile", "")
        assert manager.file_exists("nofile", "ghost.pdf") is False

    def test_re_ingest_replaces_file_record(self, manager):
        """Re-adding the same filename must not duplicate the entry."""
        manager.create_collection("Reingest", "")
        record = {
            "filename": "same.pdf", "pages": 2,
            "chunks": 10, "tokens": 500,
            "ingested_at": "2026-01-01T00:00:00+00:00", "status": "ok",
        }
        manager.add_file_record("reingest", record)
        # Re-ingest with updated chunk count
        record2 = {**record, "chunks": 15, "tokens": 750}
        manager.add_file_record("reingest", record2)

        col = manager.get_collection("reingest")
        assert len(col["files"]) == 1, "Duplicate file record should be replaced, not appended"
        assert col["total_chunks"] == 15

    def test_delete_removes_directory(self, manager, real_settings):
        manager.create_collection("ToDelete", "")
        col_dir = real_settings.data_dir / "collections" / "todelete"
        assert col_dir.exists()

        from neural_search.retrieval.dense import QdrantRetriever
        # We don't have Qdrant wired here so just test manager.delete_collection directly
        manager.delete_collection("todelete")
        assert not col_dir.exists(), "Collection directory must be removed on delete"

    def test_get_after_delete_returns_none(self, manager):
        manager.create_collection("Gone", "")
        manager.delete_collection("gone")
        assert manager.get_collection("gone") is None

    def test_collection_limit_enforced(self, manager):
        from neural_search.collections.manager import MAX_COLLECTIONS
        for i in range(MAX_COLLECTIONS):
            manager.create_collection(f"Col {i}", "")
        with pytest.raises(ValueError, match="limit"):
            manager.create_collection("One Too Many", "")

    def test_corrupt_metadata_does_not_crash_list(self, manager, real_settings):
        """One corrupt file must be skipped; other collections must still list."""
        manager.create_collection("Good Col", "")

        # Manually corrupt another collection's metadata
        corrupt_dir = real_settings.data_dir / "collections" / "corrupt"
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "metadata.json").write_text("{not valid json}")

        cols = manager.list_collections()  # must not raise
        slugs = [c["slug"] for c in cols]
        assert "good-col" in slugs, "Valid collection must survive corrupt neighbor"
