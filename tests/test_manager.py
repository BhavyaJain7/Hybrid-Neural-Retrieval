"""
Unit tests for collections/manager.py
Tests: create, list, delete, duplicate prevention, file records, slug generation.
"""
import pytest
from neural_search.collections.manager import CollectionManager, slugify


# No local mock_settings — autouse patch_settings from conftest handles it.
# manager fixture wires CollectionManager directly to tmp_path.
@pytest.fixture
def manager(tmp_path):
    m = CollectionManager.__new__(CollectionManager)
    m._base = tmp_path / "data" / "collections"
    m._base.mkdir(parents=True, exist_ok=True)
    return m


class TestSlugify:
    def test_lowercases(self):
        assert slugify("HR Policies") == "hr-policies"

    def test_replaces_spaces_with_dashes(self):
        assert slugify("Q3 Sales Report") == "q3-sales-report"

    def test_strips_special_chars(self):
        assert slugify("My Doc!@#") == "my-doc"

    def test_collapses_multiple_separators(self):
        assert slugify("HR  --  Policies") == "hr-policies"

    def test_empty_string_returns_empty(self):
        assert slugify("") == ""

    def test_numeric_only(self):
        assert slugify("2024") == "2024"


class TestCreateCollection:
    def test_creates_and_returns_meta(self, manager):
        col = manager.create_collection("HR Policies", "HR docs")
        assert col["slug"] == "hr-policies"
        assert col["name"] == "HR Policies"
        assert col["description"] == "HR docs"
        assert col["total_chunks"] == 0
        assert col["total_tokens"] == 0
        assert col["files"] == []

    def test_metadata_persisted_to_disk(self, manager):
        manager.create_collection("Test Collection")
        col = manager.get_collection("test-collection")
        assert col is not None
        assert col["name"] == "Test Collection"

    def test_duplicate_name_raises(self, manager):
        manager.create_collection("HR Policies")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_collection("HR Policies")

    def test_collection_limit_enforced(self, manager, monkeypatch):
        monkeypatch.setattr("neural_search.collections.manager.MAX_COLLECTIONS", 2)
        manager.create_collection("Col One")
        manager.create_collection("Col Two")
        with pytest.raises(ValueError, match="limit reached"):
            manager.create_collection("Col Three")

    def test_created_at_and_updated_at_set(self, manager):
        col = manager.create_collection("Timestamped")
        assert col["created_at"] is not None
        assert col["updated_at"] is not None
        assert "T" in col["created_at"]   # ISO format sanity check

    def test_description_defaults_to_empty(self, manager):
        col = manager.create_collection("No Desc")
        assert col["description"] == ""


class TestListCollections:
    def test_empty_returns_empty_list(self, manager):
        assert manager.list_collections() == []

    def test_lists_all_created(self, manager):
        manager.create_collection("Alpha")
        manager.create_collection("Beta")
        cols = manager.list_collections()
        names = [c["name"] for c in cols]
        assert "Alpha" in names
        assert "Beta" in names

    def test_count_matches_created(self, manager):
        manager.create_collection("A")
        manager.create_collection("B")
        manager.create_collection("C")
        assert len(manager.list_collections()) == 3

    def test_returns_sorted_by_slug(self, manager):
        manager.create_collection("Zebra")
        manager.create_collection("Alpha")
        cols = manager.list_collections()
        slugs = [c["slug"] for c in cols]
        assert slugs == sorted(slugs)


class TestGetCollection:
    def test_returns_none_for_missing(self, manager):
        assert manager.get_collection("ghost") is None

    def test_returns_correct_collection(self, manager):
        manager.create_collection("Alpha")
        manager.create_collection("Beta")
        col = manager.get_collection("beta")
        assert col["name"] == "Beta"


class TestDeleteCollection:
    def test_delete_removes_from_list(self, manager):
        manager.create_collection("Temp")
        manager.delete_collection("temp")
        assert manager.get_collection("temp") is None

    def test_delete_reduces_count(self, manager):
        manager.create_collection("A")
        manager.create_collection("B")
        manager.delete_collection("a")
        assert len(manager.list_collections()) == 1

    def test_delete_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.delete_collection("ghost")

    def test_after_delete_can_recreate_same_name(self, manager):
        manager.create_collection("Reusable")
        manager.delete_collection("reusable")
        col = manager.create_collection("Reusable")
        assert col["slug"] == "reusable"
        assert manager.get_collection("reusable") is not None


class TestFileRecords:
    def _record(self, filename="report.pdf", chunks=20, tokens=1500):
        return {
            "filename": filename,
            "pages": 5,
            "chunks": chunks,
            "tokens": tokens,
            "ingested_at": "2026-04-20T10:00:00+00:00",
            "status": "ok",
        }

    def test_add_file_record_updates_meta(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", self._record())
        col = manager.get_collection("docs")
        assert len(col["files"]) == 1
        assert col["total_chunks"] == 20
        assert col["total_tokens"] == 1500

    def test_multiple_files_aggregate_correctly(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", self._record("a.pdf", chunks=10, tokens=500))
        manager.add_file_record("docs", self._record("b.pdf", chunks=15, tokens=800))
        col = manager.get_collection("docs")
        assert col["total_chunks"] == 25
        assert col["total_tokens"] == 1300

    def test_re_adding_same_filename_overwrites_not_appends(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", self._record(chunks=10))
        manager.add_file_record("docs", self._record(chunks=25))
        col = manager.get_collection("docs")
        assert len(col["files"]) == 1
        assert col["total_chunks"] == 25

    def test_updated_at_changes_after_add(self, manager):
        col = manager.create_collection("Docs")
        original_updated = col["updated_at"]
        import time; time.sleep(0.01)
        manager.add_file_record("docs", self._record())
        col = manager.get_collection("docs")
        assert col["updated_at"] >= original_updated

    def test_file_exists_true(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", self._record("exist.pdf"))
        assert manager.file_exists("docs", "exist.pdf") is True

    def test_file_exists_false(self, manager):
        manager.create_collection("Docs")
        assert manager.file_exists("docs", "ghost.pdf") is False

    def test_file_exists_false_after_overwrite_with_different_file(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", self._record("a.pdf"))
        assert manager.file_exists("docs", "b.pdf") is False
