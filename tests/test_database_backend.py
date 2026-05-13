"""
Tests for SQLiteBackend — schema creation, store, and query.

All tests use in-memory SQLite (":memory:") so they leave no files on disk.
"""

import pytest
from helpers import make_element, make_doc, make_batch, make_bbox
from database_backend import SQLiteBackend, DatabaseConfig, DatabaseQueryFilters
from data_models import ElementType, ProcessingStatus


# ── Helpers ────────────────────────────────────────────────────────────────────

def _backend(**cfg_kwargs) -> SQLiteBackend:
    cfg = DatabaseConfig(db_path=":memory:", **cfg_kwargs)
    return SQLiteBackend(cfg)


def _simple_batch(n_docs=1, n_elems=2):
    docs = []
    for i in range(n_docs):
        elems = [
            make_element(ElementType.TEXT, f"text {i}-{j}", element_id=f"e{i}_{j}")
            for j in range(n_elems)
        ]
        docs.append(make_doc(document_id=f"doc_{i}", elements=elems))
    return make_batch(documents=docs, batch_id="batch_test")


# ── Schema and construction ────────────────────────────────────────────────────

class TestConstruction:
    def test_in_memory_creates_tables(self):
        db = _backend()
        tables = {
            row[0] for row in
            db._get_conn().execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "batches" in tables
        assert "documents" in tables
        assert "elements" in tables
        db.close()

    def test_table_prefix_applied(self):
        db = _backend(table_prefix="ocr_")
        tables = {
            row[0] for row in
            db._get_conn().execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "ocr_batches" in tables
        assert "ocr_elements" in tables
        db.close()

    def test_context_manager_closes_connection(self):
        with _backend() as db:
            db.store(_simple_batch())
        assert db._conn is None


# ── store() ────────────────────────────────────────────────────────────────────

class TestStore:
    def test_store_writes_batch_row(self):
        db = _backend()
        batch = _simple_batch()
        db.store(batch)
        row = db.get_batch_summary("batch_test")
        assert row is not None
        assert row["batch_id"] == "batch_test"
        db.close()

    def test_store_writes_all_elements(self):
        db = _backend()
        db.store(_simple_batch(n_docs=2, n_elems=3))
        assert db.count_elements() == 6
        db.close()

    def test_store_document_writes_elements(self):
        db = _backend()
        doc = make_doc(
            document_id="doc_x",
            elements=[make_element(ElementType.HEADING, "Title", element_id="h1")],
        )
        db.store_document(doc, batch_id="b1")
        assert db.count_elements() == 1
        db.close()

    def test_overwrite_existing_replaces_row(self):
        db = _backend(overwrite_existing=True)
        batch = _simple_batch()
        db.store(batch)
        db.store(batch)  # second store should replace, not duplicate
        assert db.count_elements() == 2  # 1 doc × 2 elems, no duplicates
        db.close()

    def test_no_overwrite_ignores_duplicate(self):
        db = _backend(overwrite_existing=False)
        batch = _simple_batch()
        db.store(batch)
        db.store(batch)
        assert db.count_elements() == 2  # still 2, not 4
        db.close()


# ── query_elements() ───────────────────────────────────────────────────────────

class TestQueryElements:
    def _populated_db(self):
        db = _backend()
        elems = [
            make_element(ElementType.TEXT,    "para",    confidence=0.9, element_id="t1"),
            make_element(ElementType.HEADING, "Title",   confidence=0.95, element_id="h1"),
            make_element(ElementType.TABLE,   "tbl",     confidence=0.5,  element_id="tb1"),
        ]
        doc = make_doc(document_id="doc_q", elements=elems)
        batch = make_batch(documents=[doc], batch_id="bq")
        db.store(batch)
        return db

    def test_query_all_returns_all(self):
        with self._populated_db() as db:
            rows = db.query_elements()
            assert len(rows) == 3

    def test_filter_by_element_type(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(element_type=ElementType.HEADING))
            assert len(rows) == 1
            assert rows[0]["element_type"] == ElementType.HEADING.value

    def test_filter_min_confidence(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(min_confidence=0.9))
            assert all(r["confidence"] >= 0.9 for r in rows)

    def test_filter_max_confidence(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(max_confidence=0.6))
            assert all(r["confidence"] <= 0.6 for r in rows)

    def test_filter_by_batch_id(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(batch_id="bq"))
            assert len(rows) == 3

    def test_filter_by_document_id(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(document_id="doc_q"))
            assert len(rows) == 3

    def test_limit_reduces_results(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(limit=2))
            assert len(rows) == 2

    def test_no_match_returns_empty(self):
        with self._populated_db() as db:
            rows = db.query_elements(DatabaseQueryFilters(batch_id="nonexistent"))
            assert rows == []

    def test_has_parent_false_filters_roots(self):
        db = _backend()
        child = make_element(ElementType.TEXT, "child", element_id="c1", parent_id="h1")
        root  = make_element(ElementType.HEADING, "root", element_id="h1")
        doc = make_doc(document_id="d1", elements=[root, child])
        db.store(make_batch(documents=[doc], batch_id="bp"))
        rows = db.query_elements(DatabaseQueryFilters(has_parent=False))
        assert all(r["parent_id"] is None for r in rows)
        db.close()


# ── get_batch_summary() ────────────────────────────────────────────────────────

class TestBatchSummary:
    def test_returns_none_for_unknown_batch(self):
        with _backend() as db:
            assert db.get_batch_summary("does_not_exist") is None

    def test_returns_dict_for_known_batch(self):
        with _backend() as db:
            db.store(_simple_batch())
            summary = db.get_batch_summary("batch_test")
            assert isinstance(summary, dict)
            assert summary["document_count"] == 1


# ── count_elements() ──────────────────────────────────────────────────────────

class TestCountElements:
    def test_count_all(self):
        with _backend() as db:
            db.store(_simple_batch(n_docs=3, n_elems=4))
            assert db.count_elements() == 12

    def test_count_by_batch(self):
        with _backend() as db:
            db.store(_simple_batch(n_docs=2, n_elems=2))
            assert db.count_elements(batch_id="batch_test") == 4
            assert db.count_elements(batch_id="other") == 0
