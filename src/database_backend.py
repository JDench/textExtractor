"""
SQLite Database Backend

Persists BatchResult / DocumentResult objects to a SQLite database so
results can be queried, filtered, and re-exported without re-running OCR.

Schema (three tables):
  batches   — one row per BatchResult
  documents — one row per DocumentResult
  elements  — one row per StructuralElement

Usage::

    backend = SQLiteBackend(DatabaseConfig(db_path="results.db"))
    backend.store(batch_result)

    rows = backend.query_elements(DatabaseQueryFilters(min_confidence=0.8))
    backend.close()

    # Context manager form
    with SQLiteBackend() as db:
        db.store(batch_result)
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_models import (
    BatchResult,
    DocumentResult,
    ElementType,
    StructuralElement,
)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class DatabaseConfig:
    """
    Configuration for the SQLite backend.

    Attributes:
        db_path: Filesystem path or ":memory:" for in-memory database.
        table_prefix: Prepended to every table name (useful for namespacing).
        overwrite_existing: When True, existing rows with the same primary
            key are replaced; when False they are ignored.
        store_element_metadata: Include the full metadata JSON column on
            the elements table (larger DB, but richer queries).
    """
    db_path: str = ":memory:"
    table_prefix: str = ""
    overwrite_existing: bool = True
    store_element_metadata: bool = True


# ── Query filters ──────────────────────────────────────────────────────────────

@dataclass
class DatabaseQueryFilters:
    """
    Filters passed to :meth:`SQLiteBackend.query_elements`.

    All filters are optional (None = no restriction).
    """
    batch_id: Optional[str] = None
    document_id: Optional[str] = None
    element_type: Optional[ElementType] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    page_number: Optional[int] = None
    has_parent: Optional[bool] = None
    limit: Optional[int] = None


# ── Backend ────────────────────────────────────────────────────────────────────

class SQLiteBackend:
    """
    Stores and queries OCR pipeline results in a SQLite database.

    Args:
        config: DatabaseConfig; in-memory database used when None.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "SQLiteBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, batch: BatchResult) -> None:
        """Persist an entire BatchResult (all documents and elements)."""
        with self._transaction() as cur:
            self._upsert_batch(cur, batch)
            for doc in batch.documents:
                self._upsert_document(cur, doc, batch.batch_id)
                for elem in doc.elements:
                    self._upsert_element(cur, elem, doc.metadata.document_id, batch.batch_id)

    def store_document(self, doc: DocumentResult, batch_id: str = "") -> None:
        """Persist a single DocumentResult and all its elements."""
        with self._transaction() as cur:
            self._upsert_document(cur, doc, batch_id)
            for elem in doc.elements:
                self._upsert_element(cur, elem, doc.metadata.document_id, batch_id)

    def query_elements(
        self,
        filters: Optional[DatabaseQueryFilters] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query elements with optional filters.

        Returns a list of dicts, one per matching element row.
        """
        f = filters or DatabaseQueryFilters()
        t = self._t("elements")

        clauses: List[str] = []
        params: List[Any] = []

        if f.batch_id is not None:
            clauses.append("batch_id = ?"); params.append(f.batch_id)
        if f.document_id is not None:
            clauses.append("document_id = ?"); params.append(f.document_id)
        if f.element_type is not None:
            clauses.append("element_type = ?"); params.append(f.element_type.value)
        if f.min_confidence is not None:
            clauses.append("confidence >= ?"); params.append(f.min_confidence)
        if f.max_confidence is not None:
            clauses.append("confidence <= ?"); params.append(f.max_confidence)
        if f.page_number is not None:
            clauses.append("page_number = ?"); params.append(f.page_number)
        if f.has_parent is True:
            clauses.append("parent_id IS NOT NULL")
        elif f.has_parent is False:
            clauses.append("parent_id IS NULL")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = f"LIMIT {f.limit}" if f.limit is not None else ""
        sql = f"SELECT * FROM {t} {where} ORDER BY document_id, page_number {limit}"

        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.row_factory = None
        return [dict(r) for r in rows]

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Return the batch row as a dict, or None if not found."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT * FROM {self._t('batches')} WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        conn.row_factory = None
        return dict(row) if row else None

    def count_elements(self, batch_id: Optional[str] = None) -> int:
        """Return total element count, optionally scoped to a batch."""
        t = self._t("elements")
        if batch_id:
            row = self._get_conn().execute(
                f"SELECT COUNT(*) FROM {t} WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        else:
            row = self._get_conn().execute(f"SELECT COUNT(*) FROM {t}").fetchone()
        return row[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._transaction() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._t('batches')} (
                    batch_id        TEXT PRIMARY KEY,
                    created_at      TEXT,
                    document_count  INTEGER,
                    total_elements  INTEGER,
                    stored_at       TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._t('documents')} (
                    document_id         TEXT PRIMARY KEY,
                    batch_id            TEXT,
                    source_file         TEXT,
                    processing_status   TEXT,
                    processing_duration REAL,
                    image_width         INTEGER,
                    image_height        INTEGER,
                    detected_language   TEXT,
                    total_elements      INTEGER,
                    average_confidence  REAL,
                    processing_timestamp TEXT
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._t('elements')} (
                    element_id      TEXT PRIMARY KEY,
                    document_id     TEXT,
                    batch_id        TEXT,
                    element_type    TEXT,
                    content_text    TEXT,
                    confidence      REAL,
                    page_number     INTEGER,
                    bbox_x_min      REAL,
                    bbox_y_min      REAL,
                    bbox_x_max      REAL,
                    bbox_y_max      REAL,
                    parent_id       TEXT,
                    child_ids       TEXT,
                    nesting_level   INTEGER,
                    metadata_json   TEXT
                )
            """)

    # ── Insert helpers ─────────────────────────────────────────────────────────

    def _upsert_batch(self, cur: sqlite3.Cursor, batch: BatchResult) -> None:
        mode = "REPLACE" if self.config.overwrite_existing else "INSERT OR IGNORE"
        total_elements = sum(len(d.elements) for d in batch.documents)
        cur.execute(
            f"{mode} INTO {self._t('batches')} VALUES (?,?,?,?,?)",
            (
                batch.batch_id,
                batch.created_at.isoformat() if batch.created_at else None,
                len(batch.documents),
                total_elements,
                datetime.now().isoformat(),
            ),
        )

    def _upsert_document(
        self, cur: sqlite3.Cursor, doc: DocumentResult, batch_id: str
    ) -> None:
        m = doc.metadata
        mode = "REPLACE" if self.config.overwrite_existing else "INSERT OR IGNORE"
        cur.execute(
            f"{mode} INTO {self._t('documents')} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                m.document_id,
                batch_id,
                m.source_file,
                m.processing_status.value if m.processing_status else None,
                m.processing_duration,
                m.image_dimensions[0] if m.image_dimensions else None,
                m.image_dimensions[1] if m.image_dimensions else None,
                m.detected_language,
                m.total_elements_extracted,
                m.average_confidence,
                m.processing_timestamp.isoformat() if m.processing_timestamp else None,
            ),
        )

    def _upsert_element(
        self,
        cur: sqlite3.Cursor,
        elem: StructuralElement,
        document_id: str,
        batch_id: str,
    ) -> None:
        mode = "REPLACE" if self.config.overwrite_existing else "INSERT OR IGNORE"
        content_text = _flatten_content(elem)
        bbox = elem.bbox
        meta_json = (
            json.dumps(elem.metadata) if self.config.store_element_metadata and elem.metadata
            else None
        )
        cur.execute(
            f"{mode} INTO {self._t('elements')} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                elem.element_id,
                document_id,
                batch_id,
                elem.element_type.value if elem.element_type else None,
                content_text,
                elem.confidence,
                elem.page_number,
                bbox.x_min if bbox else None,
                bbox.y_min if bbox else None,
                bbox.x_max if bbox else None,
                bbox.y_max if bbox else None,
                elem.parent_id,
                json.dumps(elem.child_ids) if elem.child_ids else None,
                elem.nesting_level,
                meta_json,
            ),
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _t(self, table: str) -> str:
        return f"{self.config.table_prefix}{table}" if self.config.table_prefix else table

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.config.db_path)
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ── Content flattener ──────────────────────────────────────────────────────────

def _flatten_content(elem: StructuralElement) -> Optional[str]:
    """Extract a plain-text string from any element content type."""
    c = elem.content
    if c is None:
        return None
    if isinstance(c, str):
        return c
    if hasattr(c, "content") and isinstance(c.content, str):
        return c.content
    if hasattr(c, "title"):
        return c.title
    if hasattr(c, "term"):
        return c.term
    return str(c)
