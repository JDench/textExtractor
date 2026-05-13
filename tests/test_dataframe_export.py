"""
Tests for DataFrameExporter and to_dataframe() convenience methods.

pandas is required. Tests are skipped if pandas is not installed.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pd = pytest.importorskip("pandas", reason="pandas required for DataFrame export tests")

from exporters import DataFrameExporter, ExporterConfig
from data_models import ElementType
from helpers import make_batch, make_doc, make_element


# ── Helpers ────────────────────────────────────────────────────────────────────

def _text(t: str, page: int = 1):
    return make_element(ElementType.TEXT, t, page_number=page)


def _heading(t: str):
    return make_element(ElementType.HEADING, t)


# ── DataFrameExporter.to_dataframe() ──────────────────────────────────────────

class TestToDataframe:
    def test_returns_dataframe(self):
        batch = make_batch([[_text("hello")]])
        df = DataFrameExporter().to_dataframe(batch)
        assert isinstance(df, pd.DataFrame)

    def test_empty_batch_returns_empty_dataframe(self):
        batch = make_batch([[]])
        df = DataFrameExporter().to_dataframe(batch)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_row_count_equals_element_count(self):
        elems = [_text("a"), _text("b"), _heading("Title")]
        batch = make_batch([elems])
        df = DataFrameExporter().to_dataframe(batch)
        assert len(df) == 3

    def test_multi_doc_all_rows_present(self):
        batch = make_batch([[_text("doc1")], [_text("doc2a"), _text("doc2b")]])
        df = DataFrameExporter().to_dataframe(batch)
        assert len(df) == 3

    def test_required_columns_present(self):
        batch = make_batch([[_text("x")]])
        df = DataFrameExporter().to_dataframe(batch)
        required = {
            "batch_id", "document_id", "element_id", "element_type",
            "page", "confidence", "content_text",
            "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
        }
        assert required.issubset(set(df.columns))

    def test_content_text_correct(self):
        batch = make_batch([[_text("hello world")]])
        df = DataFrameExporter().to_dataframe(batch)
        assert df.iloc[0]["content_text"] == "hello world"

    def test_element_type_column_is_string_value(self):
        batch = make_batch([[_text("x")]])
        df = DataFrameExporter().to_dataframe(batch)
        assert df.iloc[0]["element_type"] == "text"

    def test_type_columns_present_when_enabled(self):
        batch = make_batch([[_text("x")]])
        df = DataFrameExporter(ExporterConfig(include_type_columns=True)).to_dataframe(batch)
        assert "table_rows" in df.columns
        assert "formula_latex" in df.columns

    def test_type_columns_absent_when_disabled(self):
        batch = make_batch([[_text("x")]])
        df = DataFrameExporter(ExporterConfig(include_type_columns=False)).to_dataframe(batch)
        assert "table_rows" not in df.columns


# ── DataFrameExporter.export() → pickle ───────────────────────────────────────

class TestPickleExport:
    def test_export_creates_pkl_file(self, tmp_path):
        path = tmp_path / "out.pkl"
        batch = make_batch([[_text("hello")]])
        DataFrameExporter().export(batch, path)
        assert path.exists()

    def test_export_returns_dataframe(self, tmp_path):
        path = tmp_path / "out.pkl"
        batch = make_batch([[_text("hello")]])
        df = DataFrameExporter().export(batch, path)
        assert isinstance(df, pd.DataFrame)

    def test_pkl_round_trip(self, tmp_path):
        path = tmp_path / "out.pkl"
        batch = make_batch([[_text("a"), _text("b")]])
        df_written = DataFrameExporter().export(batch, path)
        df_loaded = pd.read_pickle(path)
        assert len(df_written) == len(df_loaded)
        assert list(df_written.columns) == list(df_loaded.columns)

    def test_export_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "out.pkl"
        batch = make_batch([[_text("x")]])
        DataFrameExporter().export(batch, path)
        assert path.exists()


# ── DataFrameExporter.export_document() ───────────────────────────────────────

class TestExportDocument:
    def test_export_document_no_path_returns_dataframe(self):
        doc = make_doc([_text("hello")])
        df = DataFrameExporter().export_document(doc)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_export_document_with_path_saves_pkl(self, tmp_path):
        path = tmp_path / "doc.pkl"
        doc = make_doc([_text("hello")])
        df = DataFrameExporter().export_document(doc, path)
        assert path.exists()
        assert isinstance(df, pd.DataFrame)


# ── BatchResult.to_dataframe() convenience method ─────────────────────────────

class TestBatchResultToDataframe:
    def test_batch_to_dataframe(self):
        batch = make_batch([[_text("x"), _text("y")]])
        df = batch.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_document_to_dataframe(self):
        doc = make_doc([_text("hello"), _heading("Title")])
        df = doc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


# ── Missing pandas ─────────────────────────────────────────────────────────────

class TestNoPandas:
    def test_raises_import_error_when_pandas_unavailable(self):
        import exporters as exp
        original = exp._PANDAS_AVAILABLE
        exp._PANDAS_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="pandas"):
                DataFrameExporter()
        finally:
            exp._PANDAS_AVAILABLE = original
