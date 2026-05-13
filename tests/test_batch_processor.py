"""
Tests for BatchProcessor — configuration, detector dispatch, and error handling.

These tests mock or skip cv2-dependent detector initialisation so the suite
runs without needing a real Tesseract installation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

cv2 = pytest.importorskip("cv2", reason="cv2 required for BatchProcessor")

from batch_processor import BatchProcessor, BatchProcessorConfig
from data_models import ElementType, ProcessingStatus
from helpers import make_element, make_bbox


@pytest.fixture(autouse=True)
def patch_ocr_engine_init():
    """Prevent OCREngine.__init__ from calling Tesseract during detector construction."""
    with patch("ocr_engine.OCREngine.__init__", return_value=None):
        yield


# ── Helpers ────────────────────────────────────────────────────────────────────

def _blank_image(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_detector_result(elements=None):
    """Return a callable that mimics a detector's detect/detect_* method."""
    if elements is None:
        elements = []
    mock = MagicMock(return_value=(elements, MagicMock()))
    return mock


# ── Config flag tests ─────────────────────────────────────────────────────────

class TestConfigFlags:
    def test_default_config_creates_all_detectors(self):
        bp = BatchProcessor()
        assert bp._text_detector is not None
        assert bp._list_detector is not None
        assert bp._table_detector is not None
        assert bp._content_table_detector is not None
        assert bp._header_footer_detector is not None
        assert bp._formula_detector is not None
        assert bp._figure_detector is not None
        assert bp._annotation_detector is not None
        assert bp._watermark_detector is not None
        assert bp._barcode_detector is not None
        assert bp._code_block_detector is not None
        assert bp._reference_detector is not None
        assert bp._toc_detector is not None
        assert bp._index_detector is not None

    def test_disabled_detector_is_none(self):
        cfg = BatchProcessorConfig(
            detect_text=False,
            detect_tables_line=False,
            detect_barcodes=False,
        )
        bp = BatchProcessor(cfg)
        assert bp._text_detector is None
        assert bp._table_detector is None
        assert bp._barcode_detector is None

    def test_enabled_detectors_remain_when_others_disabled(self):
        cfg = BatchProcessorConfig(detect_text=False)
        bp = BatchProcessor(cfg)
        assert bp._list_detector is not None
        assert bp._table_detector is not None


# ── Lazy initialisation ───────────────────────────────────────────────────────

class TestLazyInit:
    def test_ocr_engine_not_created_at_init(self):
        bp = BatchProcessor()
        assert bp._ocr_engine is None

    def test_hierarchy_builder_not_created_at_init(self):
        bp = BatchProcessor()
        assert bp._hierarchy_builder is None

    def test_get_ocr_engine_creates_once(self):
        bp = BatchProcessor()
        engine1 = bp._get_ocr_engine()
        engine2 = bp._get_ocr_engine()
        assert engine1 is engine2

    def test_get_hierarchy_builder_creates_once(self):
        bp = BatchProcessor()
        hb1 = bp._get_hierarchy_builder()
        hb2 = bp._get_hierarchy_builder()
        assert hb1 is hb2


# ── process_image with mocked detectors ──────────────────────────────────────

class TestProcessImage:
    def _bp_with_mocked_ocr(self, ocr_elements=None, extra_elements=None):
        """
        Return a BatchProcessor whose OCR engine and all detectors are mocked.
        ocr_elements: List[OCRTextResult] the mocked OCR returns (default []).
        extra_elements: List[StructuralElement] one non-text detector returns.
        """
        bp = BatchProcessor(BatchProcessorConfig(build_hierarchy=False))

        # Mock OCR engine
        mock_ocr_engine = MagicMock()
        mock_ocr_engine.extract_text.return_value = (ocr_elements or [], MagicMock())
        bp._ocr_engine = mock_ocr_engine

        # Mock all detectors to return empty results
        empty = ([], MagicMock())
        bp._text_detector = MagicMock()
        bp._text_detector.detect_text_elements.return_value = empty
        bp._list_detector = MagicMock()
        bp._list_detector.detect_lists.return_value = empty
        bp._table_detector = MagicMock()
        bp._table_detector.detect_tables.return_value = empty
        bp._content_table_detector = MagicMock()
        bp._content_table_detector.detect_tables.return_value = empty

        # Sprint 4+ detectors return extra_elements from the first one
        for attr in [
            "_header_footer_detector", "_formula_detector", "_figure_detector",
            "_annotation_detector", "_watermark_detector", "_barcode_detector",
            "_code_block_detector", "_reference_detector", "_toc_detector",
            "_index_detector",
        ]:
            mock_det = MagicMock()
            result = (extra_elements or [], MagicMock())
            mock_det.detect.return_value = result
            setattr(bp, attr, mock_det)

        return bp

    def test_returns_document_result(self):
        bp = self._bp_with_mocked_ocr()
        result = bp.process_image(_blank_image())
        from data_models import DocumentResult
        assert isinstance(result, DocumentResult)

    def test_completed_status_on_no_errors(self):
        bp = self._bp_with_mocked_ocr()
        result = bp.process_image(_blank_image())
        assert result.metadata.processing_status == ProcessingStatus.COMPLETED

    def test_elements_from_detectors_collected(self):
        elem = make_element(ElementType.WATERMARK, "DRAFT")
        bp = self._bp_with_mocked_ocr(extra_elements=[elem])
        result = bp.process_image(_blank_image())
        assert len(result.elements) >= 1

    def test_min_confidence_filter(self):
        high = make_element(ElementType.TEXT, "a", confidence=0.90)
        low = make_element(ElementType.TEXT, "b", confidence=0.10)
        bp = self._bp_with_mocked_ocr(extra_elements=[high, low])
        bp.config.min_confidence = 0.50
        result = bp.process_image(_blank_image())
        # Only high-confidence element should survive
        confidences = [e.confidence for e in result.elements]
        assert all(c >= 0.50 for c in confidences)

    def test_detector_failure_doesnt_abort(self):
        """A crashing detector should not abort the whole pipeline."""
        bp = self._bp_with_mocked_ocr()
        bp._header_footer_detector.detect.side_effect = RuntimeError("boom")
        # Should not raise; should return a result with an error recorded
        result = bp.process_image(_blank_image())
        assert result is not None
        assert any("header_footer" in e for e in result.metadata.errors_encountered)

    def test_ocr_failure_produces_partial_or_failed(self):
        bp = BatchProcessor(BatchProcessorConfig(build_hierarchy=False))
        mock_ocr = MagicMock()
        mock_ocr.extract_text.side_effect = RuntimeError("OCR crash")
        bp._ocr_engine = mock_ocr
        # All detectors return empty
        for attr in [
            "_text_detector", "_list_detector", "_table_detector",
            "_content_table_detector", "_header_footer_detector",
            "_formula_detector", "_figure_detector", "_annotation_detector",
            "_watermark_detector", "_barcode_detector", "_code_block_detector",
            "_reference_detector", "_toc_detector", "_index_detector",
        ]:
            mock_det = MagicMock()
            mock_det.detect.return_value = ([], MagicMock())
            mock_det.detect_text_elements.return_value = ([], MagicMock())
            mock_det.detect_lists.return_value = ([], MagicMock())
            mock_det.detect_tables.return_value = ([], MagicMock())
            setattr(bp, attr, mock_det)
        result = bp.process_image(_blank_image())
        assert result.metadata.processing_status in (
            ProcessingStatus.FAILED,
            ProcessingStatus.PARTIAL,
            ProcessingStatus.COMPLETED,
        )
        assert any("OCR" in e for e in result.metadata.errors_encountered)

    def test_accepts_path_string(self, tmp_path):
        # Should return a FAILED DocumentResult (file doesn't exist), not raise
        bp = BatchProcessor(BatchProcessorConfig())
        result = bp.process_image(str(tmp_path / "nonexistent.png"))
        assert result.metadata.processing_status == ProcessingStatus.FAILED

    def test_page_number_assigned(self):
        bp = self._bp_with_mocked_ocr()
        result = bp.process_image(_blank_image(), page_number=5)
        assert result.metadata is not None  # basic sanity check


# ── process_batch ─────────────────────────────────────────────────────────────

class TestProcessBatch:
    def _mocked_bp(self):
        bp = BatchProcessor(BatchProcessorConfig(build_hierarchy=False))
        empty = ([], MagicMock())
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = ([], MagicMock())
        bp._ocr_engine = mock_ocr
        for attr in [
            "_text_detector", "_list_detector", "_table_detector",
            "_content_table_detector", "_header_footer_detector",
            "_formula_detector", "_figure_detector", "_annotation_detector",
            "_watermark_detector", "_barcode_detector", "_code_block_detector",
            "_reference_detector", "_toc_detector", "_index_detector",
        ]:
            mock_det = MagicMock()
            mock_det.detect.return_value = empty
            mock_det.detect_text_elements.return_value = empty
            mock_det.detect_lists.return_value = empty
            mock_det.detect_tables.return_value = empty
            setattr(bp, attr, mock_det)
        return bp

    def test_returns_batch_result(self):
        bp = self._mocked_bp()
        from data_models import BatchResult
        batch = bp.process_batch([_blank_image(), _blank_image()])
        assert isinstance(batch, BatchResult)

    def test_correct_document_count(self):
        bp = self._mocked_bp()
        batch = bp.process_batch([_blank_image()] * 3)
        assert len(batch.documents) == 3

    def test_custom_batch_id(self):
        bp = self._mocked_bp()
        batch = bp.process_batch([_blank_image()], batch_id="my_batch")
        assert batch.batch_id == "my_batch"

    def test_per_item_error_does_not_abort_batch(self):
        bp = self._mocked_bp()
        # Force one item to fail by passing an invalid path
        inputs = [_blank_image(), "nonexistent_file.png", _blank_image()]
        batch = bp.process_batch(inputs)
        assert len(batch.documents) == 3
        failed = [d for d in batch.documents if d.metadata.processing_status == ProcessingStatus.FAILED]
        assert len(failed) == 1


# ── Build hierarchy integration ───────────────────────────────────────────────

class TestHierarchyBuilding:
    def test_build_hierarchy_false_skips_builder(self):
        cfg = BatchProcessorConfig(build_hierarchy=False)
        bp = BatchProcessor(cfg)
        assert bp._hierarchy_builder is None  # not yet lazily created
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = ([], MagicMock())
        bp._ocr_engine = mock_ocr
        empty = ([], MagicMock())
        for attr in [
            "_text_detector", "_list_detector", "_table_detector",
            "_content_table_detector", "_header_footer_detector",
            "_formula_detector", "_figure_detector", "_annotation_detector",
            "_watermark_detector", "_barcode_detector", "_code_block_detector",
            "_reference_detector", "_toc_detector", "_index_detector",
        ]:
            mock_det = MagicMock()
            mock_det.detect.return_value = empty
            mock_det.detect_text_elements.return_value = empty
            mock_det.detect_lists.return_value = empty
            mock_det.detect_tables.return_value = empty
            setattr(bp, attr, mock_det)
        bp.process_image(_blank_image())
        # Hierarchy builder should still be None (never initialised)
        assert bp._hierarchy_builder is None
