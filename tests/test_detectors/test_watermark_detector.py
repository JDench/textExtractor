"""
Tests for WatermarkDetector — vocabulary-match and span-based watermark detection.

Signal 1 (vocabulary) is purely text-based.
Signal 2 (span + pixel) needs a real numpy image.
Both are tested with synthetic inputs — no Tesseract call required.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

cv2 = pytest.importorskip("cv2", reason="cv2 required for WatermarkDetector")

from detectors.watermark_detector import WatermarkDetector, WatermarkDetectorConfig
from data_models import ElementType, Watermark
from helpers import make_ocr


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect(image, ocr_results, **cfg_kwargs):
    cfg = WatermarkDetectorConfig(**cfg_kwargs) if cfg_kwargs else WatermarkDetectorConfig()
    detector = WatermarkDetector(cfg)
    elements, trace = detector.detect(image, ocr_results, page_number=1)
    return elements, trace


def _blank(h=400, w=600):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray_image(value=160, h=400, w=600):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ── Signal 1: vocabulary match ────────────────────────────────────────────────

class TestVocabularyMatch:
    def test_draft_detected(self):
        img = _blank()
        ocr = make_ocr("DRAFT", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, trace = detect(img, [ocr])
        assert len(elems) == 1
        assert trace.vocabulary_matches == 1

    def test_confidential_detected(self):
        img = _blank()
        ocr = make_ocr("CONFIDENTIAL", x_min=10, y_min=10, x_max=210, y_max=40)
        elems, trace = detect(img, [ocr])
        assert len(elems) == 1

    def test_case_insensitive(self):
        img = _blank()
        ocr = make_ocr("draft", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, _ = detect(img, [ocr])
        assert len(elems) == 1

    def test_vocabulary_in_sentence(self):
        img = _blank()
        ocr = make_ocr("Please mark this as DRAFT before sending", x_min=10, y_min=10, x_max=400, y_max=35)
        elems, _ = detect(img, [ocr])
        assert len(elems) == 1

    def test_normal_text_not_detected(self):
        img = _blank()
        ocr = make_ocr("This is a regular paragraph.", x_min=10, y_min=10, x_max=300, y_max=35)
        elems, _ = detect(img, [ocr])
        assert len(elems) == 0

    def test_small_bbox_skipped(self):
        """Bounding box area below min_bbox_area should be skipped."""
        img = _blank()
        # Very small bbox: 5×5 = 25 px² < default min_bbox_area=400
        ocr = make_ocr("DRAFT", x_min=10, y_min=10, x_max=15, y_max=15)
        elems, _ = detect(img, [ocr])
        assert len(elems) == 0

    def test_multiple_vocab_words_single_result(self):
        """Two overlapping OCR results should deduplicate by bbox key."""
        img = _blank()
        ocr1 = make_ocr("DRAFT", x_min=10, y_min=10, x_max=110, y_max=40)
        ocr2 = make_ocr("DRAFT", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, _ = detect(img, [ocr1, ocr2])
        assert len(elems) == 1   # deduplicated


# ── Signal 2: large span + light pixels ──────────────────────────────────────

class TestSpanDetection:
    def test_large_span_light_gray_low_confidence_detected(self):
        # Image 400×600, bbox spans >45% of width (270px), gray pixels (value=180)
        img = _gray_image(value=180)
        # Low confidence, large bbox
        ocr = make_ocr(
            "Background watermark text",
            x_min=10, y_min=100, x_max=400, y_max=140,
            confidence=0.40,
        )
        elems, trace = detect(img, [ocr])
        assert len(elems) == 1
        assert trace.span_detections == 1

    def test_high_confidence_span_not_detected(self):
        img = _gray_image(value=180)
        ocr = make_ocr(
            "Regular wide heading",
            x_min=10, y_min=100, x_max=400, y_max=140,
            confidence=0.95,  # high confidence → not a watermark
        )
        elems, _ = detect(img, [ocr])
        assert len(elems) == 0

    def test_dark_image_span_not_detected(self):
        # Very dark image (value=30) → p10 < threshold → not light ink
        img = np.full((400, 600, 3), 30, dtype=np.uint8)
        ocr = make_ocr(
            "Something wide",
            x_min=10, y_min=100, x_max=400, y_max=140,
            confidence=0.40,
        )
        elems, _ = detect(img, [ocr])
        assert len(elems) == 0


# ── Element structure ─────────────────────────────────────────────────────────

class TestElementStructure:
    def test_element_type(self):
        img = _blank()
        ocr = make_ocr("DRAFT", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, _ = detect(img, [ocr])
        assert elems[0].element_type == ElementType.WATERMARK

    def test_content_is_watermark(self):
        img = _blank()
        ocr = make_ocr("SAMPLE", x_min=10, y_min=10, x_max=210, y_max=40)
        elems, _ = detect(img, [ocr])
        assert isinstance(elems[0].content, Watermark)

    def test_content_text_matches_ocr(self):
        img = _blank()
        ocr = make_ocr("VOID", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, _ = detect(img, [ocr])
        assert elems[0].content.content == "VOID"

    def test_confidence_in_range(self):
        img = _blank()
        ocr = make_ocr("PAID", x_min=10, y_min=10, x_max=110, y_max=40)
        elems, _ = detect(img, [ocr])
        assert 0.0 <= elems[0].confidence <= 1.0
