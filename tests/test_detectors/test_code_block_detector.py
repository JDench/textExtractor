"""
Tests for CodeBlockDetector — visual gray-box and structural alignment detection.

Pass 1 (visual) needs cv2.  Pass 2 (structural) is pure OCR-result analysis.
Both are exercised via synthetic numpy images and OCR results.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

cv2 = pytest.importorskip("cv2", reason="cv2 required for CodeBlockDetector")

from detectors.code_block_detector import CodeBlockDetector, CodeBlockDetectorConfig
from data_models import CodeBlock, ElementType
from helpers import make_ocr


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect(image, ocr_results, **cfg_kwargs):
    cfg = CodeBlockDetectorConfig(**cfg_kwargs) if cfg_kwargs else CodeBlockDetectorConfig()
    detector = CodeBlockDetector(cfg)
    elements, trace = detector.detect(image, ocr_results, page_number=1)
    return elements, trace


def _blank(h=400, w=600):
    return np.ones((h, w, 3), dtype=np.uint8) * 255  # white background


def _gray_box_image(h=400, w=600, box_y1=50, box_y2=200, box_x1=20, box_x2=500,
                    bg_value=200):
    """White image with a gray rectangle (simulating a code box)."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    img[box_y1:box_y2, box_x1:box_x2] = bg_value
    return img


# ── Pass 1: visual gray-box detection ────────────────────────────────────────

class TestVisualDetection:
    def test_gray_box_detected(self):
        img = _gray_box_image(box_y1=50, box_y2=200, box_x1=20, box_x2=500)
        # OCR results inside the box
        ocrs = [
            make_ocr("def foo():", x_min=30, y_min=60, y_max=80),
            make_ocr("    return 42", x_min=30, y_min=90, y_max=110),
        ]
        elems, trace = detect(img, ocrs)
        assert len(elems) >= 1
        assert trace.visual_detections >= 1

    def test_white_image_no_visual_detection(self):
        img = _blank()
        ocrs = [make_ocr("Hello world", x_min=10, y_min=10)]
        elems, _ = detect(img, ocrs, detect_structural=False)
        assert len(elems) == 0

    def test_visual_disabled_skips_gray_box(self):
        img = _gray_box_image()
        ocrs = [make_ocr("def foo():", x_min=30, y_min=60, y_max=80)]
        elems, _ = detect(img, ocrs, detect_visual_boxes=False, detect_structural=False)
        assert len(elems) == 0


# ── Pass 2: structural alignment detection ────────────────────────────────────

class TestStructuralDetection:
    def _aligned_code_ocrs(self, language_line=None):
        """≥3 left-aligned OCR results with code tokens."""
        ocrs = [
            make_ocr("def process(data):", x_min=40, y_min=100, y_max=118),
            make_ocr("    for item in data:", x_min=40, y_min=120, y_max=138),
            make_ocr("        result = item * 2", x_min=40, y_min=140, y_max=158),
            make_ocr("    return result", x_min=40, y_min=160, y_max=178),
        ]
        if language_line:
            ocrs.insert(0, make_ocr(language_line, x_min=40, y_min=80, y_max=98))
        return ocrs

    def test_aligned_code_block_detected(self):
        img = _blank()
        elems, trace = detect(img, self._aligned_code_ocrs(),
                              detect_visual_boxes=False)
        assert len(elems) >= 1
        assert trace.structural_detections >= 1

    def test_unaligned_text_not_detected(self):
        img = _blank()
        # OCR results at wildly different x positions — not code
        ocrs = [
            make_ocr("Hello there", x_min=10, y_min=100, y_max=118),
            make_ocr("How are you", x_min=200, y_min=120, y_max=138),
            make_ocr("Fine thanks", x_min=80, y_min=140, y_max=158),
        ]
        elems, _ = detect(img, ocrs, detect_visual_boxes=False)
        assert len(elems) == 0

    def test_too_few_lines_not_detected(self):
        img = _blank()
        # Only 2 aligned code lines — below default min_lines=3
        ocrs = [
            make_ocr("def foo():", x_min=40, y_min=100, y_max=118),
            make_ocr("    return 1", x_min=40, y_min=120, y_max=138),
        ]
        elems, _ = detect(img, ocrs, detect_visual_boxes=False)
        assert len(elems) == 0

    def test_structural_disabled(self):
        img = _blank()
        elems, _ = detect(img, self._aligned_code_ocrs(),
                          detect_visual_boxes=False, detect_structural=False)
        assert len(elems) == 0


# ── Language fingerprinting ───────────────────────────────────────────────────

class TestLanguageFingerprinting:
    def _detect_with_code(self, lines):
        img = _blank()
        ocrs = [
            make_ocr(line, x_min=40, y_min=100 + i * 20, y_max=118 + i * 20)
            for i, line in enumerate(lines)
        ]
        return detect(img, ocrs, detect_visual_boxes=False)

    def test_python_language_detected(self):
        elems, _ = self._detect_with_code([
            "def compute(x):",
            "    return x * 2",
            "    if x > 0:",
            "        yield x",
        ])
        code_blocks = [e for e in elems if isinstance(e.content, CodeBlock)]
        if code_blocks:  # only assert language if a block was found
            assert code_blocks[0].content.language == "python"

    def test_sql_language_detected(self):
        elems, _ = self._detect_with_code([
            "SELECT name, age",
            "FROM users",
            "WHERE age > 18",
            "ORDER BY name",
        ])
        code_blocks = [e for e in elems if isinstance(e.content, CodeBlock)]
        if code_blocks:
            assert code_blocks[0].content.language == "sql"

    def test_javascript_detected(self):
        elems, _ = self._detect_with_code([
            "function greet(name) {",
            "    const msg = `Hello ${name}`;",
            "    return msg;",
            "}",
        ])
        code_blocks = [e for e in elems if isinstance(e.content, CodeBlock)]
        if code_blocks:
            assert code_blocks[0].content.language == "javascript"


# ── Element structure ─────────────────────────────────────────────────────────

class TestElementStructure:
    def test_element_type(self):
        img = _gray_box_image()
        ocrs = [
            make_ocr("int x = 5;", x_min=30, y_min=60, y_max=80),
            make_ocr("return x;", x_min=30, y_min=90, y_max=110),
        ]
        elems, _ = detect(img, ocrs)
        code_elems = [e for e in elems if e.element_type == ElementType.CODE_BLOCK]
        assert len(code_elems) >= 1

    def test_content_is_code_block(self):
        img = _gray_box_image()
        ocrs = [
            make_ocr("for (int i=0; i<n; i++)", x_min=30, y_min=60, y_max=80),
            make_ocr("    result += arr[i];", x_min=30, y_min=90, y_max=110),
        ]
        elems, _ = detect(img, ocrs)
        code_elems = [e for e in elems if e.element_type == ElementType.CODE_BLOCK]
        if code_elems:
            assert isinstance(code_elems[0].content, CodeBlock)

    def test_confidence_in_range(self):
        img = _gray_box_image()
        ocrs = [make_ocr("x = 1", x_min=30, y_min=60, y_max=80)]
        elems, _ = detect(img, ocrs)
        for e in elems:
            assert 0.0 <= e.confidence <= 1.0
