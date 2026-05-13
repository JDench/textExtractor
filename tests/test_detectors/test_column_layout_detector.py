"""
Tests for ColumnLayoutDetector and sort_elements_by_column_order.

All tests use synthetic OCR results with explicit bounding boxes.
No image processing or cv2 needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import importlib.util

import pytest

# Import directly from module file to avoid triggering detectors/__init__.py (cv2 dep).
# Register in sys.modules so @dataclass can resolve cls.__module__.
_mod_path = Path(__file__).parent.parent.parent / "src" / "detectors" / "column_layout_detector.py"
_spec = importlib.util.spec_from_file_location("column_layout_detector", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["column_layout_detector"] = _mod
_spec.loader.exec_module(_mod)

ColumnLayoutConfig = _mod.ColumnLayoutConfig
ColumnLayoutDetector = _mod.ColumnLayoutDetector
ColumnLayoutResult = _mod.ColumnLayoutResult
SINGLE_COLUMN = _mod.SINGLE_COLUMN
sort_elements_by_column_order = _mod.sort_elements_by_column_order

from data_models import BoundingBox, ElementType, OCRTextResult
from helpers import make_element, make_ocr


# ── Helpers ────────────────────────────────────────────────────────────────────

PAGE_W = 1000.0
PAGE_H = 1400.0

# Left column occupies x: 50–450; right column: 550–950; gap: 450–550
COL_GAP_START = 450.0
COL_GAP_END = 550.0


def _left_ocr(n: int = 12, y_start: float = 50.0) -> list:
    """OCR results filling the left column."""
    results = []
    for i in range(n):
        y = y_start + i * 30
        results.append(make_ocr(f"left word {i}", x_min=50, y_min=y, x_max=440, y_max=y + 20))
    return results


def _right_ocr(n: int = 12, y_start: float = 50.0) -> list:
    """OCR results filling the right column."""
    results = []
    for i in range(n):
        y = y_start + i * 30
        results.append(make_ocr(f"right word {i}", x_min=560, y_min=y, x_max=950, y_max=y + 20))
    return results


def _single_column_ocr(n: int = 12) -> list:
    """OCR results spanning the full page width — no column gap."""
    results = []
    for i in range(n):
        y = 50 + i * 30
        results.append(make_ocr(f"text {i}", x_min=50, y_min=y, x_max=950, y_max=y + 20))
    return results


def detect(ocr_results, page_w=PAGE_W, page_h=PAGE_H, **cfg_kwargs):
    cfg = ColumnLayoutConfig(**cfg_kwargs) if cfg_kwargs else ColumnLayoutConfig()
    return ColumnLayoutDetector(cfg).detect(ocr_results, page_w, page_h)


# ── Single-column pages ────────────────────────────────────────────────────────

class TestSingleColumn:
    def test_single_column_not_flagged(self):
        result = detect(_single_column_ocr())
        assert not result.is_multi_column

    def test_single_column_num_columns_is_1(self):
        result = detect(_single_column_ocr())
        assert result.num_columns == 1

    def test_empty_ocr_returns_single_column(self):
        result = detect([])
        assert not result.is_multi_column

    def test_too_few_results_skips_detection(self):
        # min_ocr_results_for_detection defaults to 8; provide only 3
        result = detect(_single_column_ocr(n=3))
        assert not result.is_multi_column

    def test_zero_page_dimensions_returns_single(self):
        result = detect(_single_column_ocr(), page_w=0, page_h=PAGE_H)
        assert not result.is_multi_column


# ── Two-column pages ───────────────────────────────────────────────────────────

class TestTwoColumns:
    def test_two_column_detected(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert result.is_multi_column

    def test_two_column_num_columns(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert result.num_columns == 2

    def test_two_column_confidence_above_threshold(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert result.confidence >= 0.50

    def test_two_column_regions_returned(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert len(result.columns) == 2

    def test_gap_regions_returned(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert len(result.gap_regions) >= 1

    def test_left_column_x_range(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        left_start, left_end = result.columns[0]
        # Left column should start near page left edge
        assert left_start < 100
        # Left column should end before the gap
        assert left_end < COL_GAP_END

    def test_right_column_x_range(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        right_start, right_end = result.columns[1]
        # Right column should start after the gap
        assert right_start > COL_GAP_START
        # Right column should end near page right edge
        assert right_end > 800


# ── Gap requirements ───────────────────────────────────────────────────────────

class TestGapRequirements:
    def test_narrow_gap_not_detected_as_column(self):
        # Create a very narrow gap (2% of page width = 20px) with tight threshold
        ocr = []
        for i in range(10):
            y = 50 + i * 30
            # Left column up to x=490, right column from x=510 (only 20px gap)
            ocr.append(make_ocr(f"left {i}", x_min=50, y_min=y, x_max=488, y_max=y + 20))
            ocr.append(make_ocr(f"right {i}", x_min=512, y_min=y, x_max=950, y_max=y + 20))
        # min_gap_fraction=0.10 means gap must be 100px; 20px gap fails
        result = detect(ocr, min_gap_fraction=0.10)
        assert not result.is_multi_column

    def test_wide_gap_detected(self):
        # Large gap (150px out of 1000px = 15%)
        ocr = []
        for i in range(12):
            y = 50 + i * 30
            ocr.append(make_ocr(f"left {i}", x_min=50, y_min=y, x_max=400, y_max=y + 20))
            ocr.append(make_ocr(f"right {i}", x_min=550, y_min=y, x_max=950, y_max=y + 20))
        result = detect(ocr, min_gap_fraction=0.03)
        assert result.is_multi_column


# ── Column width requirements ──────────────────────────────────────────────────

class TestColumnWidth:
    def test_too_narrow_column_not_detected(self):
        # Columns only 50px wide in a 1000px page
        ocr = []
        for i in range(12):
            y = 50 + i * 30
            ocr.append(make_ocr(f"left {i}", x_min=50, y_min=y, x_max=100, y_max=y + 20))
            ocr.append(make_ocr(f"right {i}", x_min=900, y_min=y, x_max=950, y_max=y + 20))
        # min_column_width_fraction=0.30 → column must be >= 300px
        result = detect(ocr, min_column_width_fraction=0.30)
        assert not result.is_multi_column


# ── Max columns ────────────────────────────────────────────────────────────────

class TestMaxColumns:
    def test_too_many_columns_rejected(self):
        # Three columns
        ocr = []
        for i in range(12):
            y = 50 + i * 30
            ocr.append(make_ocr(f"a {i}", x_min=20, y_min=y, x_max=260, y_max=y + 20))
            ocr.append(make_ocr(f"b {i}", x_min=370, y_min=y, x_max=630, y_max=y + 20))
            ocr.append(make_ocr(f"c {i}", x_min=740, y_min=y, x_max=980, y_max=y + 20))
        # max_columns=2 → three-column layout is rejected
        result = detect(ocr, max_columns=2)
        assert not result.is_multi_column

    def test_three_columns_detected_when_allowed(self):
        # Three evenly spaced columns with clear gaps
        ocr = []
        for i in range(14):
            y = 50 + i * 30
            ocr.append(make_ocr(f"a {i}", x_min=20, y_min=y, x_max=260, y_max=y + 20))
            ocr.append(make_ocr(f"b {i}", x_min=370, y_min=y, x_max=630, y_max=y + 20))
            ocr.append(make_ocr(f"c {i}", x_min=740, y_min=y, x_max=980, y_max=y + 20))
        result = detect(ocr, max_columns=4)
        assert result.is_multi_column
        assert result.num_columns == 3


# ── column_index_for_x ─────────────────────────────────────────────────────────

class TestColumnIndexForX:
    def test_x_in_left_column_returns_0(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert result.column_index_for_x(200.0) == 0

    def test_x_in_right_column_returns_1(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        assert result.column_index_for_x(800.0) == 1

    def test_x_in_gap_returns_default(self):
        ocr = _left_ocr() + _right_ocr()
        result = detect(ocr)
        # Gap returns 0 (default fallback)
        idx = result.column_index_for_x(500.0)
        assert idx == 0


# ── sort_elements_by_column_order ─────────────────────────────────────────────

class TestSortByColumnOrder:
    def _two_col_layout(self) -> ColumnLayoutResult:
        return ColumnLayoutResult(
            is_multi_column=True,
            num_columns=2,
            columns=[(0.0, 450.0), (550.0, 1000.0)],
            gap_regions=[(450.0, 550.0)],
            confidence=0.9,
            page_width=PAGE_W,
            page_height=PAGE_H,
        )

    def test_single_column_sorts_by_y(self):
        elems = [
            make_element(ElementType.TEXT, "c", y_min=100, y_max=120),
            make_element(ElementType.TEXT, "a", y_min=10,  y_max=30),
            make_element(ElementType.TEXT, "b", y_min=50,  y_max=70),
        ]
        sorted_elems = sort_elements_by_column_order(elems, SINGLE_COLUMN)
        y_vals = [e.bbox.y_min for e in sorted_elems]
        assert y_vals == sorted(y_vals)

    def test_two_column_left_before_right(self):
        layout = self._two_col_layout()
        # Right column element at y=10, left column element at y=100
        # Column order: left (col 0) must come first regardless of y
        right_elem = make_element(ElementType.TEXT, "right", x_min=600, x_max=900, y_min=10, y_max=30)
        left_elem = make_element(ElementType.TEXT, "left",  x_min=50,  x_max=400, y_min=100, y_max=120)
        sorted_elems = sort_elements_by_column_order([right_elem, left_elem], layout)
        assert sorted_elems[0].content == "left"
        assert sorted_elems[1].content == "right"

    def test_two_column_top_bottom_within_column(self):
        layout = self._two_col_layout()
        top_left    = make_element(ElementType.TEXT, "top_left",    x_min=50,  x_max=400, y_min=10,  y_max=30)
        bottom_left = make_element(ElementType.TEXT, "bottom_left", x_min=50,  x_max=400, y_min=200, y_max=220)
        top_right   = make_element(ElementType.TEXT, "top_right",   x_min=600, x_max=900, y_min=10,  y_max=30)
        sorted_elems = sort_elements_by_column_order(
            [bottom_left, top_right, top_left], layout
        )
        contents = [e.content for e in sorted_elems]
        assert contents.index("top_left") < contents.index("bottom_left")
        assert contents.index("bottom_left") < contents.index("top_right")

    def test_empty_list_returns_empty(self):
        result = sort_elements_by_column_order([], SINGLE_COLUMN)
        assert result == []

    def test_elements_without_bbox_do_not_crash(self):
        elem = make_element(ElementType.TEXT, "no bbox", y_min=0)
        elem.bbox = None
        layout = self._two_col_layout()
        result = sort_elements_by_column_order([elem], layout)
        assert len(result) == 1
