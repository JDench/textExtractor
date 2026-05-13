"""
Tests for IndexDetector — back-of-book index entry extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from detectors.index_detector import IndexDetector, IndexDetectorConfig
from data_models import ElementType, IndexEntry
from helpers import make_ocr


_BLANK = np.zeros((400, 600, 3), dtype=np.uint8)


def detect(ocr_results, **cfg_kwargs):
    cfg = IndexDetectorConfig(**cfg_kwargs) if cfg_kwargs else IndexDetectorConfig()
    detector = IndexDetector(cfg)
    elements, trace = detector.detect(_BLANK, ocr_results, page_number=1)
    return elements, trace


# ── No index heading ──────────────────────────────────────────────────────────

class TestNoIndex:
    def test_no_heading_returns_empty(self):
        ocrs = [make_ocr("Some body text", y_min=0), make_ocr("More text", y_min=30)]
        elems, trace = detect(ocrs)
        assert elems == []
        assert trace.index_found is False

    def test_too_few_entries_returns_empty(self):
        ocrs = [
            make_ocr("Index", x_min=10, y_min=0),
            make_ocr("alpha, 5", x_min=10, y_min=30),
            make_ocr("beta, 7", x_min=10, y_min=60),
        ]
        # default min_entries_to_confirm=3 but we only have 2
        elems, trace = detect(ocrs, min_entries_to_confirm=3)
        assert elems == []


# ── Heading recognition ───────────────────────────────────────────────────────

class TestHeadingVariants:
    @pytest.mark.parametrize("heading", [
        "Index",
        "Subject Index",
        "Author Index",
        "Name Index",
        "General Index",
    ])
    def test_heading_variants(self, heading):
        ocrs = [
            make_ocr(heading, x_min=10, y_min=0),
            make_ocr("alpha, 5", x_min=10, y_min=30),
            make_ocr("beta, 7", x_min=10, y_min=60),
            make_ocr("gamma, 9", x_min=10, y_min=90),
        ]
        elems, trace = detect(ocrs, min_entries_to_confirm=3)
        assert trace.heading_match_text is not None
        assert len(elems) == 3


# ── Term and page number extraction ──────────────────────────────────────────

class TestTermExtraction:
    def _index_ocrs(self, entries):
        """entries = [(text, x_min, y_min)]"""
        header = [make_ocr("Index", x_min=10, y_min=0)]
        return header + [make_ocr(t, x_min=x, y_min=y) for t, x, y in entries]

    def test_simple_term_and_page(self):
        ocrs = self._index_ocrs([
            ("algorithm, 5", 10, 30),
            ("binary search, 12", 10, 60),
            ("complexity, 18", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        terms = [e.content.term for e in elems]
        assert "algorithm" in terms

    def test_multiple_pages(self):
        ocrs = self._index_ocrs([
            ("sorting, 5, 10, 15", 10, 30),
            ("searching, 20, 25", 10, 60),
            ("hashing, 30", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        sorting_elem = next(e for e in elems if "sorting" in e.content.term)
        assert 5 in sorting_elem.content.page_numbers
        assert 10 in sorting_elem.content.page_numbers
        assert 15 in sorting_elem.content.page_numbers

    def test_page_range_expanded(self):
        ocrs = self._index_ocrs([
            ("loops, 5-8", 10, 30),
            ("functions, 12", 10, 60),
            ("classes, 20", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        loops = next(e for e in elems if "loops" in e.content.term)
        assert loops.content.page_numbers == [5, 6, 7, 8]

    def test_see_also_extracted(self):
        ocrs = self._index_ocrs([
            ("sorting. See also algorithms", 10, 30),
            ("searching, 20", 10, 60),
            ("hashing, 30", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        sorting = next(e for e in elems if "sorting" in e.content.term)
        assert "algorithms" in sorting.content.see_also

    def test_see_reference(self):
        ocrs = self._index_ocrs([
            ("BST. See binary search tree", 10, 30),
            ("graph, 50", 10, 60),
            ("tree, 60", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        bst = next(e for e in elems if "BST" in e.content.term)
        assert len(bst.content.see_also) > 0


# ── Indentation levels ────────────────────────────────────────────────────────

class TestLevels:
    def _ocrs(self, entries):
        header = [make_ocr("Index", x_min=10, y_min=0)]
        return header + [make_ocr(t, x_min=x, y_min=y) for t, x, y in entries]

    def test_main_entry_level_1(self):
        ocrs = self._ocrs([
            ("sorting, 5", 10, 30),
            ("searching, 20", 10, 60),
            ("hashing, 30", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        assert all(e.content.level == 1 for e in elems)

    def test_sub_entry_level_2(self):
        # x_min=40 → indented beyond default level_indent_threshold=15 (ref_x=10, indent=30)
        ocrs = self._ocrs([
            ("sorting, 5", 10, 30),
            ("  quicksort, 6", 40, 60),   # sub-entry
            ("searching, 20", 10, 90),
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=3)
        sub = next(e for e in elems if "quicksort" in e.content.term)
        assert sub.content.level == 2

    def test_alpha_dividers_skipped(self):
        ocrs = self._ocrs([
            ("A", 10, 30),               # alphabetical divider
            ("algorithm, 5", 10, 60),
            ("binary search, 12", 10, 90),
            ("B", 10, 120),              # divider
        ])
        elems, _ = detect(ocrs, min_entries_to_confirm=2, skip_alpha_dividers=True)
        terms = [e.content.term for e in elems]
        assert "A" not in terms
        assert "B" not in terms


# ── Element structure ─────────────────────────────────────────────────────────

class TestElementStructure:
    def _three_entry_ocrs(self):
        return [
            make_ocr("Index", x_min=10, y_min=0),
            make_ocr("alpha, 5", x_min=10, y_min=30),
            make_ocr("beta, 7", x_min=10, y_min=60),
            make_ocr("gamma, 9", x_min=10, y_min=90),
        ]

    def test_element_type(self):
        elems, _ = detect(self._three_entry_ocrs(), min_entries_to_confirm=3)
        assert all(e.element_type == ElementType.INDEX for e in elems)

    def test_content_is_index_entry(self):
        elems, _ = detect(self._three_entry_ocrs(), min_entries_to_confirm=3)
        assert all(isinstance(e.content, IndexEntry) for e in elems)

    def test_trace(self):
        elems, trace = detect(self._three_entry_ocrs(), min_entries_to_confirm=3)
        assert trace.index_found is True
        assert trace.entries_found == 3
        assert trace.main_entries == 3
        assert trace.sub_entries == 0
