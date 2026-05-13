"""
Tests for TOCDetector — table-of-contents entry extraction.

All tests pass synthetic OCR results; no real image or Tesseract call is made.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from detectors.toc_detector import TOCDetector, TOCDetectorConfig
from data_models import ElementType, TableOfContents
from helpers import make_ocr


# ── Helpers ────────────────────────────────────────────────────────────────────

_BLANK = np.zeros((400, 600, 3), dtype=np.uint8)


def detect(ocr_results, **cfg_kwargs):
    cfg = TOCDetectorConfig(**cfg_kwargs) if cfg_kwargs else TOCDetectorConfig()
    detector = TOCDetector(cfg)
    elements, trace = detector.detect(_BLANK, ocr_results, page_number=1)
    return elements, trace


def toc_ocrs(entries):
    """entries = list of (text, x_min, y_min) tuples."""
    return [make_ocr(text, x_min=x, y_min=y) for text, x, y in entries]


# ── No TOC heading ─────────────────────────────────────────────────────────────

class TestNoTOC:
    def test_no_heading_returns_empty(self):
        ocrs = [make_ocr("Some random text"), make_ocr("More text", y_min=30)]
        elems, trace = detect(ocrs)
        assert elems == []
        assert trace.toc_found is False
        assert trace.heading_match_text is None

    def test_partial_match_not_enough_entries(self):
        # Has a heading but only 1 entry — below min_entries_to_confirm=2
        ocrs = [
            make_ocr("Table of Contents", x_min=10, y_min=0),
            make_ocr("Introduction ...... 1", x_min=10, y_min=30),
        ]
        elems, trace = detect(ocrs)
        assert elems == []
        assert trace.toc_found is False


# ── TOC heading recognition ────────────────────────────────────────────────────

class TestTOCHeadingVariants:
    @pytest.mark.parametrize("heading", [
        "Table of Contents",
        "Contents",
        "TABLE OF CONTENTS",
        "TOC",
        "Sommaire",
    ])
    def test_heading_variants(self, heading):
        ocrs = [
            make_ocr(heading, x_min=10, y_min=0),
            make_ocr("Introduction ...... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ......... 5", x_min=10, y_min=60),
        ]
        elems, trace = detect(ocrs, min_entries_to_confirm=2)
        assert trace.heading_match_text is not None
        assert len(elems) == 2


# ── Entry pattern matching ────────────────────────────────────────────────────

class TestEntryPatterns:
    def _base(self):
        return [make_ocr("Table of Contents", x_min=10, y_min=0)]

    def test_dots_leader_entries(self):
        ocrs = self._base() + [
            make_ocr("Introduction ......... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ........... 5", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert len(elems) == 2
        titles = [e.content.title for e in elems]
        assert "Introduction" in titles
        assert "Chapter 1" in titles

    def test_space_leader_entries(self):
        ocrs = self._base() + [
            make_ocr("Introduction            1", x_min=10, y_min=30),
            make_ocr("Chapter 1               5", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert len(elems) == 2

    def test_page_numbers_extracted(self):
        ocrs = self._base() + [
            make_ocr("Intro ............. 3", x_min=10, y_min=30),
            make_ocr("Conclusion ........ 42", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        page_nums = {e.content.page_number for e in elems}
        assert 3 in page_nums
        assert 42 in page_nums

    def test_split_title_and_page_number(self):
        """Tesseract sometimes splits the title and the page number."""
        ocrs = self._base() + [
            make_ocr("Introduction", x_min=10, y_min=30),
            make_ocr("1", x_min=200, y_min=30),            # page-only follow-up
            make_ocr("Chapter One", x_min=10, y_min=60),
            make_ocr("5", x_min=200, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert len(elems) == 2


# ── Indentation levels ────────────────────────────────────────────────────────

class TestIndentationLevels:
    def test_sub_entry_gets_higher_level(self):
        ocrs = [
            make_ocr("Table of Contents", x_min=10, y_min=0),
            make_ocr("Chapter 1 ........ 5", x_min=10, y_min=30),
            make_ocr("  Section 1.1 .... 7", x_min=40, y_min=60),
            make_ocr("Chapter 2 ........ 15", x_min=10, y_min=90),
        ]
        elems, _ = detect(ocrs)
        levels = [e.content.level for e in elems]
        # "Chapter 1" and "Chapter 2" at level 1; "Section 1.1" at level 2
        assert levels[0] == 1  # Chapter 1
        assert levels[1] == 2  # Section 1.1
        assert levels[2] == 1  # Chapter 2


# ── Config parameters ─────────────────────────────────────────────────────────

class TestConfig:
    def test_min_entries_zero_accepts_one(self):
        ocrs = [
            make_ocr("Contents", x_min=10, y_min=0),
            make_ocr("Intro ......... 1", x_min=10, y_min=30),
        ]
        elems, _ = detect(ocrs, min_entries_to_confirm=1)
        assert len(elems) == 1

    def test_low_confidence_ocr_skipped(self):
        ocrs = [
            make_ocr("Table of Contents", x_min=10, y_min=0, confidence=0.9),
            # This entry has confidence below the default min_ocr_confidence=0.25
            make_ocr("Intro ...... 1", x_min=10, y_min=30, confidence=0.10),
            make_ocr("Chapter 1 .. 5", x_min=10, y_min=60, confidence=0.10),
        ]
        elems, _ = detect(ocrs)
        # Both entries skipped → below min_entries_to_confirm
        assert len(elems) == 0


# ── Element structure ─────────────────────────────────────────────────────────

class TestElementStructure:
    def test_element_type(self):
        ocrs = [
            make_ocr("Contents", x_min=10, y_min=0),
            make_ocr("Intro ......... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ..... 5", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert all(e.element_type == ElementType.TABLE_OF_CONTENTS for e in elems)

    def test_content_is_table_of_contents(self):
        ocrs = [
            make_ocr("Contents", x_min=10, y_min=0),
            make_ocr("Intro ......... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ..... 5", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert all(isinstance(e.content, TableOfContents) for e in elems)

    def test_target_heading_id_none(self):
        """HierarchyBuilder fills this later; detector leaves it None."""
        ocrs = [
            make_ocr("Contents", x_min=10, y_min=0),
            make_ocr("Intro ......... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ..... 5", x_min=10, y_min=60),
        ]
        elems, _ = detect(ocrs)
        assert all(e.content.target_heading_id is None for e in elems)

    def test_trace_records_entry_count(self):
        ocrs = [
            make_ocr("Contents", x_min=10, y_min=0),
            make_ocr("Intro ......... 1", x_min=10, y_min=30),
            make_ocr("Chapter 1 ..... 5", x_min=10, y_min=60),
        ]
        elems, trace = detect(ocrs)
        assert trace.entries_found == 2
        assert trace.toc_found is True
