"""
Tests for ReferenceDetector — in-text citations, footnotes, and bibliography.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from detectors.reference_detector import ReferenceDetector, ReferenceDetectorConfig
from data_models import ElementType, Reference
from helpers import make_ocr


# ── Helper ─────────────────────────────────────────────────────────────────────

_PAGE_H = 600
_PAGE_W = 400
_BLANK = np.zeros((_PAGE_H, _PAGE_W, 3), dtype=np.uint8)
_SHAPE = (_PAGE_H, _PAGE_W)


def detect(ocr_results, image=_BLANK, page_number=1, **cfg_kwargs):
    cfg = ReferenceDetectorConfig(**cfg_kwargs) if cfg_kwargs else ReferenceDetectorConfig()
    detector = ReferenceDetector(cfg)
    elements, trace = detector.detect(image, ocr_results, page_number)
    return elements, trace


def detect_shape(ocr_results, **cfg_kwargs):
    """Use tuple shape instead of ndarray (ReferenceDetector supports both)."""
    return detect(ocr_results, image=_SHAPE, **cfg_kwargs)


# ── No references ─────────────────────────────────────────────────────────────

class TestNoReferences:
    def test_plain_text_returns_empty(self):
        ocrs = [make_ocr("This is a sentence with no citations.")]
        elems, _ = detect(ocrs)
        assert elems == []


# ── In-text numeric citations ─────────────────────────────────────────────────

class TestNumericCitations:
    def test_simple_bracket(self):
        ocrs = [make_ocr("As shown in [1], this is true.")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) == 1
        assert citations[0].content.reference_id == "1"

    def test_multiple_in_one_line(self):
        ocrs = [make_ocr("See [1] and [2] for details.")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) == 2

    def test_range_citation(self):
        ocrs = [make_ocr("Results are shown in [2-4].")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) == 1
        assert "2" in citations[0].content.reference_id

    def test_comma_list_citation(self):
        ocrs = [make_ocr("Multiple sources [1, 3, 5] confirm this.")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) >= 1

    def test_disabled_numeric_returns_none(self):
        ocrs = [make_ocr("See [1] for details.")]
        elems, _ = detect(ocrs, detect_in_text_numeric=False)
        citations = [e for e in elems if e.content.ref_type == "citation"
                     and e.content.location == "in-text"]
        assert len(citations) == 0


# ── In-text author-year citations ─────────────────────────────────────────────

class TestAuthorYearCitations:
    def test_simple(self):
        ocrs = [make_ocr("As noted by (Smith, 2020), this holds.")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) >= 1
        assert "Smith" in citations[0].content.reference_id

    def test_et_al(self):
        ocrs = [make_ocr("This was shown by (Smith et al., 2019).")]
        elems, _ = detect(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) >= 1

    def test_disabled_author_year(self):
        ocrs = [make_ocr("(Smith, 2020) proposed this.")]
        elems, _ = detect(ocrs, detect_in_text_author_year=False)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) == 0


# ── Footnotes ─────────────────────────────────────────────────────────────────

class TestFootnotes:
    def _footnote_y(self):
        """y_min in the footnote zone (bottom 22% of 600px = below y=468)."""
        return 490

    def test_numbered_footnote_in_zone(self):
        y = self._footnote_y()
        ocrs = [make_ocr("1. This is a footnote.", x_min=10, y_min=y, y_max=y + 18)]
        elems, _ = detect(ocrs)
        footnotes = [e for e in elems if e.content.ref_type == "footnote"]
        assert len(footnotes) == 1

    def test_footnote_above_zone_not_detected(self):
        # y_min well above the footnote zone
        ocrs = [make_ocr("1. This note is in the body.", x_min=10, y_min=50, y_max=70)]
        elems, _ = detect(ocrs)
        footnotes = [e for e in elems if e.content.ref_type == "footnote"]
        assert len(footnotes) == 0

    def test_disabled_footnotes(self):
        y = self._footnote_y()
        ocrs = [make_ocr("1. Footnote text.", x_min=10, y_min=y, y_max=y + 18)]
        elems, _ = detect(ocrs, detect_footnotes=False)
        footnotes = [e for e in elems if e.content.ref_type == "footnote"]
        assert len(footnotes) == 0


# ── Bibliography ──────────────────────────────────────────────────────────────

class TestBibliography:
    def _bib_ocrs(self):
        return [
            make_ocr("References", x_min=10, y_min=100),
            make_ocr("[1] Smith, J. (2020). Title. Journal.", x_min=10, y_min=130),
            make_ocr("[2] Jones, A. (2019). Another work. Press.", x_min=10, y_min=160),
        ]

    def test_bibliography_entries_detected(self):
        elems, _ = detect(self._bib_ocrs())
        bib = [e for e in elems if e.content.ref_type == "bibliography"]
        assert len(bib) >= 2

    def test_disabled_bibliography(self):
        elems, _ = detect(self._bib_ocrs(), detect_bibliography=False)
        bib = [e for e in elems if e.content.ref_type == "bibliography"]
        assert len(bib) == 0

    def test_heading_variants(self):
        for heading in ["Bibliography", "Works Cited", "Sources"]:
            ocrs = [
                make_ocr(heading, x_min=10, y_min=100),
                make_ocr("[1] Author A (2021). Book.", x_min=10, y_min=130),
                make_ocr("[2] Author B (2022). Article.", x_min=10, y_min=160),
            ]
            elems, _ = detect(ocrs)
            bib = [e for e in elems if e.content.ref_type == "bibliography"]
            assert len(bib) >= 1, f"Heading '{heading}' not recognised"


# ── Cross-linking ─────────────────────────────────────────────────────────────

class TestCrossLinking:
    def test_numeric_citation_linked_to_bibliography(self):
        ocrs = [
            make_ocr("According to [1] this is true.", x_min=10, y_min=50),
            make_ocr("References", x_min=10, y_min=300),
            make_ocr("[1] Smith, J. (2020). Title.", x_min=10, y_min=330),
            make_ocr("[2] Jones, A. (2021). Another.", x_min=10, y_min=360),
        ]
        elems, _ = detect(ocrs)
        citation = next(
            (e for e in elems if e.content.ref_type == "citation" and
             e.content.reference_id == "1"),
            None,
        )
        bib = next(
            (e for e in elems if e.content.ref_type == "bibliography"),
            None,
        )
        assert citation is not None
        assert bib is not None
        assert citation.content.target_ref == bib.element_id

    def test_cross_link_disabled(self):
        ocrs = [
            make_ocr("See [1].", x_min=10, y_min=50),
            make_ocr("References", x_min=10, y_min=300),
            make_ocr("[1] Author (2020). Title.", x_min=10, y_min=330),
        ]
        elems, _ = detect(ocrs, cross_link_numeric=False)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        for c in citations:
            assert c.content.target_ref is None


# ── Tuple image shape (API compatibility) ─────────────────────────────────────

class TestTupleImageShape:
    def test_accepts_tuple(self):
        ocrs = [make_ocr("See [1] and [2] for details.")]
        elems, _ = detect_shape(ocrs)
        citations = [e for e in elems if e.content.ref_type == "citation"]
        assert len(citations) == 2


# ── Element structure ─────────────────────────────────────────────────────────

class TestElementStructure:
    def test_element_type(self):
        ocrs = [make_ocr("According to [1] this is true.")]
        elems, _ = detect(ocrs)
        assert all(e.element_type == ElementType.REFERENCE for e in elems)

    def test_content_is_reference(self):
        ocrs = [make_ocr("See [1].")]
        elems, _ = detect(ocrs)
        assert all(isinstance(e.content, Reference) for e in elems)

    def test_trace(self):
        ocrs = [
            make_ocr("See [1] and [2].", x_min=10, y_min=50),
        ]
        _, trace = detect(ocrs)
        assert trace.citations_found >= 2
