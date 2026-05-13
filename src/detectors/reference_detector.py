"""
Reference / Citation Detection Module

Detects in-document reference structures from pre-computed OCR text:

  REFERENCE (ref_type="citation")
    In-text markers that point to a bibliography entry.
    Supported formats:
      - Numeric bracket:   [1]  [1,2,3]  [1–3]  [Smith2020]
      - Author-year paren: (Smith, 2020)  (Smith & Jones, 2020)
                           (Smith et al., 2020)  (Smith et al. 2020)

  REFERENCE (ref_type="footnote")
    Numbered or symbol-prefixed short text lines that appear in the lower
    portion of the page (configurable zone fraction).  Detects markers
    such as "1.", "¹", "*", "†", "‡".

  REFERENCE (ref_type="bibliography")
    Entries in a References / Bibliography section.  Identified by:
      (a) an OCR result that contains a known section heading keyword, and
      (b) subsequent OCR results in the same vertical region whose content
          matches a bibliography-entry pattern (starts with a numeric or
          author-name token followed by year information).

Each REFERENCE element stores:
  - content : the verbatim OCR text of the reference (faithful extraction)
  - reference_id : the detected marker string ([1], ¹, "(Smith, 2020)", …)
  - ref_type : "citation", "footnote", or "bibliography"
  - location : "in-text", "footnote", or "bibliography"
  - target_ref : filled in when an in-text citation matches a bibliography
                 entry by reference_id; left None otherwise.

Cross-linking is best-effort within a single page: in-text [1] is linked to
the bibliography element whose reference_id is also "1".  Multi-page linking
is deferred to the batch-processing layer.

Follows Config + Detector + Trace pattern.
"""

import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from data_models import (
    BoundingBox,
    ElementType,
    OCRTextResult,
    Reference,
    StructuralElement,
)

import logging
logger = logging.getLogger(__name__)


# ── In-text citation patterns ─────────────────────────────────────────────────

_NUMERIC_CITE_RE = re.compile(
    r"\[(\d+(?:[\-–]\d+)?(?:,\s*\d+(?:[\-–]\d+)?)*)\]"
)

_AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][a-zA-Z\-']+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][a-zA-Z\-']+)?)"
    r",?\s+(\d{4}[a-z]?)\)"
)

# ── Footnote marker patterns ──────────────────────────────────────────────────
# Superscript digits: ¹ (1) ² (2) ³ (3) ⁴-⁹ (4-9) ⁰ (0)
# Footnote symbols: * † (dagger) ‡ (double-dagger) § (section) ¶ (pilcrow)

_FOOTNOTE_MARKER_RE = re.compile(
    r"^(?P<marker>\d{1,3}[\.\)]\s"
    r"|[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s"
    r"|[*†‡§¶]+\s)"
)

# ── Bibliography section headings (case-insensitive) ─────────────────────────

_BIB_HEADING_RE = re.compile(
    r"\b(?:References|Bibliography|Works\s+Cited|Sources|Literature)\b",
    re.IGNORECASE,
)

# ── Bibliography entry: starts with [N] or N. followed by a capital letter ───

_BIB_ENTRY_RE = re.compile(
    r"^(?:\[(?P<nid>\d+)\]|\[?(?P<aid>[A-Z][a-zA-Z\-]+(?:\d{4})?)\]?|(?P<nid2>\d+)[\.\)])\s+[A-Z]"
)


@dataclass
class ReferenceDetectorConfig:
    """Configuration for ReferenceDetector."""

    # In-text citation detection
    detect_in_text_numeric: bool = True
    detect_in_text_author_year: bool = True

    # Footnote detection
    detect_footnotes: bool = True
    # Fraction of page height from the bottom that is the "footnote zone"
    footnote_zone_fraction: float = 0.22

    # Bibliography detection
    detect_bibliography: bool = True
    # Minimum number of consecutive OCR results after a heading to form a bibliography
    min_bib_entries: int = 2

    # Cross-link in-text numeric citations to bibliography entries on the same page
    cross_link_numeric: bool = True

    # Minimum OCR confidence for a result to be considered for reference detection
    min_ocr_confidence: float = 0.30

    min_output_confidence: float = 0.55


@dataclass
class ReferenceDetectionTrace:
    """Records what ReferenceDetector.detect() found."""
    citations_found: int = 0
    footnotes_found: int = 0
    bibliography_entries_found: int = 0
    cross_links_made: int = 0
    ocr_results_scanned: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[ReferenceDetectorConfig] = None


class ReferenceDetector:
    """
    Detects citations, footnotes, and bibliography entries from OCR results.

    Args:
        config: ReferenceDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[ReferenceDetectorConfig] = None) -> None:
        self.config = config or ReferenceDetectorConfig()

    def detect(
        self,
        image: Union[np.ndarray, Tuple[int, int]],
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], ReferenceDetectionTrace]:
        """
        Detect references in OCR results.

        This detector is text-pattern only — no pixel data is used.
        The ``image`` argument accepts either a NumPy array (consistent with
        the other detectors) or a plain (height, width) tuple for callers
        that do not want to pass the full image.

        Args:
            image:       NumPy image array, or (height, width) tuple.
            ocr_results: Pre-computed OCR results for the page.
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        cfg = self.config
        trace = ReferenceDetectionTrace(config=cfg)
        trace.ocr_results_scanned = len(ocr_results)
        elements: List[StructuralElement] = []

        if isinstance(image, np.ndarray):
            image_h, image_w = image.shape[:2]
        else:
            image_h, image_w = image
        footnote_y_threshold = image_h * (1.0 - cfg.footnote_zone_fraction)

        # Filter out very low-confidence results up front
        valid_ocrs = [r for r in ocr_results if r.confidence >= cfg.min_ocr_confidence]

        # ── In-text citations ─────────────────────────────────────────────────
        in_text_numeric_map: Dict[str, str] = {}  # ref_id → element_id

        for ocr in valid_ocrs:
            text = ocr.text

            if cfg.detect_in_text_numeric:
                for m in _NUMERIC_CITE_RE.finditer(text):
                    ref_id = m.group(1).strip()
                    elem = self._make_citation(
                        ocr=ocr,
                        ref_id=ref_id,
                        content=m.group(0),
                        page_number=page_number,
                    )
                    elements.append(elem)
                    in_text_numeric_map[ref_id] = elem.element_id
                    trace.citations_found += 1

            if cfg.detect_in_text_author_year:
                for m in _AUTHOR_YEAR_RE.finditer(text):
                    ref_id = f"{m.group(1).strip()}, {m.group(2)}"
                    elem = self._make_citation(
                        ocr=ocr,
                        ref_id=ref_id,
                        content=m.group(0),
                        page_number=page_number,
                    )
                    elements.append(elem)
                    trace.citations_found += 1

        # ── Footnotes ─────────────────────────────────────────────────────────
        if cfg.detect_footnotes:
            footnote_ocrs = [
                r for r in valid_ocrs
                if r.bbox.y_min >= footnote_y_threshold
            ]
            # Sort top-to-bottom
            footnote_ocrs.sort(key=lambda r: r.bbox.y_min)
            for ocr in footnote_ocrs:
                m = _FOOTNOTE_MARKER_RE.match(ocr.text.strip())
                if m:
                    marker = m.group("marker").strip().rstrip(".")
                    elem = self._make_footnote(
                        ocr=ocr,
                        marker=marker,
                        page_number=page_number,
                    )
                    elements.append(elem)
                    trace.footnotes_found += 1

        # ── Bibliography ──────────────────────────────────────────────────────
        bib_id_map: Dict[str, str] = {}  # ref_id → element_id

        if cfg.detect_bibliography:
            sorted_ocrs = sorted(valid_ocrs, key=lambda r: r.bbox.y_min)
            bib_start = None
            for i, ocr in enumerate(sorted_ocrs):
                if _BIB_HEADING_RE.search(ocr.text):
                    bib_start = i + 1
                    break

            if bib_start is not None:
                bib_candidates = sorted_ocrs[bib_start:]
                consecutive: List[OCRTextResult] = []
                for ocr in bib_candidates:
                    m = _BIB_ENTRY_RE.match(ocr.text.strip())
                    if m:
                        consecutive.append(ocr)
                    else:
                        if len(consecutive) >= cfg.min_bib_entries:
                            for b_ocr in consecutive:
                                bm = _BIB_ENTRY_RE.match(b_ocr.text.strip())
                                b_ref_id = (
                                    bm.group("nid") or bm.group("nid2") or bm.group("aid") or ""
                                ).strip()
                                elem = self._make_bib_entry(
                                    ocr=b_ocr,
                                    ref_id=b_ref_id,
                                    page_number=page_number,
                                )
                                elements.append(elem)
                                if b_ref_id:
                                    bib_id_map[b_ref_id] = elem.element_id
                                trace.bibliography_entries_found += 1
                        consecutive = []
                # flush last group
                if len(consecutive) >= cfg.min_bib_entries:
                    for b_ocr in consecutive:
                        bm = _BIB_ENTRY_RE.match(b_ocr.text.strip())
                        b_ref_id = (
                            bm.group("nid") or bm.group("nid2") or bm.group("aid") or ""
                        ).strip()
                        elem = self._make_bib_entry(
                            ocr=b_ocr,
                            ref_id=b_ref_id,
                            page_number=page_number,
                        )
                        elements.append(elem)
                        if b_ref_id:
                            bib_id_map[b_ref_id] = elem.element_id
                        trace.bibliography_entries_found += 1

        # ── Cross-link in-text numeric → bibliography ─────────────────────────
        if cfg.cross_link_numeric and bib_id_map:
            for elem in elements:
                if (
                    isinstance(elem.content, Reference)
                    and elem.content.ref_type == "citation"
                    and elem.content.location == "in-text"
                ):
                    rid = elem.content.reference_id
                    if rid in bib_id_map:
                        elem.content.target_ref = bib_id_map[rid]
                        trace.cross_links_made += 1

        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_citation(
        self,
        ocr: OCRTextResult,
        ref_id: str,
        content: str,
        page_number: int,
    ) -> StructuralElement:
        ref = Reference(
            content=content,
            ref_type="citation",
            reference_id=ref_id,
            bbox=ocr.bbox,
            location="in-text",
        )
        return StructuralElement(
            element_id=f"ref_{uuid.uuid4().hex[:8]}",
            element_type=ElementType.REFERENCE,
            content=ref,
            bbox=ocr.bbox,
            confidence=max(self.config.min_output_confidence, ocr.confidence),
            page_number=page_number,
            processing_method="reference_detector_in_text",
        )

    def _make_footnote(
        self,
        ocr: OCRTextResult,
        marker: str,
        page_number: int,
    ) -> StructuralElement:
        ref = Reference(
            content=ocr.text.strip(),
            ref_type="footnote",
            reference_id=marker,
            bbox=ocr.bbox,
            location="footnote",
        )
        return StructuralElement(
            element_id=f"ref_{uuid.uuid4().hex[:8]}",
            element_type=ElementType.REFERENCE,
            content=ref,
            bbox=ocr.bbox,
            confidence=max(self.config.min_output_confidence, ocr.confidence),
            page_number=page_number,
            processing_method="reference_detector_footnote",
        )

    def _make_bib_entry(
        self,
        ocr: OCRTextResult,
        ref_id: str,
        page_number: int,
    ) -> StructuralElement:
        ref = Reference(
            content=ocr.text.strip(),
            ref_type="bibliography",
            reference_id=ref_id,
            bbox=ocr.bbox,
            location="bibliography",
        )
        return StructuralElement(
            element_id=f"ref_{uuid.uuid4().hex[:8]}",
            element_type=ElementType.REFERENCE,
            content=ref,
            bbox=ocr.bbox,
            confidence=max(self.config.min_output_confidence, ocr.confidence),
            page_number=page_number,
            processing_method="reference_detector_bibliography",
        )
