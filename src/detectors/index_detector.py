"""
Index Detection Module

Extracts back-of-book index entries from OCR results.

Detection strategy
------------------
1. Scan for an "Index" section heading ("Index", "Subject Index", "Author Index", …).
2. From that heading onward, parse each OCR result as an index entry:

      [indent]  Term, page, page, page-range
      [indent]  Term. See also OtherTerm
      [indent]  Term

   Main entries start at the left margin; sub-entries are indented (larger
   x_min).  Level is derived by comparing x_min against the reference margin.

3. Page numbers are extracted with a regex that handles:
      - Single pages: 12
      - Comma lists: 12, 45, 67
      - Ranges: 12-15 or 12–15

4. "See also" and "See" cross-references are captured into IndexEntry.see_also.

Output
------
Each index entry → StructuralElement with:
  element_type = ElementType.INDEX
  content      = IndexEntry(term, page_numbers, level, bbox, confidence, see_also)

Levels:
  level = 1  → main entry (x_min ≤ reference_margin + cluster_gap)
  level = 2  → sub-entry
  level = 3  → sub-sub-entry (rarely occurs; clamped at max_levels)

Follows Config + Detector + Trace pattern.
"""

import re
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from data_models import (
    BoundingBox,
    ElementType,
    IndexEntry,
    OCRTextResult,
    StructuralElement,
)

import logging
logger = logging.getLogger(__name__)


# ── Pattern constants ─────────────────────────────────────────────────────────

_INDEX_HEADING_RE = re.compile(
    r"^\s*(?:Subject\s+|Author\s+|Name\s+|General\s+)?Index\s*$",
    re.IGNORECASE,
)

# Page number tokens in an index line (single or range)
_PAGE_TOKEN_RE = re.compile(r"\b(\d{1,5})(?:[\-–](\d{1,5}))?\b")

# See / See also cross-references
_SEE_ALSO_RE = re.compile(r"\bsee\s+also\s+(.+)", re.IGNORECASE)
_SEE_RE = re.compile(r"\bsee\s+(?!also\b)(.+)", re.IGNORECASE)

# Split term from its page list: "Term, 12, 45" → term part and page part
# The page section starts at the first comma followed by a digit
_TERM_PAGES_SPLIT_RE = re.compile(r"^(.+?),\s*(?=\d)")


@dataclass
class IndexDetectorConfig:
    """Configuration for IndexDetector."""

    # Minimum OCR confidence to consider a result
    min_ocr_confidence: float = 0.25

    # Pixel margin beyond the reference x_min to still be considered level 1
    level_indent_threshold: float = 15.0

    # Maximum indentation levels
    max_levels: int = 3

    # Stop extracting after this many consecutive non-entry lines
    max_gap_lines: int = 8

    # Minimum number of valid entries before committing (avoids false positives)
    min_entries_to_confirm: int = 3

    # Lines that look like alphabetical section dividers ("A", "B", …) are skipped
    skip_alpha_dividers: bool = True

    min_output_confidence: float = 0.50


@dataclass
class IndexDetectionTrace:
    """Records what IndexDetector.detect() found."""
    index_found: bool = False
    entries_found: int = 0
    main_entries: int = 0
    sub_entries: int = 0
    see_also_references: int = 0
    heading_match_text: Optional[str] = None
    ocr_results_scanned: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[IndexDetectorConfig] = None


class IndexDetector:
    """
    Extracts back-of-book index entries from OCR results.

    Args:
        config: IndexDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[IndexDetectorConfig] = None) -> None:
        self.config = config or IndexDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], IndexDetectionTrace]:
        """
        Detect index entries in an image.

        Args:
            image:       NumPy image array (used only for API consistency;
                         pixel data is not needed by this detector).
            ocr_results: Pre-computed OCR results for the page.
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        cfg = self.config
        trace = IndexDetectionTrace(config=cfg)
        trace.ocr_results_scanned = len(ocr_results)
        elements: List[StructuralElement] = []

        sorted_ocrs = sorted(
            [r for r in ocr_results if r.confidence >= cfg.min_ocr_confidence],
            key=lambda r: (r.bbox.y_min, r.bbox.x_min),
        )

        # ── Find index heading ─────────────────────────────────────────────────
        index_start_idx: Optional[int] = None
        for i, ocr in enumerate(sorted_ocrs):
            if _INDEX_HEADING_RE.match(ocr.text.strip()):
                index_start_idx = i + 1
                trace.heading_match_text = ocr.text.strip()
                break

        if index_start_idx is None:
            trace.processing_time_seconds = time.perf_counter() - t0
            return elements, trace

        # ── Determine reference (leftmost) x_min ──────────────────────────────
        candidate_ocrs = sorted_ocrs[index_start_idx:]
        if not candidate_ocrs:
            trace.processing_time_seconds = time.perf_counter() - t0
            return elements, trace
        ref_x = min(r.bbox.x_min for r in candidate_ocrs)

        # ── Parse entries ─────────────────────────────────────────────────────
        raw_entries: List[Tuple[OCRTextResult, str, List[int], List[str], int]] = []
        # (ocr, term, page_numbers, see_also, level)
        gap_count = 0

        for ocr in candidate_ocrs:
            text = ocr.text.strip()
            if not text:
                continue

            # Skip single-letter alphabetical dividers ("A", "B", …)
            if cfg.skip_alpha_dividers and re.match(r"^[A-Z]$", text):
                continue

            # ── Extract see-also / see references ────────────────────────────
            see_also: List[str] = []
            m_sa = _SEE_ALSO_RE.search(text)
            m_s = _SEE_RE.search(text)
            if m_sa:
                see_also = [s.strip() for s in re.split(r"[;,]", m_sa.group(1)) if s.strip()]
                text = text[: m_sa.start()].strip().rstrip(",")
            elif m_s:
                see_also = [s.strip() for s in re.split(r"[;,]", m_s.group(1)) if s.strip()]
                text = text[: m_s.start()].strip().rstrip(",")

            # ── Extract page numbers ──────────────────────────────────────────
            page_nums: List[int] = []
            m_split = _TERM_PAGES_SPLIT_RE.match(text)
            if m_split:
                term = m_split.group(1).strip()
                page_section = text[m_split.end():]
                for pm in _PAGE_TOKEN_RE.finditer(page_section):
                    start = int(pm.group(1))
                    end = int(pm.group(2)) if pm.group(2) else start
                    page_nums.extend(range(start, end + 1) if end - start < 50 else [start, end])
            else:
                # No comma split — the whole text is the term (no pages on this line)
                term = text.rstrip(",").strip()
                # Still try to find any trailing numbers
                for pm in _PAGE_TOKEN_RE.finditer(text):
                    page_nums.append(int(pm.group(1)))
                if page_nums:
                    # Remove the numbers from the term
                    term = _PAGE_TOKEN_RE.sub("", term).strip().rstrip(",").strip()

            if not term or not re.search(r"[A-Za-z]", term):
                gap_count += 1
                if gap_count > cfg.max_gap_lines:
                    break
                continue

            # ── Determine indentation level ───────────────────────────────────
            indent = ocr.bbox.x_min - ref_x
            if indent <= cfg.level_indent_threshold:
                level = 1
            elif indent <= cfg.level_indent_threshold * 2:
                level = 2
            else:
                level = min(3, cfg.max_levels)

            raw_entries.append((ocr, term, page_nums, see_also, level))
            gap_count = 0

        if len(raw_entries) < cfg.min_entries_to_confirm:
            trace.processing_time_seconds = time.perf_counter() - t0
            return elements, trace

        trace.index_found = True

        # ── Build elements ─────────────────────────────────────────────────────
        for ocr, term, page_nums, see_also, level in raw_entries:
            # page_numbers must all be ≥ 1 per IndexEntry validation
            valid_pages = [p for p in page_nums if p >= 1]
            if not valid_pages:
                valid_pages = [1]  # fallback to avoid validation failure

            conf = max(cfg.min_output_confidence, ocr.confidence)
            entry = IndexEntry(
                term=term,
                page_numbers=valid_pages,
                level=level,
                bbox=ocr.bbox,
                confidence=conf,
                see_also=see_also,
            )
            elem = StructuralElement(
                element_id=f"idx_{uuid.uuid4().hex[:8]}",
                element_type=ElementType.INDEX,
                content=entry,
                bbox=ocr.bbox,
                confidence=conf,
                page_number=page_number,
                nesting_level=level - 1,
                processing_method="index_detector_pattern",
                metadata={"page_references": valid_pages, "see_also": see_also},
            )
            elements.append(elem)

            if level == 1:
                trace.main_entries += 1
            else:
                trace.sub_entries += 1
            if see_also:
                trace.see_also_references += len(see_also)

        trace.entries_found = len(elements)
        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace
