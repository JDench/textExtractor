"""
Table-of-Contents Detection Module

Extracts table-of-contents entries from OCR results.

Detection strategy
------------------
1. Scan OCR results for a TOC heading line ("Table of Contents", "Contents", …).
2. For every OCR result that follows the heading (until a clear non-TOC line
   ends the section), attempt to match the entry pattern:

      [indent]  [optional §number]  Title text  ...  page_num

   Both dots-leader ("Introduction ........... 3") and space-leader
   ("Introduction          3") formats are matched.

3. When a line is a title with no page number on the same OCR result, the
   immediately following OCR result is checked; if it contains only digits it
   is treated as the page number for the preceding title (handles Tesseract
   splitting the leader from the text).

4. The x_min of each matching OCR result is used to derive the hierarchy
   level: x-min values are clustered into up to 6 bands and mapped to levels
   1 (leftmost) through 6 (most indented).

Output
------
Each TOC entry becomes a StructuralElement with:
  element_type = ElementType.TABLE_OF_CONTENTS
  content      = TableOfContents(title, page_number, level, bbox, confidence)

target_heading_id on the TableOfContents object is left None here; the
HierarchyBuilder fills it after all elements have been collected.

Follows Config + Detector + Trace pattern.
"""

import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_models import (
    BoundingBox,
    ElementType,
    OCRTextResult,
    StructuralElement,
    TableOfContents,
)

import logging
logger = logging.getLogger(__name__)


# ── Pattern constants ─────────────────────────────────────────────────────────

_TOC_HEADING_RE = re.compile(
    r"\b(?:Table\s+of\s+Contents?|Contents?|TOC|Sommaire|Table\s+des\s+mati[eè]res)\b",
    re.IGNORECASE,
)

# Full entry: title + dots-or-spaces leader + page number
_ENTRY_DOTS_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<title>.+?)"
    r"\s*[.\s]{3,}\s*"          # leader: 3+ dots or spaces
    r"(?P<pagenum>\d{1,5})"
    r"\s*$"
)

# Fallback entry: title + 4+ spaces + page number (no dots)
_ENTRY_SPACE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<title>.+?)"
    r"\s{4,}"
    r"(?P<pagenum>\d{1,5})"
    r"\s*$"
)

# A line that is just a page number (sometimes split from the title by Tesseract)
_PAGE_ONLY_RE = re.compile(r"^\s*(?:–\s*|\-\s*)?(\d{1,5})\s*$")

# Lines that clearly terminate the TOC section
_SECTION_BREAK_RE = re.compile(
    r"^(?:Chapter|Part|Section|Appendix|Introduction|Preface|Abstract|Summary)",
    re.IGNORECASE,
)


@dataclass
class TOCDetectorConfig:
    """Configuration for TOCDetector."""

    # OCR confidence below which we skip a result
    min_ocr_confidence: float = 0.25

    # Maximum number of blank/non-matching lines to tolerate before giving up
    max_gap_lines: int = 5

    # Minimum number of entries that must match before we commit to "this is a TOC"
    min_entries_to_confirm: int = 2

    # Maximum levels in the hierarchy (deeper indentation is clamped)
    max_levels: int = 6

    # Minimum x-min pixel difference between two indentation clusters
    level_cluster_min_gap: float = 12.0

    min_output_confidence: float = 0.50


@dataclass
class TOCDetectionTrace:
    """Records what TOCDetector.detect() found."""
    toc_found: bool = False
    entries_found: int = 0
    max_level_seen: int = 0
    heading_match_text: Optional[str] = None
    ocr_results_scanned: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[TOCDetectorConfig] = None


class TOCDetector:
    """
    Extracts table-of-contents entries from OCR results.

    Args:
        config: TOCDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[TOCDetectorConfig] = None) -> None:
        self.config = config or TOCDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], TOCDetectionTrace]:
        """
        Detect TOC entries in an image.

        Args:
            image:       NumPy image array (not used for pixel data; accepted for
                         API consistency with other detectors).
            ocr_results: Pre-computed OCR results for the page.
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        cfg = self.config
        trace = TOCDetectionTrace(config=cfg)
        trace.ocr_results_scanned = len(ocr_results)
        elements: List[StructuralElement] = []

        # Sort top-to-bottom
        sorted_ocrs = sorted(
            [r for r in ocr_results if r.confidence >= cfg.min_ocr_confidence],
            key=lambda r: (r.bbox.y_min, r.bbox.x_min),
        )

        # ── Find TOC heading ───────────────────────────────────────────────────
        toc_start_idx: Optional[int] = None
        for i, ocr in enumerate(sorted_ocrs):
            if _TOC_HEADING_RE.search(ocr.text):
                toc_start_idx = i + 1
                trace.heading_match_text = ocr.text.strip()
                break

        if toc_start_idx is None:
            trace.processing_time_seconds = time.perf_counter() - t0
            return elements, trace

        # ── Extract entries ────────────────────────────────────────────────────
        raw_entries: List[Tuple[OCRTextResult, str, int, float]] = []
        # (ocr_result, title, pagenum_int, x_min)
        gap_count = 0
        pending_title: Optional[Tuple[OCRTextResult, str, float]] = None

        for ocr in sorted_ocrs[toc_start_idx:]:
            text = ocr.text.strip()
            if not text:
                continue

            # Try to claim a pending title-only line with a page-number follow-up
            if pending_title is not None:
                m = _PAGE_ONLY_RE.match(text)
                if m:
                    p_ocr, p_title, p_x = pending_title
                    raw_entries.append((p_ocr, p_title, int(m.group(1)), p_x))
                    pending_title = None
                    gap_count = 0
                    continue
                else:
                    pending_title = None  # not paired — discard

            # Try full pattern matches
            m = _ENTRY_DOTS_RE.match(text) or _ENTRY_SPACE_RE.match(text)
            if m:
                title = m.group("title").strip()
                pagenum = int(m.group("pagenum"))
                if title:
                    raw_entries.append((ocr, title, pagenum, ocr.bbox.x_min))
                    gap_count = 0
                    continue

            # Is this a title-only line? (no page number yet)
            # Accept lines that look like section titles (contain letters, not just numbers)
            if re.search(r"[A-Za-z]{3,}", text) and not _PAGE_ONLY_RE.match(text):
                pending_title = (ocr, text, ocr.bbox.x_min)
                gap_count += 1
            else:
                gap_count += 1

            if gap_count > cfg.max_gap_lines:
                break

        if len(raw_entries) < cfg.min_entries_to_confirm:
            trace.processing_time_seconds = time.perf_counter() - t0
            return elements, trace

        trace.toc_found = True

        # ── Compute indentation levels ─────────────────────────────────────────
        x_mins = sorted(set(round(x) for _, _, _, x in raw_entries))
        level_map = self._cluster_levels(x_mins, cfg.level_cluster_min_gap, cfg.max_levels)

        # ── Build elements ─────────────────────────────────────────────────────
        for ocr, title, pagenum, x_min in raw_entries:
            level = level_map.get(round(x_min), 1)
            trace.max_level_seen = max(trace.max_level_seen, level)

            toc_entry = TableOfContents(
                title=title,
                page_number=pagenum,
                level=level,
                bbox=ocr.bbox,
                confidence=max(cfg.min_output_confidence, ocr.confidence),
            )
            elements.append(StructuralElement(
                element_id=f"toc_{uuid.uuid4().hex[:8]}",
                element_type=ElementType.TABLE_OF_CONTENTS,
                content=toc_entry,
                bbox=ocr.bbox,
                confidence=max(cfg.min_output_confidence, ocr.confidence),
                page_number=page_number,
                nesting_level=level - 1,
                processing_method="toc_detector_pattern",
                metadata={"referenced_page": pagenum},
            ))

        trace.entries_found = len(elements)
        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _cluster_levels(
        self,
        x_mins: List[int],
        min_gap: float,
        max_levels: int,
    ) -> Dict[int, int]:
        """
        Cluster x-min values into indentation levels.

        The smallest x-min is level 1; each new cluster (gap ≥ min_gap) is
        the next level, up to max_levels.
        """
        if not x_mins:
            return {}
        clusters: List[int] = [x_mins[0]]
        for x in x_mins[1:]:
            if x - clusters[-1] >= min_gap:
                clusters.append(x)
                if len(clusters) >= max_levels:
                    break
        level_map: Dict[int, int] = {}
        for x in x_mins:
            # Assign to nearest cluster
            nearest = min(clusters, key=lambda c: abs(c - x))
            level_map[x] = clusters.index(nearest) + 1
        return level_map
