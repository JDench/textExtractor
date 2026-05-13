"""
Caption ↔ Figure/Table Linking

Detects CAPTION elements and links them to spatially adjacent FIGURE or TABLE
elements within the same document page.

Linking strategy:
  1. Parse caption text to classify it as a figure or table caption.
  2. Find candidate FIGURE/TABLE elements on the same page.
  3. Compute bbox-edge distance between caption and each candidate.
  4. If the nearest candidate is within max_proximity_px, write cross-references
     into both elements' metadata dicts.

The operation is non-destructive: if no confident link can be made, both
elements remain unchanged and no metadata is written.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from data_models import (
    BoundingBox,
    Caption,
    DocumentResult,
    ElementType,
    StructuralElement,
)


# ── Caption text helpers ───────────────────────────────────────────────────────

_FIG_PATTERN = re.compile(r"\b(?:fig(?:ure)?s?|fig\.)\s*\d+", re.IGNORECASE)
_TBL_PATTERN = re.compile(r"\b(?:table|tbl\.?)\s*\d+", re.IGNORECASE)


def _caption_text(elem: StructuralElement) -> str:
    """Extract plain text from a caption element's content."""
    c = elem.content
    if isinstance(c, str):
        return c
    if isinstance(c, Caption):
        return c.content
    if hasattr(c, "content") and isinstance(c.content, str):
        return c.content
    return ""


def _classify_caption(elem: StructuralElement) -> str:
    """Return 'figure', 'table', or 'unknown' by parsing the caption text."""
    text = _caption_text(elem)
    if _FIG_PATTERN.search(text):
        return "figure"
    if _TBL_PATTERN.search(text):
        return "table"
    # Also check Caption dataclass type field
    if isinstance(elem.content, Caption):
        ct = elem.content.caption_type.lower()
        if "fig" in ct:
            return "figure"
        if "table" in ct or "tbl" in ct:
            return "table"
    return "unknown"


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _bbox_edge_distance(a: BoundingBox, b: BoundingBox) -> float:
    """
    Shortest distance between the edges of two bounding boxes.
    Returns 0.0 when the boxes overlap or touch.
    """
    h_gap = max(0.0, max(a.x_min, b.x_min) - min(a.x_max, b.x_max))
    v_gap = max(0.0, max(a.y_min, b.y_min) - min(a.y_max, b.y_max))
    return math.sqrt(h_gap ** 2 + v_gap ** 2)


def _find_closest(
    caption: StructuralElement,
    candidates: List[StructuralElement],
) -> Tuple[StructuralElement, float]:
    """Return (element, distance) for the nearest candidate."""
    best = candidates[0]
    best_dist = _bbox_edge_distance(caption.bbox, candidates[0].bbox)
    for c in candidates[1:]:
        d = _bbox_edge_distance(caption.bbox, c.bbox)
        if d < best_dist:
            best, best_dist = c, d
    return best, best_dist


# ── Config & Trace ─────────────────────────────────────────────────────────────

@dataclass
class CaptionLinkerConfig:
    """
    Configuration for CaptionLinker.

    Attributes:
        max_proximity_px: Maximum edge-to-edge distance (pixels) for a link to
            be made. Captions further than this from any figure/table are left
            unlinked.
        same_page_only: When True (default) only link elements on the same page.
    """
    max_proximity_px: float = 150.0
    same_page_only: bool = True


@dataclass
class CaptionLinkerTrace:
    """Diagnostic trace from one CaptionLinker.link() call."""
    captions_found: int = 0
    captions_linked: int = 0
    figures_linked: int = 0
    tables_linked: int = 0


# ── Linker ─────────────────────────────────────────────────────────────────────

class CaptionLinker:
    """
    Links CAPTION elements to spatially adjacent FIGURE or TABLE elements.

    Metadata written on a successful link:

    On the CAPTION element::

        metadata["linked_element_id"]   = figure_or_table.element_id
        metadata["linked_element_type"] = "figure" | "table"
        metadata["link_distance_px"]    = float

    On the FIGURE/TABLE element::

        metadata["caption_id"]   = caption.element_id
        metadata["caption_text"] = str

    Usage::

        linker = CaptionLinker()
        updated_doc, trace = linker.link(doc)
    """

    def __init__(self, config: Optional[CaptionLinkerConfig] = None) -> None:
        self.config = config or CaptionLinkerConfig()

    def link(
        self, doc: DocumentResult
    ) -> Tuple[DocumentResult, CaptionLinkerTrace]:
        """
        Link captions to figures/tables in *doc*.

        Elements are mutated in-place (metadata dicts are updated).
        The same DocumentResult object is returned so callers can chain.
        """
        trace = CaptionLinkerTrace()
        cfg = self.config

        captions = [e for e in doc.elements if e.element_type == ElementType.CAPTION]
        figures = [e for e in doc.elements if e.element_type == ElementType.FIGURE]
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]

        trace.captions_found = len(captions)

        if not captions or (not figures and not tables):
            return doc, trace

        for caption in captions:
            caption_class = _classify_caption(caption)

            if caption_class == "figure":
                pool = figures
            elif caption_class == "table":
                pool = tables
            else:
                pool = figures + tables

            if cfg.same_page_only:
                pool = [c for c in pool if c.page_number == caption.page_number]

            if not pool:
                continue

            best, dist = _find_closest(caption, pool)

            if dist > cfg.max_proximity_px:
                continue

            # Write cross-references into both elements' metadata
            caption.metadata["linked_element_id"] = best.element_id
            caption.metadata["linked_element_type"] = best.element_type.value
            caption.metadata["link_distance_px"] = round(dist, 1)

            best.metadata["caption_id"] = caption.element_id
            best.metadata["caption_text"] = _caption_text(caption)

            trace.captions_linked += 1
            if best.element_type == ElementType.FIGURE:
                trace.figures_linked += 1
            else:
                trace.tables_linked += 1

        return doc, trace
