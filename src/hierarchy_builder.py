"""
Element Hierarchy Builder

Post-processing utility that wires parent-child relationships between
StructuralElements after all detectors have run.

Two passes are performed:

Pass 1 — Heading hierarchy
    Elements are sorted into reading order (page → y → x).  A stack of open
    headings is maintained.  Each HEADING element becomes a child of the
    nearest enclosing heading at a strictly higher level.  Every non-heading
    element (TEXT, LIST, TABLE, FIGURE, FORMULA, …) that does not already
    have a parent_id is assigned to the deepest open heading on the stack.

    Heading level is read from metadata["heading_level"] when present (set
    by TextDetector), with a fallback of nesting_level + 1.

Pass 2 — TOC → Heading cross-link
    Each TABLE_OF_CONTENTS entry whose target_heading_id is still None is
    matched against HEADING elements.  Matching is done by normalized string
    containment: the TOC title (lowercased, punctuation removed) must appear
    as a substring of the heading text, or vice versa.  The first match is
    used.

The input list is mutated in-place and also returned for chaining.

Design notes
------------
- Existing parent_id values are respected: this builder never overwrites a
  link that a detector already established (e.g., caption→figure, page_num→header).
- Elements with element_type HEADER, FOOTER, PAGE_NUMBER, CAPTION keep their
  detector-assigned relationships.
- The builder is idempotent: running it twice on the same list is safe.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data_models import ElementType, StructuralElement, TableOfContents

import logging
logger = logging.getLogger(__name__)


# Element types that should never be auto-parented by the heading hierarchy
# because they have their own explicit linking logic.
_SKIP_PARENT_TYPES = {
    ElementType.HEADER,
    ElementType.FOOTER,
    ElementType.PAGE_NUMBER,
    ElementType.CAPTION,        # linked to figure/table by detectors
}


@dataclass
class HierarchyConfig:
    """Configuration for HierarchyBuilder."""

    # Whether to link non-heading elements to the current heading
    link_content_to_headings: bool = True

    # Whether to cross-link TOC entries to matching HEADING elements
    link_toc_to_headings: bool = True

    # Minimum character overlap fraction for a TOC→heading fuzzy match
    # (normalised title must be a substring of the heading, or vice versa)
    toc_match_min_overlap: float = 0.80


@dataclass
class HierarchyBuildTrace:
    """Records what HierarchyBuilder.build() did."""
    elements_processed: int = 0
    heading_links_made: int = 0
    content_links_made: int = 0
    toc_links_made: int = 0
    config: Optional[HierarchyConfig] = None


class HierarchyBuilder:
    """
    Builds parent-child relationships between StructuralElements.

    Usage::

        builder = HierarchyBuilder()
        elements, trace = builder.build(all_elements)

    Args:
        config: HierarchyConfig; defaults used when None.
    """

    def __init__(self, config: Optional[HierarchyConfig] = None) -> None:
        self.config = config or HierarchyConfig()

    def build(
        self,
        elements: List[StructuralElement],
    ) -> Tuple[List[StructuralElement], HierarchyBuildTrace]:
        """
        Wire parent-child relationships into the element list.

        The list is mutated in-place.  Returns the same list plus a trace.
        """
        cfg = self.config
        trace = HierarchyBuildTrace(config=cfg, elements_processed=len(elements))

        if not elements:
            return elements, trace

        # Build a fast lookup map
        elem_map: Dict[str, StructuralElement] = {e.element_id: e for e in elements}

        # ── Pass 1: heading hierarchy ──────────────────────────────────────────
        # Reading order: page → top-to-bottom → left-to-right
        sorted_elems = sorted(
            elements,
            key=lambda e: (e.page_number, e.bbox.y_min, e.bbox.x_min),
        )

        # Stack entries: (heading_level: int, element_id: str)
        # Lower heading_level number = higher in the document hierarchy
        heading_stack: List[Tuple[int, str]] = []

        for elem in sorted_elems:
            if elem.element_type == ElementType.HEADING:
                h_level = self._heading_level(elem)

                # Pop headings of equal or deeper level from the stack
                while heading_stack and heading_stack[-1][0] >= h_level:
                    heading_stack.pop()

                # Parent this heading to the top of the stack (if any)
                if heading_stack and elem.parent_id is None:
                    parent_id = heading_stack[-1][1]
                    elem.parent_id = parent_id
                    parent_elem = elem_map.get(parent_id)
                    if parent_elem and elem.element_id not in parent_elem.child_ids:
                        parent_elem.child_ids.append(elem.element_id)
                    trace.heading_links_made += 1

                heading_stack.append((h_level, elem.element_id))

            elif cfg.link_content_to_headings:
                if elem.element_type in _SKIP_PARENT_TYPES:
                    continue
                if elem.parent_id is not None:
                    continue  # already linked by a detector
                if heading_stack:
                    parent_id = heading_stack[-1][1]
                    elem.parent_id = parent_id
                    parent_elem = elem_map.get(parent_id)
                    if parent_elem and elem.element_id not in parent_elem.child_ids:
                        parent_elem.child_ids.append(elem.element_id)
                    trace.content_links_made += 1

        # ── Pass 2: TOC → heading cross-link ──────────────────────────────────
        if cfg.link_toc_to_headings:
            headings = [e for e in elements if e.element_type == ElementType.HEADING]
            toc_entries = [
                e for e in elements
                if e.element_type == ElementType.TABLE_OF_CONTENTS
                and isinstance(e.content, TableOfContents)
                and e.content.target_heading_id is None
            ]
            for toc_elem in toc_entries:
                toc_title = _normalize(toc_elem.content.title)  # type: ignore[union-attr]
                match = self._find_heading_match(toc_title, headings, cfg.toc_match_min_overlap)
                if match is not None:
                    toc_elem.content.target_heading_id = match.element_id  # type: ignore[union-attr]
                    trace.toc_links_made += 1

        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _heading_level(elem: StructuralElement) -> int:
        """Return the numeric heading level (1=H1, 2=H2, …) for a HEADING element."""
        level = elem.metadata.get("heading_level")
        if isinstance(level, int) and 1 <= level <= 6:
            return level
        # Fallback: nesting_level (0=H1 in TextDetector convention)
        return max(1, elem.nesting_level + 1)

    @staticmethod
    def _find_heading_match(
        toc_title_norm: str,
        headings: List[StructuralElement],
        min_overlap: float,
    ) -> Optional[StructuralElement]:
        """
        Find the HEADING element whose text best matches the TOC title.

        Uses normalised substring containment with a minimum character
        overlap fraction to avoid spurious short-string matches.
        """
        best: Optional[StructuralElement] = None
        best_score = 0.0
        for h in headings:
            h_text = _normalize(h.content if isinstance(h.content, str) else str(h.content))
            if not h_text or not toc_title_norm:
                continue
            shorter = min(len(toc_title_norm), len(h_text))
            if shorter == 0:
                continue
            if toc_title_norm in h_text or h_text in toc_title_norm:
                score = shorter / max(len(toc_title_norm), len(h_text))
                if score >= min_overlap and score > best_score:
                    best_score = score
                    best = h
        return best


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
