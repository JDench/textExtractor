"""
Cross-Page Element Coordination

Detects and merges structural elements that span page boundaries in
multi-page documents processed by BatchProcessor.process_batch().

Each process_image() call produces one DocumentResult for that page.
This coordinator inspects consecutive page pairs and attempts to join:

  - TEXT blocks whose content continues across the page break
  - LIST elements where the item sequence continues onto the next page
  - TABLE elements where rows continue onto the next page

Confidence scoring
------------------
For each candidate pair the coordinator computes a score in [0, 1]:

  score >= merge_confidence_threshold
      → Elements are merged. The merged element lives in page-N's document,
        carries metadata["cross_page_merged"]=True, metadata["spans_pages"],
        and metadata["merge_confidence"]. The page-N+1 element is removed
        from its document and its ID is noted in metadata["merged_from_ids"].

  hint_confidence_threshold <= score < merge_confidence_threshold
      → Both elements receive continuation hints in metadata but are NOT
        structurally changed. Content remains fully usable as-is.

  score < hint_confidence_threshold
      → No action. Elements are left completely unchanged.

The coordinator never discards content.  If a merge cannot be completed
(e.g. column-count mismatch on a table), it falls back to hints.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from data_models import (
    BatchResult,
    BoundingBox,
    DocumentResult,
    ElementType,
    ListItem,
    ListStructure,
    StructuralElement,
    TableCell,
    TableStructure,
)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class CrossPageConfig:
    """
    Configuration for CrossPageCoordinator.

    Attributes:
        merge_text:   Attempt to merge TEXT elements across page breaks.
        merge_lists:  Attempt to merge LIST elements across page breaks.
        merge_tables: Attempt to merge TABLE elements across page breaks.
        page_edge_fraction: Fraction of page height that defines the "edge zone"
            for candidate elements (e.g. 0.15 → bottom/top 15% of the page).
        merge_confidence_threshold: Score at or above which elements are merged.
        hint_confidence_threshold:  Score at or above which continuation hints
            are added to metadata without merging.
    """
    merge_text: bool = True
    merge_lists: bool = True
    merge_tables: bool = True
    page_edge_fraction: float = 0.15
    merge_confidence_threshold: float = 0.65
    hint_confidence_threshold: float = 0.35


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class CrossPageTrace:
    """Diagnostic trace from a single CrossPageCoordinator.coordinate() call."""
    page_pairs_examined: int = 0
    text_merges: int = 0
    list_merges: int = 0
    table_merges: int = 0
    continuation_hints_added: int = 0


# ── Coordinator ────────────────────────────────────────────────────────────────

class CrossPageCoordinator:
    """
    Merges cross-page elements in a BatchResult produced by process_batch().

    Usage::

        coordinator = CrossPageCoordinator()
        updated_batch, trace = coordinator.coordinate(batch)

    The returned BatchResult is a new object; the input is not mutated.
    """

    def __init__(self, config: Optional[CrossPageConfig] = None) -> None:
        self.config = config or CrossPageConfig()

    def coordinate(
        self, batch: BatchResult
    ) -> Tuple[BatchResult, CrossPageTrace]:
        """
        Examine consecutive page pairs and merge/annotate continuation elements.

        Args:
            batch: BatchResult where each document represents one page.

        Returns:
            (updated_batch, trace)
        """
        trace = CrossPageTrace()

        if len(batch.documents) < 2:
            return batch, trace

        # Work on shallow-copied documents with fresh element lists so the
        # originals remain unchanged.
        docs: List[DocumentResult] = []
        for doc in batch.documents:
            new_doc = copy.copy(doc)
            new_doc.elements = list(doc.elements)
            docs.append(new_doc)

        cfg = self.config

        for i in range(len(docs) - 1):
            page_doc = docs[i]
            next_doc = docs[i + 1]
            trace.page_pairs_examined += 1

            page_h = float(page_doc.metadata.image_dimensions[1])
            next_h = float(next_doc.metadata.image_dimensions[1])

            if cfg.merge_text:
                n = self._coordinate_text(page_doc, next_doc, page_h, next_h, trace)
                trace.text_merges += n

            if cfg.merge_lists:
                n = self._coordinate_lists(page_doc, next_doc, page_h, next_h, trace)
                trace.list_merges += n

            if cfg.merge_tables:
                n = self._coordinate_tables(page_doc, next_doc, page_h, next_h, trace)
                trace.table_merges += n

        new_batch = BatchResult(
            batch_id=batch.batch_id,
            created_at=batch.created_at,
            documents=docs,
            batch_config=batch.batch_config,
        )
        return new_batch, trace

    # ── Text coordination ──────────────────────────────────────────────────────

    def _coordinate_text(
        self,
        page_doc: DocumentResult,
        next_doc: DocumentResult,
        page_h: float,
        next_h: float,
        trace: CrossPageTrace,
    ) -> int:
        cfg = self.config
        trailing = _trailing_elem(page_doc, ElementType.TEXT, page_h, cfg.page_edge_fraction)
        leading = _leading_elem(next_doc, ElementType.TEXT, next_h, cfg.page_edge_fraction)

        if trailing is None or leading is None:
            return 0
        if not isinstance(trailing.content, str) or not isinstance(leading.content, str):
            return 0

        score = _text_score(trailing, leading)

        if score >= cfg.merge_confidence_threshold:
            self._merge_text(trailing, leading, next_doc, score)
            return 1

        if score >= cfg.hint_confidence_threshold:
            _add_hints(trailing, leading, page_doc, next_doc)
            trace.continuation_hints_added += 1

        return 0

    @staticmethod
    def _merge_text(
        trailing: StructuralElement,
        leading: StructuralElement,
        next_doc: DocumentResult,
        score: float,
    ) -> None:
        combined = (trailing.content + " " + leading.content).strip()
        trailing.content = combined
        trailing.metadata.update({
            "cross_page_merged": True,
            "spans_pages": [trailing.page_number, leading.page_number],
            "merge_confidence": round(score, 3),
            "merged_from_ids": [leading.element_id],
        })
        next_doc.elements = [e for e in next_doc.elements
                             if e.element_id != leading.element_id]

    # ── List coordination ──────────────────────────────────────────────────────

    def _coordinate_lists(
        self,
        page_doc: DocumentResult,
        next_doc: DocumentResult,
        page_h: float,
        next_h: float,
        trace: CrossPageTrace,
    ) -> int:
        cfg = self.config
        trailing = _trailing_elem(page_doc, ElementType.LIST, page_h, cfg.page_edge_fraction)
        leading = _leading_elem(next_doc, ElementType.LIST, next_h, cfg.page_edge_fraction)

        if trailing is None or leading is None:
            return 0
        if not isinstance(trailing.content, ListStructure) or not isinstance(leading.content, ListStructure):
            return 0

        score = _list_score(trailing, leading)

        if score >= cfg.merge_confidence_threshold:
            self._merge_lists(trailing, leading, next_doc, score)
            return 1

        if score >= cfg.hint_confidence_threshold:
            _add_hints(trailing, leading, page_doc, next_doc)
            trace.continuation_hints_added += 1

        return 0

    @staticmethod
    def _merge_lists(
        trailing: StructuralElement,
        leading: StructuralElement,
        next_doc: DocumentResult,
        score: float,
    ) -> None:
        t_lst: ListStructure = trailing.content
        l_lst: ListStructure = leading.content

        merged = ListStructure(
            items=list(t_lst.items) + list(l_lst.items),
            root_item_ids=list(t_lst.root_item_ids) + list(l_lst.root_item_ids),
            bbox=t_lst.bbox,
            confidence=min(t_lst.confidence, l_lst.confidence),
            list_type=t_lst.list_type,
        )
        trailing.content = merged
        trailing.metadata.update({
            "cross_page_merged": True,
            "spans_pages": [trailing.page_number, leading.page_number],
            "merge_confidence": round(score, 3),
            "merged_from_ids": [leading.element_id],
        })
        next_doc.elements = [e for e in next_doc.elements
                             if e.element_id != leading.element_id]

    # ── Table coordination ─────────────────────────────────────────────────────

    def _coordinate_tables(
        self,
        page_doc: DocumentResult,
        next_doc: DocumentResult,
        page_h: float,
        next_h: float,
        trace: CrossPageTrace,
    ) -> int:
        cfg = self.config
        trailing = _trailing_elem(page_doc, ElementType.TABLE, page_h, cfg.page_edge_fraction)
        leading = _leading_elem(next_doc, ElementType.TABLE, next_h, cfg.page_edge_fraction)

        if trailing is None or leading is None:
            return 0
        if not isinstance(trailing.content, TableStructure) or not isinstance(leading.content, TableStructure):
            return 0

        score = _table_score(trailing, leading)

        if score >= cfg.merge_confidence_threshold:
            merged = self._merge_tables(trailing, leading, next_doc, score)
            if merged:
                return 1
            # Column mismatch — can't safely merge; fall back to hints
            _add_hints(trailing, leading, page_doc, next_doc)
            trace.continuation_hints_added += 1
            return 0

        if score >= cfg.hint_confidence_threshold:
            _add_hints(trailing, leading, page_doc, next_doc)
            trace.continuation_hints_added += 1

        return 0

    @staticmethod
    def _merge_tables(
        trailing: StructuralElement,
        leading: StructuralElement,
        next_doc: DocumentResult,
        score: float,
    ) -> bool:
        """Merge table rows from *leading* into *trailing*. Returns True on success."""
        t_tbl: TableStructure = trailing.content
        l_tbl: TableStructure = leading.content

        if t_tbl.num_cols != l_tbl.num_cols:
            return False  # incompatible tables

        row_offset = t_tbl.num_rows
        extra_cells: List[TableCell] = [
            TableCell(
                content=cell.content,
                row_index=cell.row_index + row_offset,
                col_index=cell.col_index,
                bbox=cell.bbox,
                confidence=cell.confidence,
                colspan=cell.colspan,
                rowspan=cell.rowspan,
                is_header=False,
            )
            for cell in l_tbl.cells
        ]

        merged_tbl = TableStructure(
            cells=list(t_tbl.cells) + extra_cells,
            bbox=t_tbl.bbox,
            confidence=min(t_tbl.confidence, l_tbl.confidence),
            headers=t_tbl.headers,
        )
        trailing.content = merged_tbl
        trailing.metadata.update({
            "cross_page_merged": True,
            "spans_pages": [trailing.page_number, leading.page_number],
            "merge_confidence": round(score, 3),
            "merged_from_ids": [leading.element_id],
        })
        next_doc.elements = [e for e in next_doc.elements
                             if e.element_id != leading.element_id]
        return True


# ── Module-level helpers ───────────────────────────────────────────────────────

def _trailing_elem(
    doc: DocumentResult,
    etype: ElementType,
    page_h: float,
    edge_fraction: float,
) -> Optional[StructuralElement]:
    """Return the element of *etype* closest to the bottom page edge, or None."""
    threshold = page_h * (1 - edge_fraction)
    candidates = [
        e for e in doc.elements
        if e.element_type == etype and e.bbox.y_max > threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.bbox.y_max)


def _leading_elem(
    doc: DocumentResult,
    etype: ElementType,
    page_h: float,
    edge_fraction: float,
) -> Optional[StructuralElement]:
    """Return the element of *etype* closest to the top page edge, or None."""
    threshold = page_h * edge_fraction
    candidates = [
        e for e in doc.elements
        if e.element_type == etype and e.bbox.y_min < threshold
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.bbox.y_min)


def _text_score(trailing: StructuralElement, leading: StructuralElement) -> float:
    """Score probability that trailing TEXT continues in leading TEXT."""
    score = 0.5  # base: both are near page edges

    text_a: str = trailing.content if isinstance(trailing.content, str) else ""
    text_b: str = leading.content if isinstance(leading.content, str) else ""

    # No sentence-ending punctuation → text probably continues
    if text_a and text_a.rstrip()[-1:] not in ".!?:":
        score += 0.20

    # Continuation starts with lowercase
    if text_b and text_b.lstrip()[:1].islower():
        score += 0.20

    # Similar column width (same paragraph column)
    wa = trailing.bbox.width()
    wb = leading.bbox.width()
    if max(wa, wb) > 0 and min(wa, wb) / max(wa, wb) > 0.80:
        score += 0.10

    return min(1.0, score)


def _list_score(trailing: StructuralElement, leading: StructuralElement) -> float:
    """Score probability that trailing LIST continues in leading LIST."""
    score = 0.5

    t_lst = trailing.content if isinstance(trailing.content, ListStructure) else None
    l_lst = leading.content if isinstance(leading.content, ListStructure) else None

    if t_lst is None or l_lst is None:
        return score

    # Same list type
    if t_lst.list_type == l_lst.list_type:
        score += 0.20

    # Numbered list: consecutive numbering
    if t_lst.list_type == "number" and l_lst.list_type == "number":
        last_nums = [item.number for item in t_lst.items if item.number is not None]
        first_nums = [item.number for item in l_lst.items if item.number is not None]
        if last_nums and first_nums and first_nums[0] == last_nums[-1] + 1:
            score += 0.25

    return min(1.0, score)


def _table_score(trailing: StructuralElement, leading: StructuralElement) -> float:
    """Score probability that trailing TABLE rows continue in leading TABLE."""
    score = 0.5

    t_tbl = trailing.content if isinstance(trailing.content, TableStructure) else None
    l_tbl = leading.content if isinstance(leading.content, TableStructure) else None

    if t_tbl is None or l_tbl is None:
        return score

    # Same column count → strong signal
    if t_tbl.num_cols == l_tbl.num_cols:
        score += 0.30
    else:
        score -= 0.20  # mismatch → probably different tables

    # Leading table has no header row (continuation pages don't repeat headers)
    if l_tbl.num_rows > 0 and not any(cell.is_header for cell in l_tbl.get_row(0)):
        score += 0.10

    # Similar widths
    wt = trailing.bbox.width()
    wl = leading.bbox.width()
    if max(wt, wl) > 0 and min(wt, wl) / max(wt, wl) > 0.85:
        score += 0.10

    return min(1.0, score)


def _add_hints(
    trailing: StructuralElement,
    leading: StructuralElement,
    page_doc: DocumentResult,
    next_doc: DocumentResult,
) -> None:
    """Add continuation hints to both elements without merging them."""
    trailing.metadata["possible_continuation_on_page"] = next_doc.metadata.source_file
    trailing.metadata["possible_continuation_page_number"] = leading.page_number
    leading.metadata["possible_continuation_from_page"] = page_doc.metadata.source_file
    leading.metadata["possible_continuation_page_number"] = trailing.page_number
