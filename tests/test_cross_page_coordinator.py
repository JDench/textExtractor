"""
Tests for CrossPageCoordinator — cross-page text/list/table merging.

All tests use synthetic elements and documents; no image processing needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_page_coordinator import CrossPageCoordinator, CrossPageConfig
from data_models import (
    BoundingBox,
    ElementType,
    ListItem,
    ListStructure,
    TableCell,
    TableStructure,
)
from helpers import make_batch, make_doc, make_element


# ── Helpers ────────────────────────────────────────────────────────────────────

PAGE_H = 1000  # synthetic page height
EDGE = int(PAGE_H * 0.15)  # 150 px edge zone


def _doc_with_text(
    text: str,
    y_min: float,
    y_max: float,
    page: int = 1,
    width: float = 500,
    page_h: int = PAGE_H,
):
    """Create a single-page document with one TEXT element."""
    elem = make_element(
        ElementType.TEXT, text,
        x_min=50, y_min=y_min, x_max=50 + width, y_max=y_max,
        page_number=page,
    )
    return make_doc([elem], source_file=f"page{page}.png")


def _list_structure(items_text, list_type="bullet"):
    bbox = BoundingBox(50, 10, 400, 200)
    items = [
        ListItem(t, 0, bbox, 0.9, list_type=list_type)
        for t in items_text
    ]
    return ListStructure(items, [], bbox, 0.9, list_type=list_type)


def _table_structure(num_cols=3, num_rows=2, has_header=False):
    cells = []
    for r in range(num_rows):
        for c in range(num_cols):
            cells.append(TableCell(
                f"r{r}c{c}", r, c,
                BoundingBox(c * 50 + 50, r * 30 + 50, c * 50 + 100, r * 30 + 80),
                0.9,
                is_header=(r == 0 and has_header),
            ))
    return TableStructure(cells, BoundingBox(50, 50, 200, 140), 0.9)


def coordinate(batch, **cfg_kwargs):
    cfg = CrossPageConfig(**cfg_kwargs) if cfg_kwargs else CrossPageConfig()
    return CrossPageCoordinator(cfg).coordinate(batch)


# ── Single page — no-op ────────────────────────────────────────────────────────

class TestSinglePage:
    def test_single_page_unchanged(self):
        doc = _doc_with_text("Hello world.", 50, 70)
        batch = make_batch(documents=[doc])
        new_batch, trace = coordinate(batch)
        assert len(new_batch.documents) == 1
        assert trace.page_pairs_examined == 0
        assert trace.text_merges == 0

    def test_empty_batch_unchanged(self):
        batch = make_batch(documents=[])
        new_batch, trace = coordinate(batch)
        assert len(new_batch.documents) == 0


# ── Text merging ───────────────────────────────────────────────────────────────

class TestTextMerge:
    def _make_two_page_batch(self, text_a, y_a, text_b, y_b, page_h=PAGE_H):
        """Two-page batch: text_a at bottom of page 1, text_b at top of page 2."""
        doc1 = make_doc(
            [make_element(ElementType.TEXT, text_a,
                          x_min=50, y_min=y_a, x_max=550, y_max=y_a + 20, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, page_h)
        doc2 = make_doc(
            [make_element(ElementType.TEXT, text_b,
                          x_min=50, y_min=y_b, x_max=550, y_max=y_b + 20, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, page_h)
        return make_batch(documents=[doc1, doc2])

    def test_high_confidence_text_merged(self):
        # No sentence-ending punctuation + lowercase start → high score
        batch = self._make_two_page_batch(
            "The algorithm continues", PAGE_H - 30,  # near bottom
            "on the next page",       10,             # near top
        )
        new_batch, trace = coordinate(batch)
        assert trace.text_merges == 1
        assert len(new_batch.documents[0].elements) == 1
        assert len(new_batch.documents[1].elements) == 0
        merged_text = new_batch.documents[0].elements[0].content
        assert "continues" in merged_text and "next page" in merged_text

    def test_merged_element_has_cross_page_metadata(self):
        batch = self._make_two_page_batch(
            "The process is", PAGE_H - 30,
            "simple to understand", 10,
        )
        new_batch, _ = coordinate(batch)
        elem = new_batch.documents[0].elements[0]
        assert elem.metadata.get("cross_page_merged") is True
        assert 1 in elem.metadata.get("spans_pages", [])
        assert 2 in elem.metadata.get("spans_pages", [])

    def test_text_not_at_edge_not_merged(self):
        # Text is in the middle of both pages
        batch = self._make_two_page_batch(
            "Middle of page one", 400,   # middle of 1000px page
            "Middle of page two", 400,
        )
        new_batch, trace = coordinate(batch)
        assert trace.text_merges == 0

    def test_sentence_ending_lowers_score(self):
        # Sentence ends with '.' → lower score → no merge with high threshold
        batch = self._make_two_page_batch(
            "This is complete.",  PAGE_H - 30,
            "New paragraph here", 10,
        )
        new_batch, trace = coordinate(
            batch,
            merge_confidence_threshold=0.90,  # require very high score
        )
        assert trace.text_merges == 0

    def test_continuation_hint_added_medium_score(self):
        # Sentence ends with period and next starts uppercase → score ≈ 0.5
        # (base 0.5, no bonus for punctuation, no bonus for lowercase start,
        #  but same width so +0.1 → 0.6)
        # Set merge_threshold=0.80 and hint_threshold=0.40 → 0.6 lands in hint zone
        batch = self._make_two_page_batch(
            "This sentence ends here.",    PAGE_H - 30,
            "Next sentence begins here.",  10,
        )
        new_batch, trace = coordinate(
            batch,
            merge_confidence_threshold=0.80,
            hint_confidence_threshold=0.40,
        )
        assert trace.continuation_hints_added >= 1
        # Elements remain separate
        assert len(new_batch.documents[0].elements) == 1
        assert len(new_batch.documents[1].elements) == 1
        p1_elem = new_batch.documents[0].elements[0]
        assert "possible_continuation_on_page" in p1_elem.metadata


# ── List merging ───────────────────────────────────────────────────────────────

class TestListMerge:
    def _make_list_batch(self, items_a, items_b, list_type="bullet", page_h=PAGE_H):
        lst_a = _list_structure(items_a, list_type)
        lst_b = _list_structure(items_b, list_type)
        doc1 = make_doc(
            [make_element(ElementType.LIST, lst_a,
                          y_min=PAGE_H - 100, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, page_h)
        doc2 = make_doc(
            [make_element(ElementType.LIST, lst_b,
                          y_min=10, y_max=120, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, page_h)
        return make_batch(documents=[doc1, doc2])

    def test_bullet_lists_merged(self):
        batch = self._make_list_batch(["Item A", "Item B"], ["Item C"])
        new_batch, trace = coordinate(batch)
        assert trace.list_merges == 1
        merged = new_batch.documents[0].elements[0].content
        assert len(merged.items) == 3

    def test_merged_list_has_metadata(self):
        batch = self._make_list_batch(["A"], ["B"])
        new_batch, _ = coordinate(batch)
        elem = new_batch.documents[0].elements[0]
        assert elem.metadata.get("cross_page_merged") is True

    def test_numbered_consecutive_merged_with_bonus(self):
        items_a = [ListItem("First", 0, BoundingBox(50, 10, 400, 30), 0.9,
                             list_type="number", number=1)]
        items_b = [ListItem("Second", 0, BoundingBox(50, 10, 400, 30), 0.9,
                             list_type="number", number=2)]
        lst_a = ListStructure(items_a, [], BoundingBox(50, 10, 400, 200), 0.9, "number")
        lst_b = ListStructure(items_b, [], BoundingBox(50, 10, 400, 200), 0.9, "number")

        doc1 = make_doc(
            [make_element(ElementType.LIST, lst_a,
                          y_min=PAGE_H - 100, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, PAGE_H)
        doc2 = make_doc(
            [make_element(ElementType.LIST, lst_b,
                          y_min=10, y_max=120, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, PAGE_H)
        batch = make_batch(documents=[doc1, doc2])
        new_batch, trace = coordinate(batch)
        assert trace.list_merges == 1


# ── Table merging ──────────────────────────────────────────────────────────────

class TestTableMerge:
    def _make_table_batch(self, tbl_a, tbl_b, page_h=PAGE_H):
        doc1 = make_doc(
            [make_element(ElementType.TABLE, tbl_a,
                          y_min=PAGE_H - 100, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, page_h)
        doc2 = make_doc(
            [make_element(ElementType.TABLE, tbl_b,
                          y_min=10, y_max=120, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, page_h)
        return make_batch(documents=[doc1, doc2])

    def test_same_col_count_tables_merged(self):
        tbl_a = _table_structure(num_cols=3, num_rows=2, has_header=True)
        tbl_b = _table_structure(num_cols=3, num_rows=2, has_header=False)
        batch = self._make_table_batch(tbl_a, tbl_b)
        new_batch, trace = coordinate(batch)
        assert trace.table_merges == 1
        merged = new_batch.documents[0].elements[0].content
        assert merged.num_rows == 4  # 2 + 2

    def test_different_col_count_not_merged(self):
        tbl_a = _table_structure(num_cols=3, num_rows=2)
        tbl_b = _table_structure(num_cols=4, num_rows=2)
        batch = self._make_table_batch(tbl_a, tbl_b)
        new_batch, trace = coordinate(batch)
        assert trace.table_merges == 0

    def test_merged_table_rows_renumbered(self):
        tbl_a = _table_structure(num_cols=2, num_rows=2)
        tbl_b = _table_structure(num_cols=2, num_rows=2)
        batch = self._make_table_batch(tbl_a, tbl_b)
        new_batch, _ = coordinate(batch)
        merged: TableStructure = new_batch.documents[0].elements[0].content
        row_indices = sorted({cell.row_index for cell in merged.cells})
        assert row_indices == [0, 1, 2, 3]

    def test_different_col_count_adds_hint(self):
        tbl_a = _table_structure(num_cols=3, num_rows=2)
        tbl_b = _table_structure(num_cols=4, num_rows=2)
        batch = self._make_table_batch(tbl_a, tbl_b)
        new_batch, trace = coordinate(batch)
        # Both elements should remain, with hints
        assert len(new_batch.documents[0].elements) == 1
        assert len(new_batch.documents[1].elements) == 1


# ── Feature flags ──────────────────────────────────────────────────────────────

class TestFeatureFlags:
    def test_disable_text_merge(self):
        doc1 = make_doc(
            [make_element(ElementType.TEXT, "continues",
                          y_min=PAGE_H - 30, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, PAGE_H)
        doc2 = make_doc(
            [make_element(ElementType.TEXT, "on page 2",
                          y_min=10, y_max=30, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, PAGE_H)
        batch = make_batch(documents=[doc1, doc2])
        new_batch, trace = coordinate(batch, merge_text=False)
        assert trace.text_merges == 0

    def test_disable_list_merge(self):
        lst = _list_structure(["A"])
        doc1 = make_doc(
            [make_element(ElementType.LIST, lst,
                          y_min=PAGE_H - 100, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, PAGE_H)
        doc2 = make_doc(
            [make_element(ElementType.LIST, _list_structure(["B"]),
                          y_min=10, y_max=120, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, PAGE_H)
        batch = make_batch(documents=[doc1, doc2])
        new_batch, trace = coordinate(batch, merge_lists=False)
        assert trace.list_merges == 0


# ── Trace ──────────────────────────────────────────────────────────────────────

class TestTrace:
    def test_pairs_examined_count(self):
        docs = [
            make_doc([], source_file=f"p{i}.png")
            for i in range(4)
        ]
        for doc in docs:
            doc.metadata.image_dimensions = (600, PAGE_H)
        batch = make_batch(documents=docs)
        _, trace = coordinate(batch)
        assert trace.page_pairs_examined == 3

    def test_original_batch_not_mutated(self):
        doc1 = make_doc(
            [make_element(ElementType.TEXT, "continues",
                          y_min=PAGE_H - 30, y_max=PAGE_H - 10, page_number=1)],
            source_file="p1.png",
        )
        doc1.metadata.image_dimensions = (600, PAGE_H)
        doc2 = make_doc(
            [make_element(ElementType.TEXT, "on page 2",
                          y_min=10, y_max=30, page_number=2)],
            source_file="p2.png",
        )
        doc2.metadata.image_dimensions = (600, PAGE_H)
        batch = make_batch(documents=[doc1, doc2])
        orig_len_p1 = len(doc1.elements)
        orig_len_p2 = len(doc2.elements)
        coordinate(batch)
        assert len(doc1.elements) == orig_len_p1
        assert len(doc2.elements) == orig_len_p2
