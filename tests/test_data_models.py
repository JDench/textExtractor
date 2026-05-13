"""
Tests for data_models.py — validation, geometry helpers, and utility methods.
"""

import pytest
from datetime import datetime

from data_models import (
    Annotation,
    Barcode,
    BatchResult,
    BlockQuote,
    BoundingBox,
    Caption,
    CodeBlock,
    ConfidenceLevel,
    Coordinates,
    DocumentMetadata,
    DocumentResult,
    ElementType,
    FigureRegion,
    FormulaExpression,
    IndexEntry,
    ListItem,
    ListStructure,
    OCRTextResult,
    ProcessingStatus,
    Reference,
    StructuralElement,
    TableCell,
    TableOfContents,
    TableStructure,
    Watermark,
)
from helpers import make_bbox, make_element, make_doc, make_batch


# ── ConfidenceLevel ────────────────────────────────────────────────────────────

class TestConfidenceLevel:
    def test_classifications(self):
        assert ConfidenceLevel.from_score(0.1) == ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.from_score(0.45) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.70) == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.from_score(0.88) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.97) == ConfidenceLevel.VERY_HIGH

    def test_boundary_values(self):
        assert ConfidenceLevel.from_score(0.0) == ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.from_score(1.0) == ConfidenceLevel.VERY_HIGH

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ConfidenceLevel.from_score(1.5)
        with pytest.raises(ValueError):
            ConfidenceLevel.from_score(-0.1)


# ── Coordinates ────────────────────────────────────────────────────────────────

class TestCoordinates:
    def test_valid(self):
        c = Coordinates(10.0, 20.0)
        assert c.x == 10.0 and c.y == 20.0

    def test_zero_valid(self):
        Coordinates(0, 0)  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            Coordinates(-1, 10)
        with pytest.raises(ValueError):
            Coordinates(10, -1)


# ── BoundingBox ────────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_dimensions(self):
        b = BoundingBox(10, 20, 110, 70)
        assert b.width() == 100
        assert b.height() == 50
        assert b.area() == 5000

    def test_contains_point(self):
        b = BoundingBox(10, 10, 100, 100)
        assert b.contains_point(50, 50)
        assert b.contains_point(10, 10)   # boundary
        assert not b.contains_point(5, 50)

    def test_intersection_overlap(self):
        b1 = BoundingBox(0, 0, 100, 100)
        b2 = BoundingBox(50, 50, 150, 150)
        inter = b1.intersection(b2)
        assert inter is not None
        assert inter.x_min == 50 and inter.y_min == 50
        assert inter.x_max == 100 and inter.y_max == 100

    def test_intersection_none_when_no_overlap(self):
        b1 = BoundingBox(0, 0, 50, 50)
        b2 = BoundingBox(60, 60, 100, 100)
        assert b1.intersection(b2) is None

    def test_union(self):
        b1 = BoundingBox(0, 0, 50, 50)
        b2 = BoundingBox(40, 40, 100, 100)
        u = b1.union(b2)
        assert u.x_min == 0 and u.y_min == 0
        assert u.x_max == 100 and u.y_max == 100

    def test_overlap_percentage(self):
        b1 = BoundingBox(0, 0, 100, 100)
        b2 = BoundingBox(0, 0, 50, 100)  # half of b1
        pct = b1.overlap_percentage(b2)
        assert abs(pct - 0.5) < 1e-6

    def test_overlap_percentage_no_overlap(self):
        b1 = BoundingBox(0, 0, 50, 50)
        b2 = BoundingBox(60, 60, 100, 100)
        assert b1.overlap_percentage(b2) == 0.0

    # ── Validation rejections ──────────────────────────────────────────────────

    def test_rejects_negative_coords(self):
        with pytest.raises(ValueError):
            BoundingBox(-1, 0, 100, 50)

    def test_rejects_x_min_ge_x_max(self):
        with pytest.raises(ValueError):
            BoundingBox(100, 0, 50, 50)
        with pytest.raises(ValueError):
            BoundingBox(50, 0, 50, 50)  # equal also invalid

    def test_rejects_y_min_ge_y_max(self):
        with pytest.raises(ValueError):
            BoundingBox(0, 100, 50, 50)

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValueError):
            BoundingBox(0, 0, 100, 50, confidence=1.5)


# ── TableCell ─────────────────────────────────────────────────────────────────

class TestTableCell:
    def test_valid(self):
        cell = TableCell("A", 0, 0, make_bbox(), 0.9, colspan=2)
        assert cell.colspan == 2

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError):
            TableCell("A", -1, 0, make_bbox(), 0.9)

    def test_rejects_colspan_zero(self):
        with pytest.raises(ValueError):
            TableCell("A", 0, 0, make_bbox(), 0.9, colspan=0)

    def test_rejects_confidence_over_one(self):
        with pytest.raises(ValueError):
            TableCell("A", 0, 0, make_bbox(), 1.5)


# ── TableStructure ─────────────────────────────────────────────────────────────

def _make_table():
    bbox = BoundingBox(0, 0, 200, 100)
    cells = [
        TableCell("H1", 0, 0, BoundingBox(0, 0, 50, 30), 0.95, is_header=True),
        TableCell("H2", 0, 1, BoundingBox(50, 0, 100, 30), 0.95, is_header=True),
        TableCell("D1", 1, 0, BoundingBox(0, 30, 50, 60), 0.90),
        TableCell("D2", 1, 1, BoundingBox(50, 30, 100, 60), 0.90),
    ]
    return TableStructure(cells, bbox, 0.92)


class TestTableStructure:
    def test_dimensions(self):
        t = _make_table()
        assert t.num_rows == 2
        assert t.num_cols == 2

    def test_get_cell(self):
        t = _make_table()
        assert t.get_cell(0, 0).content == "H1"
        assert t.get_cell(1, 1).content == "D2"
        assert t.get_cell(5, 5) is None

    def test_get_row(self):
        t = _make_table()
        row = t.get_row(0)
        assert len(row) == 2
        assert row[0].content == "H1"

    def test_get_column(self):
        t = _make_table()
        col = t.get_column(1)
        assert len(col) == 2
        assert col[0].content == "H2"

    def test_to_2d_array(self):
        t = _make_table()
        arr = t.to_2d_array()
        assert arr[0][0] == "H1"
        assert arr[1][1] == "D2"

    def test_to_2d_array_empty(self):
        t = TableStructure([], BoundingBox(0, 0, 10, 10), 0.5)
        assert t.to_2d_array() == []

    def test_to_markdown(self):
        t = _make_table()
        md = t.to_markdown()
        assert "H1" in md
        assert "---" in md

    def test_to_csv(self):
        t = _make_table()
        csv_text = t.to_csv()
        assert "H1" in csv_text
        assert "D2" in csv_text

    def test_merged_cell_sets_irregular(self):
        bbox = BoundingBox(0, 0, 200, 100)
        cells = [
            TableCell("A", 0, 0, BoundingBox(0, 0, 50, 30), 0.9, colspan=2),
        ]
        t = TableStructure(cells, bbox, 0.9)
        assert t.has_irregular_structure is True


# ── StructuralElement ─────────────────────────────────────────────────────────

class TestStructuralElement:
    def test_valid(self):
        e = make_element(ElementType.HEADING, "Title")
        assert e.element_type == ElementType.HEADING

    def test_to_dict(self):
        e = make_element(ElementType.TEXT, "para", element_id="e1")
        d = e.to_dict()
        assert d["element_id"] == "e1"
        assert d["element_type"] == "text"
        assert d["confidence"] == 0.90

    def test_to_json_is_valid(self):
        import json
        e = make_element(ElementType.TEXT, "para")
        data = json.loads(e.to_json())
        assert "element_id" in data

    def test_in_region(self):
        e = make_element(ElementType.TEXT, "x", x_min=10, y_min=10, x_max=50, y_max=40)
        inside = BoundingBox(0, 0, 100, 100)
        outside = BoundingBox(0, 0, 20, 20)
        assert e.in_region(inside)
        assert not e.in_region(outside)

    def test_overlaps_with(self):
        e1 = make_element(ElementType.TEXT, "x", x_min=0, y_min=0, x_max=60, y_max=40)
        e2 = make_element(ElementType.TEXT, "y", x_min=40, y_min=20, x_max=100, y_max=60)
        e3 = make_element(ElementType.TEXT, "z", x_min=200, y_min=200, x_max=300, y_max=250)
        assert e1.overlaps_with(e2)
        assert not e1.overlaps_with(e3)

    def test_descendants(self):
        parent = make_element(ElementType.HEADING, "H1", element_id="p")
        child = make_element(ElementType.TEXT, "para", element_id="c", parent_id="p")
        parent.child_ids = ["c"]
        desc = parent.get_descendants([parent, child])
        assert len(desc) == 1 and desc[0].element_id == "c"

    def test_ancestors(self):
        parent = make_element(ElementType.HEADING, "H1", element_id="p")
        child = make_element(ElementType.TEXT, "para", element_id="c", parent_id="p")
        anc = child.get_ancestors([parent, child])
        assert len(anc) == 1 and anc[0].element_id == "p"

    def test_rejects_invalid_type(self):
        with pytest.raises((ValueError, AttributeError)):
            make_element("not_a_type", "text")

    def test_rejects_confidence_over_one(self):
        with pytest.raises(ValueError):
            make_element(ElementType.TEXT, "x", confidence=1.5)

    def test_rejects_negative_page(self):
        with pytest.raises(ValueError):
            StructuralElement(
                "x", ElementType.TEXT, "x",
                BoundingBox(0, 0, 10, 10), 0.9, page_number=0,
            )


# ── DocumentMetadata ──────────────────────────────────────────────────────────

class TestDocumentMetadata:
    def _valid_meta(self, **overrides):
        defaults = dict(
            source_file="img.png",
            document_id="doc_1",
            processing_timestamp=datetime.now(),
            processing_duration=1.0,
            image_dimensions=(800, 600),
            detected_language="eng",
            total_elements_extracted=5,
            average_confidence=0.85,
            processing_status=ProcessingStatus.COMPLETED,
        )
        defaults.update(overrides)
        return DocumentMetadata(**defaults)

    def test_valid(self):
        m = self._valid_meta()
        assert m.document_id == "doc_1"

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError):
            self._valid_meta(processing_duration=0)

    def test_rejects_zero_dimensions(self):
        with pytest.raises(ValueError):
            self._valid_meta(image_dimensions=(0, 600))
        with pytest.raises(ValueError):
            self._valid_meta(image_dimensions=(800, 0))

    def test_rejects_negative_confidence(self):
        with pytest.raises(ValueError):
            self._valid_meta(average_confidence=-0.1)


# ── DocumentResult ────────────────────────────────────────────────────────────

class TestDocumentResult:
    def test_index_built(self):
        e1 = make_element(ElementType.HEADING, "H1", element_id="h1")
        doc = make_doc([e1])
        assert "h1" in doc.element_index

    def test_get_by_type(self):
        h = make_element(ElementType.HEADING, "H", element_id="h")
        p = make_element(ElementType.TEXT, "P", element_id="p")
        doc = make_doc([h, p])
        headings = doc.get_elements_by_type(ElementType.HEADING)
        assert len(headings) == 1 and headings[0].element_id == "h"

    def test_get_on_page(self):
        e1 = make_element(ElementType.TEXT, "a", page_number=1, element_id="e1")
        e2 = make_element(ElementType.TEXT, "b", page_number=2, element_id="e2")
        doc = make_doc([e1, e2])
        pg1 = doc.get_elements_on_page(1)
        assert len(pg1) == 1 and pg1[0].element_id == "e1"

    def test_get_in_region(self):
        e = make_element(ElementType.TEXT, "x", x_min=10, y_min=10, x_max=50, y_max=40, element_id="e")
        doc = make_doc([e])
        found = doc.get_elements_in_region(BoundingBox(0, 0, 100, 100))
        assert any(el.element_id == "e" for el in found)


# ── BatchResult ───────────────────────────────────────────────────────────────

class TestBatchResult:
    def test_statistics_computed(self):
        elems = [
            make_element(ElementType.HEADING, "H"),
            make_element(ElementType.TEXT, "P"),
        ]
        batch = make_batch([elems])
        assert batch.statistics is not None
        assert batch.statistics.total_elements == 2
        assert batch.statistics.successful_documents == 1

    def test_filter_by_type(self):
        elems = [
            make_element(ElementType.HEADING, "H"),
            make_element(ElementType.TEXT, "P"),
        ]
        batch = make_batch([elems])
        filtered = batch.filter_by_type(ElementType.HEADING)
        assert len(filtered.documents) == 1
        assert len(filtered.documents[0].elements) == 1

    def test_filter_by_confidence(self):
        elems = [
            make_element(ElementType.TEXT, "hi", confidence=0.95),
            make_element(ElementType.TEXT, "lo", confidence=0.40),
        ]
        batch = make_batch([elems])
        filtered = batch.filter_by_confidence(0.90)
        assert len(filtered.documents[0].elements) == 1

    def test_empty_batch_no_statistics_error(self):
        batch = make_batch([[]])
        # statistics should be None or have zero elements
        if batch.statistics is not None:
            assert batch.statistics.total_elements == 0


# ── Domain objects ─────────────────────────────────────────────────────────────

class TestDomainObjects:
    def test_annotation(self):
        a = Annotation("text", make_bbox(), "highlight", 0.9)
        assert a.annotation_type == "highlight"

    def test_code_block(self):
        cb = CodeBlock("def f(): pass", make_bbox(), 0.88, language="python")
        assert cb.language == "python"

    def test_code_block_empty_raises(self):
        with pytest.raises(ValueError):
            CodeBlock("", make_bbox(), 0.88)

    def test_caption(self):
        c = Caption("Figure 1: Example", "figure", make_bbox(), 0.92)
        assert c.caption_type == "figure"

    def test_barcode(self):
        b = Barcode("QR_CODE", "https://example.com", make_bbox(), 0.95)
        assert b.barcode_type == "QR_CODE"

    def test_barcode_empty_value_raises(self):
        with pytest.raises(ValueError):
            Barcode("QR", "", make_bbox(), 0.9)

    def test_watermark(self):
        w = Watermark("DRAFT", make_bbox(), 0.75, opacity_estimate=0.3)
        assert w.opacity_estimate == 0.3

    def test_watermark_bad_opacity_raises(self):
        with pytest.raises(ValueError):
            Watermark("X", make_bbox(), 0.7, opacity_estimate=1.5)

    def test_figure_region(self):
        f = FigureRegion(make_bbox(), 0.80, figure_type="chart")
        assert f.figure_type == "chart"

    def test_toc_entry(self):
        t = TableOfContents("Intro", page_number=1, level=1, bbox=make_bbox(), confidence=0.9)
        assert t.level == 1

    def test_toc_page_zero_raises(self):
        with pytest.raises(ValueError):
            TableOfContents("X", page_number=0, level=1, bbox=make_bbox(), confidence=0.9)

    def test_index_entry(self):
        e = IndexEntry("term", [1, 5, 10], level=1, bbox=make_bbox(), confidence=0.8)
        assert e.term == "term"

    def test_index_entry_page_zero_raises(self):
        with pytest.raises(ValueError):
            IndexEntry("x", [0], level=1, bbox=make_bbox(), confidence=0.8)

    def test_formula_empty_raises(self):
        with pytest.raises(ValueError):
            FormulaExpression("", make_bbox(), 0.8)

    def test_reference(self):
        r = Reference("[1] Author 2020", "bibliography", "1", make_bbox())
        assert r.ref_type == "bibliography"
