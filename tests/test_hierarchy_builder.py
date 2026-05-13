"""
Tests for hierarchy_builder.py — HierarchyBuilder and HierarchyConfig.
"""

import pytest

from data_models import ElementType, TableOfContents
from hierarchy_builder import HierarchyBuilder, HierarchyConfig
from helpers import make_element, make_heading, make_bbox


# ── Helpers ────────────────────────────────────────────────────────────────────

def build(elements, **cfg_overrides):
    cfg = HierarchyConfig(**cfg_overrides) if cfg_overrides else None
    hb = HierarchyBuilder(cfg)
    result, trace = hb.build(elements)
    return result, trace


# ── Pass 1: heading hierarchy ──────────────────────────────────────────────────

class TestHeadingHierarchy:
    def test_empty_input(self):
        elems, trace = build([])
        assert elems == []
        assert trace.heading_links_made == 0

    def test_single_h1_has_no_parent(self):
        h1 = make_heading("H1", level=1, element_id="h1")
        elems, _ = build([h1])
        assert elems[0].parent_id is None

    def test_h2_parented_to_h1(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        h2 = make_heading("H2", level=2, y_min=30, element_id="h2")
        _, trace = build([h1, h2])
        assert h2.parent_id == "h1"
        assert "h2" in h1.child_ids
        assert trace.heading_links_made == 1

    def test_second_h1_resets_to_root(self):
        """After a second H1, H2 following it should be parented to the second H1."""
        h1a = make_heading("H1a", level=1, y_min=0, element_id="h1a")
        h2 = make_heading("H2", level=2, y_min=30, element_id="h2")
        h1b = make_heading("H1b", level=1, y_min=60, element_id="h1b")
        h2b = make_heading("H2b", level=2, y_min=90, element_id="h2b")
        build([h1a, h2, h1b, h2b])
        assert h2.parent_id == "h1a"
        assert h1b.parent_id is None   # second H1 → root
        assert h2b.parent_id == "h1b"

    def test_h3_nested_under_h2_under_h1(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        h2 = make_heading("H2", level=2, y_min=30, element_id="h2")
        h3 = make_heading("H3", level=3, y_min=60, element_id="h3")
        build([h1, h2, h3])
        assert h3.parent_id == "h2"
        assert h2.parent_id == "h1"

    def test_heading_links_count(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        h2a = make_heading("H2a", level=2, y_min=30, element_id="h2a")
        h2b = make_heading("H2b", level=2, y_min=60, element_id="h2b")
        _, trace = build([h1, h2a, h2b])
        assert trace.heading_links_made == 2


# ── Pass 1: content parenting ─────────────────────────────────────────────────

class TestContentParenting:
    def test_text_under_heading(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        p = make_element(ElementType.TEXT, "para", y_min=30, y_max=50, element_id="p")
        _, trace = build([h1, p])
        assert p.parent_id == "h1"
        assert "p" in h1.child_ids
        assert trace.content_links_made == 1

    def test_content_before_any_heading_has_no_parent(self):
        p = make_element(ElementType.TEXT, "intro", y_min=0, y_max=20, element_id="p")
        h1 = make_heading("H1", level=1, y_min=30, element_id="h1")
        build([h1, p])
        # p comes before h1 in reading order → no heading on stack yet
        assert p.parent_id is None

    def test_caption_not_auto_parented(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        cap = make_element(ElementType.CAPTION, "Figure 1", y_min=30, y_max=50, element_id="cap")
        build([h1, cap])
        # CAPTION is in _SKIP_PARENT_TYPES
        assert cap.parent_id is None

    def test_header_not_auto_parented(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        hdr = make_element(ElementType.HEADER, "Running Header", y_min=30, y_max=50, element_id="hdr")
        build([h1, hdr])
        assert hdr.parent_id is None

    def test_footer_not_auto_parented(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        ftr = make_element(ElementType.FOOTER, "Page 1", y_min=30, y_max=50, element_id="ftr")
        build([h1, ftr])
        assert ftr.parent_id is None

    def test_existing_parent_not_overwritten(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        p = make_element(
            ElementType.TEXT, "para", y_min=30, y_max=50,
            element_id="p", parent_id="external_parent",
        )
        build([h1, p])
        # Pre-existing parent_id must not be overwritten
        assert p.parent_id == "external_parent"

    def test_link_content_false_skips_parenting(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        p = make_element(ElementType.TEXT, "para", y_min=30, y_max=50, element_id="p")
        build([h1, p], link_content_to_headings=False)
        assert p.parent_id is None


# ── Pass 2: TOC → heading cross-link ─────────────────────────────────────────

class TestTOCCrossLink:
    def _toc_elem(self, title, page=1, element_id=None):
        from data_models import BoundingBox
        toc_content = TableOfContents(
            title=title,
            page_number=page,
            level=1,
            bbox=BoundingBox(0, 0, 100, 20),
            confidence=0.85,
        )
        return make_element(
            ElementType.TABLE_OF_CONTENTS,
            toc_content,
            element_id=element_id or f"toc_{title[:4]}",
        )

    def test_toc_linked_to_heading(self):
        h = make_heading("Introduction", level=1, y_min=0, element_id="h_intro")
        toc = self._toc_elem("Introduction", element_id="toc_intro")
        build([h, toc])
        assert toc.content.target_heading_id == "h_intro"

    def test_toc_not_linked_when_no_match(self):
        h = make_heading("Methods", level=1, y_min=0, element_id="h_methods")
        toc = self._toc_elem("Results", element_id="toc_results")
        build([h, toc])
        assert toc.content.target_heading_id is None

    def test_toc_cross_link_false_skips(self):
        h = make_heading("Introduction", level=1, y_min=0, element_id="h_intro")
        toc = self._toc_elem("Introduction", element_id="toc_intro")
        build([h, toc], link_toc_to_headings=False)
        assert toc.content.target_heading_id is None

    def test_already_linked_toc_not_overwritten(self):
        h = make_heading("Introduction", level=1, y_min=0, element_id="h_intro")
        toc = self._toc_elem("Introduction", element_id="toc_intro")
        toc.content.target_heading_id = "pre_existing_id"
        build([h, toc])
        # Should NOT overwrite an already-set target_heading_id
        assert toc.content.target_heading_id == "pre_existing_id"

    def test_toc_links_count(self):
        h1 = make_heading("Introduction", level=1, y_min=0, element_id="h1")
        h2 = make_heading("Methods", level=2, y_min=30, element_id="h2")
        toc1 = self._toc_elem("Introduction", element_id="toc1")
        toc2 = self._toc_elem("Methods", element_id="toc2")
        _, trace = build([h1, h2, toc1, toc2])
        assert trace.toc_links_made == 2


# ── Idempotency ───────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_double_build_is_safe(self):
        """Running builder twice on the same list must not duplicate relationships."""
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        p = make_element(ElementType.TEXT, "para", y_min=30, y_max=50, element_id="p")
        build([h1, p])
        first_parent = p.parent_id
        first_children = list(h1.child_ids)
        # Run again
        build([h1, p])
        # parent_id already set; builder respects it
        assert p.parent_id == first_parent
        # child_ids should not be duplicated (element already in list)
        assert h1.child_ids.count("p") == 1


# ── Trace metadata ─────────────────────────────────────────────────────────────

class TestTrace:
    def test_trace_records_counts(self):
        h1 = make_heading("H1", level=1, y_min=0, element_id="h1")
        p = make_element(ElementType.TEXT, "para", y_min=30, y_max=50, element_id="p")
        _, trace = build([h1, p])
        assert trace.elements_processed == 2
        assert trace.content_links_made == 1
        assert trace.heading_links_made == 0

    def test_trace_stores_config(self):
        cfg = HierarchyConfig(link_content_to_headings=False)
        hb = HierarchyBuilder(cfg)
        _, trace = hb.build([])
        assert trace.config is cfg
