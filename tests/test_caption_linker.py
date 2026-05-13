"""
Tests for CaptionLinker — spatial caption ↔ figure/table linking.

No image processing or OCR needed; all tests use synthetic StructuralElements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from caption_linker import CaptionLinker, CaptionLinkerConfig
from data_models import BoundingBox, ElementType
from helpers import make_doc, make_element


# ── Helpers ────────────────────────────────────────────────────────────────────

def _caption(text: str, x_min=0, y_min=0, x_max=300, y_max=20, page=1):
    return make_element(
        ElementType.CAPTION, text,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        page_number=page,
    )


def _figure(x_min=0, y_min=50, x_max=300, y_max=250, page=1):
    return make_element(
        ElementType.FIGURE, None,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        page_number=page,
    )


def _table(x_min=0, y_min=50, x_max=300, y_max=250, page=1):
    return make_element(
        ElementType.TABLE, None,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        page_number=page,
    )


def link(elements, **cfg_kwargs):
    cfg = CaptionLinkerConfig(**cfg_kwargs) if cfg_kwargs else CaptionLinkerConfig()
    linker = CaptionLinker(cfg)
    doc = make_doc(elements)
    return linker.link(doc)


# ── Basic linking ──────────────────────────────────────────────────────────────

class TestBasicLinking:
    def test_caption_below_figure_gets_linked(self):
        fig = _figure(y_min=10, y_max=200)
        cap = _caption("Figure 1: A chart.", y_min=210, y_max=225)
        doc, trace = link([fig, cap])
        assert cap.metadata.get("linked_element_id") == fig.element_id
        assert fig.metadata.get("caption_id") == cap.element_id

    def test_caption_above_figure_gets_linked(self):
        cap = _caption("Figure 2: Diagram.", y_min=10, y_max=25)
        fig = _figure(y_min=30, y_max=200)
        doc, trace = link([cap, fig])
        assert cap.metadata.get("linked_element_id") == fig.element_id

    def test_table_caption_links_to_table(self):
        tbl = _table(y_min=10, y_max=150)
        cap = _caption("Table 1: Summary data.", y_min=155, y_max=170)
        doc, trace = link([tbl, cap])
        assert cap.metadata.get("linked_element_id") == tbl.element_id
        assert cap.metadata.get("linked_element_type") == "table"

    def test_figure_gets_caption_text_metadata(self):
        fig = _figure(y_min=10, y_max=200)
        cap = _caption("Figure 3: Example output.", y_min=205, y_max=220)
        link([fig, cap])
        assert fig.metadata.get("caption_text") == "Figure 3: Example output."


# ── Distance threshold ─────────────────────────────────────────────────────────

class TestDistanceThreshold:
    def test_far_caption_not_linked(self):
        fig = _figure(y_min=10, y_max=200)
        cap = _caption("Figure 1.", y_min=500, y_max=515)  # > 150 px gap
        doc, trace = link([fig, cap], max_proximity_px=150.0)
        assert "linked_element_id" not in cap.metadata

    def test_proximity_within_threshold_links(self):
        fig = _figure(y_min=10, y_max=200)
        cap = _caption("Figure 1.", y_min=201, y_max=216)  # 1 px gap
        doc, trace = link([fig, cap], max_proximity_px=5.0)
        assert cap.metadata.get("linked_element_id") == fig.element_id

    def test_link_distance_recorded_in_metadata(self):
        fig = _figure(y_min=10, y_max=200)
        cap = _caption("Figure 1.", y_min=210, y_max=225)  # 10 px gap
        link([fig, cap])
        dist = cap.metadata.get("link_distance_px", None)
        assert dist is not None
        assert dist >= 0.0


# ── Caption classification ─────────────────────────────────────────────────────

class TestCaptionClassification:
    def test_figure_caption_prefers_figure_over_table(self):
        fig = _figure(y_min=10, y_max=100)
        tbl = _table(y_min=110, y_max=200)
        cap = _caption("Figure 1: A plot.", y_min=205, y_max=220)
        link([fig, tbl, cap])
        assert cap.metadata.get("linked_element_type") == "figure"

    def test_table_caption_prefers_table_over_figure(self):
        fig = _figure(y_min=10, y_max=100)
        tbl = _table(y_min=110, y_max=200)
        cap = _caption("Table 2: Data summary.", y_min=205, y_max=220)
        link([fig, tbl, cap])
        assert cap.metadata.get("linked_element_type") == "table"

    def test_unknown_caption_links_to_nearest_any(self):
        fig = _figure(y_min=10, y_max=100)
        cap = _caption("Description of the element shown above.", y_min=105, y_max=120)
        doc, trace = link([fig, cap])
        # Should still link if within proximity
        assert cap.metadata.get("linked_element_id") is not None


# ── Multiple elements ──────────────────────────────────────────────────────────

class TestMultipleElements:
    def test_each_caption_links_to_nearest(self):
        fig1 = _figure(y_min=10, y_max=100)
        cap1 = _caption("Figure 1.", y_min=105, y_max=120)
        fig2 = _figure(y_min=200, y_max=300)
        cap2 = _caption("Figure 2.", y_min=305, y_max=320)
        link([fig1, cap1, fig2, cap2])
        assert cap1.metadata.get("linked_element_id") == fig1.element_id
        assert cap2.metadata.get("linked_element_id") == fig2.element_id

    def test_no_elements_no_crash(self):
        doc, trace = link([])
        assert trace.captions_found == 0
        assert trace.captions_linked == 0

    def test_captions_without_figures_or_tables_unlinked(self):
        cap = _caption("Figure 1.", y_min=0, y_max=20)
        text = make_element(ElementType.TEXT, "body text", y_min=50, y_max=70)
        doc, trace = link([cap, text])
        assert "linked_element_id" not in cap.metadata


# ── Same-page constraint ───────────────────────────────────────────────────────

class TestSamePageConstraint:
    def test_different_page_not_linked_by_default(self):
        fig = _figure(page=1, y_min=10, y_max=200)
        cap = _caption("Figure 1.", y_min=0, y_max=20, page=2)
        doc = make_doc([fig, cap])
        linker = CaptionLinker(CaptionLinkerConfig(same_page_only=True))
        linker.link(doc)
        assert "linked_element_id" not in cap.metadata

    def test_different_page_linked_when_disabled(self):
        fig = _figure(page=1, y_min=10, y_max=200)
        cap = _caption("Figure 1.", y_min=10, y_max=30, page=2)
        doc = make_doc([fig, cap])
        linker = CaptionLinker(CaptionLinkerConfig(same_page_only=False, max_proximity_px=50.0))
        linker.link(doc)
        # Within 50 px edge distance even across pages since y coords overlap
        assert cap.metadata.get("linked_element_id") == fig.element_id


# ── Trace ──────────────────────────────────────────────────────────────────────

class TestTrace:
    def test_trace_counts_captions_found(self):
        cap1 = _caption("Figure 1.", y_min=0, y_max=20)
        cap2 = _caption("Figure 2.", y_min=100, y_max=120)
        _, trace = link([cap1, cap2])
        assert trace.captions_found == 2

    def test_trace_counts_linked(self):
        fig = _figure(y_min=30, y_max=200)
        cap = _caption("Figure 1.", y_min=205, y_max=220)
        _, trace = link([fig, cap])
        assert trace.captions_linked == 1
        assert trace.figures_linked == 1
        assert trace.tables_linked == 0

    def test_trace_table_linked_count(self):
        tbl = _table(y_min=30, y_max=200)
        cap = _caption("Table 1.", y_min=205, y_max=220)
        _, trace = link([tbl, cap])
        assert trace.tables_linked == 1
        assert trace.figures_linked == 0
