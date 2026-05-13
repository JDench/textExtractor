"""
Tests for LayoutDetector and HeuristicLayoutModel.

No Tesseract or real document images needed — all tests use synthetic numpy arrays.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

cv2 = pytest.importorskip("cv2", reason="cv2 required for LayoutDetector")

from detectors.layout_detector import (
    LayoutDetector,
    LayoutDetectorConfig,
    LayoutDetectionTrace,
    LayoutPrediction,
    HeuristicLayoutModel,
    ExternalLayoutModel,
)
from data_models import BoundingBox, ElementType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _white(h=400, w=600):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _blank(h=400, w=600):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _image_with_gray_rect(rect_x, rect_y, rect_w, rect_h, gray=128, bg=255):
    """White image with a gray rectangle (simulates a figure region)."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * bg
    img[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = gray
    return img


def _image_with_grid_lines(line_spacing=40):
    """White image with a regular grid of black lines (simulates a table)."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Horizontal lines
    for y in range(50, 350, line_spacing):
        img[y, 50:550] = 0
    # Vertical lines
    for x in range(50, 550, line_spacing * 2):
        img[50:350, x] = 0
    return img


def detect(image, **cfg_kwargs):
    cfg = LayoutDetectorConfig(**cfg_kwargs) if cfg_kwargs else LayoutDetectorConfig()
    detector = LayoutDetector(cfg)
    return detector.detect(image, [], page_number=1)


# ── HeuristicLayoutModel: figure detection ────────────────────────────────────

class TestHeuristicFigureDetection:
    def test_large_dark_rect_detected_as_figure(self):
        # Black rectangle: high non-white pixel fraction → detected as figure
        img = _image_with_gray_rect(50, 50, 200, 150, gray=0)
        model = HeuristicLayoutModel(min_figure_area=5_000, figure_text_density_threshold=0.05)
        preds = model.predict(img)
        figures = [p for p in preds if p.element_type == ElementType.FIGURE]
        assert len(figures) >= 1

    def test_white_image_has_no_figures(self):
        model = HeuristicLayoutModel()
        preds = model.predict(_white())
        figures = [p for p in preds if p.element_type == ElementType.FIGURE]
        assert len(figures) == 0

    def test_small_rect_below_min_area_skipped(self):
        # 20×20 = 400 px² < default min_figure_area=8000
        img = _image_with_gray_rect(10, 10, 20, 20, gray=30)
        model = HeuristicLayoutModel(min_figure_area=8_000)
        preds = model.predict(img)
        figures = [p for p in preds if p.element_type == ElementType.FIGURE]
        assert len(figures) == 0


# ── HeuristicLayoutModel: table detection ────────────────────────────────────

class TestHeuristicTableDetection:
    def test_grid_lines_detected_as_table(self):
        img = _image_with_grid_lines(line_spacing=40)
        model = HeuristicLayoutModel()
        preds = model.predict(img)
        tables = [p for p in preds if p.element_type == ElementType.TABLE]
        assert len(tables) >= 1

    def test_blank_image_has_no_tables(self):
        model = HeuristicLayoutModel()
        preds = model.predict(_blank())
        tables = [p for p in preds if p.element_type == ElementType.TABLE]
        assert len(tables) == 0


# ── LayoutDetector.detect() interface ────────────────────────────────────────

class TestLayoutDetectorInterface:
    def test_returns_elements_and_trace(self):
        elements, trace = detect(_white())
        assert isinstance(elements, list)
        assert isinstance(trace, LayoutDetectionTrace)

    def test_trace_model_backend_recorded(self):
        _, trace = detect(_white(), model_backend="heuristic")
        assert trace.model_backend_used == "heuristic"

    def test_trace_processing_time_non_negative(self):
        _, trace = detect(_white())
        assert trace.processing_time_seconds >= 0.0

    def test_min_confidence_filters_low_predictions(self):
        img = _image_with_gray_rect(50, 50, 200, 150, gray=30)
        elems_low, _ = detect(img, min_confidence=0.0)
        elems_high, _ = detect(img, min_confidence=0.99)
        assert len(elems_high) <= len(elems_low)

    def test_element_has_bounding_box(self):
        img = _image_with_gray_rect(50, 50, 200, 150, gray=30)
        elements, _ = detect(img, min_confidence=0.0)
        if elements:
            assert elements[0].bbox is not None

    def test_element_page_number_set(self):
        img = _image_with_gray_rect(50, 50, 200, 150, gray=30)
        cfg = LayoutDetectorConfig(min_confidence=0.0)
        det = LayoutDetector(cfg)
        elements, _ = det.detect(img, [], page_number=3)
        if elements:
            assert elements[0].page_number == 3


# ── ExternalLayoutModel ───────────────────────────────────────────────────────

class TestExternalModel:
    def test_external_model_called(self):
        called = []

        def my_model(image):
            called.append(True)
            return [LayoutPrediction(
                bbox=BoundingBox(0, 0, 100, 100),
                element_type=ElementType.FIGURE,
                confidence=0.8,
            )]

        cfg = LayoutDetectorConfig(model_backend="external", external_model=my_model)
        det = LayoutDetector(cfg)
        elements, _ = det.detect(_white(), [], page_number=1)
        assert called
        assert len(elements) == 1
        assert elements[0].element_type == ElementType.FIGURE

    def test_external_model_none_raises(self):
        cfg = LayoutDetectorConfig(model_backend="external", external_model=None)
        with pytest.raises(ValueError, match="external_model"):
            LayoutDetector(cfg)


# ── Trace counts ──────────────────────────────────────────────────────────────

class TestTraceCounts:
    def test_figures_count_in_trace(self):
        img = _image_with_gray_rect(50, 50, 200, 150, gray=30)
        _, trace = detect(img, min_confidence=0.0)
        assert trace.figures_found == trace.regions_predicted - trace.tables_found
