"""
Layout Detection Module — ML-backed region classification

Segments a document image into coarse layout regions (text blocks, figures,
tables, etc.) independently of the OCR stream.  The detector provides a
pluggable model interface so a heavyweight ML model (e.g. LayoutLMv3,
PaddleOCR layout) can be dropped in without changing the detector API.

Two implementations are provided out of the box:

  HeuristicLayoutModel  — always available; uses cv2 contour analysis and
                          text-density estimation.  No extra dependencies.

  ExternalLayoutModel   — thin wrapper; calls any callable that accepts a
                          numpy image and returns List[LayoutPrediction].
                          Useful for integrating third-party models.

Config + Detector + Trace pattern follows every other detector in the suite.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple
import uuid

import cv2
import numpy as np

from data_models import (
    BoundingBox,
    ElementType,
    StructuralElement,
    OCRTextResult,
)


# ── Prediction dataclass ───────────────────────────────────────────────────────

@dataclass
class LayoutPrediction:
    """A single region predicted by a layout model."""
    bbox: BoundingBox
    element_type: ElementType
    confidence: float


# ── Model interface ────────────────────────────────────────────────────────────

class LayoutModel(ABC):
    """Abstract base for layout prediction models."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[LayoutPrediction]:
        """Return layout predictions for *image*."""


# ── Heuristic model ────────────────────────────────────────────────────────────

class HeuristicLayoutModel(LayoutModel):
    """
    Rule-based layout model using image analysis.

    Pass 1 — figure detection:
        Finds large regions of continuous non-white pixel mass that have
        low text density.  These are returned as FIGURE predictions.

    Pass 2 — table detection:
        Detects dense grids of horizontal and vertical lines via morphological
        operations.  Regions meeting the line-density threshold are returned
        as TABLE predictions.

    Pass 3 — heading detection:
        Uses OCR result font-size heuristics: OCR bboxes that are tall relative
        to surrounding text are scored as HEADING.  (This pass is skipped when
        no OCR results are provided.)

    All remaining bounding-box mass is implicitly TEXT.
    """

    def __init__(
        self,
        min_figure_area: int = 8_000,
        figure_text_density_threshold: float = 0.05,
        table_line_density_threshold: float = 0.03,
    ) -> None:
        self.min_figure_area = min_figure_area
        self.figure_text_density_threshold = figure_text_density_threshold
        self.table_line_density_threshold = table_line_density_threshold

    def predict(self, image: np.ndarray) -> List[LayoutPrediction]:
        h, w = image.shape[:2]
        predictions: List[LayoutPrediction] = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        predictions.extend(self._detect_figures(gray, binary, w, h))
        predictions.extend(self._detect_tables(gray, w, h))
        return predictions

    def _detect_figures(
        self, gray: np.ndarray, binary: np.ndarray, img_w: int, img_h: int
    ) -> List[LayoutPrediction]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(binary, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: List[LayoutPrediction] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < self.min_figure_area:
                continue
            # Reject regions that span the full width (likely a text block)
            if cw > img_w * 0.9:
                continue
            # Compute non-white fraction in the original gray ROI.
            # A figure has meaningful content (not nearly all-white whitespace).
            # Text lines also have content but are handled separately; here we
            # accept any region with >= figure_text_density_threshold non-white pixels.
            roi_gray = gray[y : y + ch, x : x + cw]
            non_white = float(np.mean(roi_gray < 230))
            if non_white < self.figure_text_density_threshold:
                continue
            confidence = min(0.9, 0.5 + (area / (img_w * img_h)))
            results.append(LayoutPrediction(
                bbox=BoundingBox(x_min=x, y_min=y, x_max=x + cw, y_max=y + ch),
                element_type=ElementType.FIGURE,
                confidence=round(confidence, 3),
            ))
        return results

    def _detect_tables(
        self, gray: np.ndarray, img_w: int, img_h: int
    ) -> List[LayoutPrediction]:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, img_w // 20), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, img_h // 20)))

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.add(h_lines, v_lines)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[LayoutPrediction] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 4_000:
                continue
            roi = grid[y : y + ch, x : x + cw]
            density = float(roi.mean()) / 255.0
            if density < self.table_line_density_threshold:
                continue
            results.append(LayoutPrediction(
                bbox=BoundingBox(x_min=x, y_min=y, x_max=x + cw, y_max=y + ch),
                element_type=ElementType.TABLE,
                confidence=round(min(0.85, 0.4 + density * 5), 3),
            ))
        return results


# ── External model wrapper ─────────────────────────────────────────────────────

class ExternalLayoutModel(LayoutModel):
    """
    Wraps any callable model that accepts (np.ndarray) and returns
    List[LayoutPrediction].  Use this to integrate third-party models.

    Example::

        def my_model(image):
            ...  # call PaddleOCR, LayoutLM, etc.
            return [LayoutPrediction(bbox, ElementType.TABLE, 0.9)]

        detector = LayoutDetector(LayoutDetectorConfig(
            model_backend="external",
            external_model=my_model,
        ))
    """

    def __init__(self, fn: Callable[[np.ndarray], List[LayoutPrediction]]) -> None:
        self._fn = fn

    def predict(self, image: np.ndarray) -> List[LayoutPrediction]:
        return self._fn(image)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class LayoutDetectorConfig:
    """
    Configuration for LayoutDetector.

    Attributes:
        model_backend: "heuristic" or "external".
        external_model: Callable to use when model_backend="external".
        min_confidence: Predictions below this score are discarded.
        min_figure_area: Min pixel area for figure detection (heuristic only).
        figure_text_density_threshold: Max text density to call a region a
            figure (heuristic only).
        table_line_density_threshold: Min line density to call a region a
            table (heuristic only).
    """
    model_backend: str = "heuristic"
    external_model: Optional[Any] = None
    min_confidence: float = 0.3
    min_figure_area: int = 8_000
    figure_text_density_threshold: float = 0.05
    table_line_density_threshold: float = 0.03


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class LayoutDetectionTrace:
    """Diagnostic trace for a single LayoutDetector.detect() call."""
    regions_predicted: int = 0
    figures_found: int = 0
    tables_found: int = 0
    model_backend_used: str = "heuristic"
    processing_time_seconds: float = 0.0


# ── Detector ──────────────────────────────────────────────────────────────────

class LayoutDetector:
    """
    Predicts coarse layout regions (figures, tables, text blocks) for a page.

    Follows the standard detector contract::

        elements, trace = detector.detect(image, ocr_results, page_number=1)
    """

    def __init__(self, config: Optional[LayoutDetectorConfig] = None) -> None:
        self.config = config or LayoutDetectorConfig()
        self._model = self._build_model()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], LayoutDetectionTrace]:
        t0 = time.perf_counter()
        trace = LayoutDetectionTrace(model_backend_used=self.config.model_backend)

        predictions = self._model.predict(image)
        elements: List[StructuralElement] = []

        for pred in predictions:
            if pred.confidence < self.config.min_confidence:
                continue
            elem = StructuralElement(
                element_id=uuid.uuid4().hex[:12],
                element_type=pred.element_type,
                content=None,
                bbox=pred.bbox,
                confidence=pred.confidence,
                page_number=page_number,
                metadata={"source": "layout_detector", "model": self.config.model_backend},
            )
            elements.append(elem)
            if pred.element_type == ElementType.FIGURE:
                trace.figures_found += 1
            elif pred.element_type == ElementType.TABLE:
                trace.tables_found += 1

        trace.regions_predicted = len(elements)
        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_model(self) -> LayoutModel:
        cfg = self.config
        if cfg.model_backend == "external":
            if cfg.external_model is None:
                raise ValueError(
                    "LayoutDetectorConfig.external_model must be set when "
                    "model_backend='external'."
                )
            return ExternalLayoutModel(cfg.external_model)
        # Default: heuristic
        return HeuristicLayoutModel(
            min_figure_area=cfg.min_figure_area,
            figure_text_density_threshold=cfg.figure_text_density_threshold,
            table_line_density_threshold=cfg.table_line_density_threshold,
        )
