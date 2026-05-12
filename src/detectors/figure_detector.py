"""
Figure Detection Module

Detects non-text visual regions (figures, charts, diagrams, photographs)
and links them to nearby captions in the document.

Strategy:
  1. Build a text-coverage mask from all OCR result bboxes (dilated slightly)
  2. Threshold the grayscale image for non-white content
  3. Subtract the text mask → figure-candidate mask
  4. Close small gaps with morphology, then find external contours
  5. Filter candidates by minimum area and minimum pixel std-dev (rejects blanks)
  6. Reject candidates whose internal text density exceeds max_text_density
  7. Classify figure type by color saturation and aspect ratio
  8. Search OCR results in a vertical margin above/below each figure for caption patterns
  9. Link caption ↔ figure via FigureRegion.caption_id and Caption.referenced_element_id

Follows Config + Detector + Trace pattern.
"""

import re
import uuid
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import logging

from data_models import (
    BoundingBox,
    Caption,
    ElementType,
    FigureRegion,
    OCRTextResult,
    StructuralElement,
)

logger = logging.getLogger(__name__)


# Caption label patterns, ordered most-specific first
_CAPTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:Figure|Fig\.?)\s*\.?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:Chart|Graph|Diagram|Illustration)\s*\.?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bPlate\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bPanel\s+([A-Za-z]|\d+)\b", re.IGNORECASE),
]


@dataclass
class FigureDetectorConfig:
    """Configuration for FigureDetector."""
    min_figure_area: int = 5000
    max_text_density: float = 0.15
    min_content_std: float = 8.0
    caption_search_margin: int = 80
    min_confidence: float = 0.4
    language: str = "eng"


@dataclass
class FigureDetectionTrace:
    """Records timing and counts for a single FigureDetector.detect() call."""
    figures_found: int = 0
    captions_found: int = 0
    candidates_evaluated: int = 0
    rejected_too_small: int = 0
    rejected_low_content: int = 0
    rejected_high_text_density: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[FigureDetectorConfig] = None


class FigureDetector:
    """
    Detects figure/diagram regions and their associated captions.

    Args:
        config: FigureDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[FigureDetectorConfig] = None) -> None:
        self.config = config or FigureDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], FigureDetectionTrace]:
        """
        Detect figures and captions in an image.

        Args:
            image:       BGR (or grayscale) NumPy array.
            ocr_results: Pre-computed OCR results for the same image.
            page_number: Page index assigned to emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).  elements contains FIGURE entries
            each optionally followed by a linked CAPTION entry.
        """
        t0 = time.perf_counter()
        trace = FigureDetectionTrace(config=self.config)
        elements: List[StructuralElement] = []

        image_h, image_w = image.shape[:2]
        text_mask = self._build_text_mask(image_h, image_w, ocr_results)
        candidates = self._find_figure_candidates(image, text_mask, trace)

        for (x, y, w, h) in candidates:
            density = self._compute_text_density(text_mask, x, y, w, h)
            if density > self.config.max_text_density:
                trace.rejected_high_text_density += 1
                continue

            bbox = BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            )
            roi = image[y : y + h, x : x + w]
            figure_type = self._classify_figure_type(roi)
            confidence = self._estimate_confidence(roi, density)

            fig_id = f"fig_{uuid.uuid4().hex[:8]}"
            figure_region = FigureRegion(
                bbox=bbox,
                confidence=confidence,
                figure_type=figure_type,
            )
            fig_elem = StructuralElement(
                element_id=fig_id,
                element_type=ElementType.FIGURE,
                content=figure_region,
                bbox=bbox,
                confidence=confidence,
                page_number=page_number,
                processing_method="figure_detector_contour",
            )

            caption_elem = self._find_caption(
                ocr_results=ocr_results,
                figure_bbox=bbox,
                figure_id=fig_id,
                page_number=page_number,
            )
            if caption_elem is not None:
                figure_region.caption_id = caption_elem.element_id
                fig_elem.child_ids.append(caption_elem.element_id)
                caption_elem.parent_id = fig_id
                elements.append(caption_elem)
                trace.captions_found += 1

            elements.append(fig_elem)
            trace.figures_found += 1

        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_text_mask(
        self,
        image_h: int,
        image_w: int,
        ocr_results: List[OCRTextResult],
    ) -> np.ndarray:
        """Binary mask (255 = text-covered pixel) dilated by a small kernel."""
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        for r in ocr_results:
            x1 = max(0, int(r.bbox.x_min))
            y1 = max(0, int(r.bbox.y_min))
            x2 = min(image_w, int(r.bbox.x_max))
            y2 = min(image_h, int(r.bbox.y_max))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 6))
        return cv2.dilate(mask, kernel, iterations=1)

    def _find_figure_candidates(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        trace: FigureDetectionTrace,
    ) -> List[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) rects of candidate figure regions."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        # Non-white content (ink, lines, images)
        _, content_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Strip text from the content mask
        figure_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(text_mask))
        # Close gaps within figure bodies
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        figure_mask = cv2.morphologyEx(figure_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            figure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            trace.candidates_evaluated += 1
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self.config.min_figure_area:
                trace.rejected_too_small += 1
                continue
            # Reject blank / near-uniform areas (blank white paper strips)
            roi_gray = gray[y : y + h, x : x + w]
            if roi_gray.size > 0 and float(np.std(roi_gray)) < self.config.min_content_std:
                trace.rejected_low_content += 1
                continue
            candidates.append((x, y, w, h))
        return candidates

    def _compute_text_density(
        self,
        text_mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        roi = text_mask[y : y + h, x : x + w]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi)) / float(roi.size)

    def _classify_figure_type(self, roi: np.ndarray) -> str:
        if roi.size == 0:
            return "unknown"
        h, w = roi.shape[:2]
        aspect = w / h if h > 0 else 1.0
        if len(roi.shape) == 3:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
            if sat_mean > 40:
                return "chart" if aspect > 1.8 else "photo"
        return "chart" if aspect > 2.5 else "diagram"

    def _estimate_confidence(self, roi: np.ndarray, text_density: float) -> float:
        if roi.size == 0:
            return self.config.min_confidence
        gray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        )
        # More visual content (higher std dev) + lower text overlap → higher confidence
        std_norm = min(float(np.std(gray)) / 60.0, 1.0)
        density_penalty = text_density / max(self.config.max_text_density, 1e-6)
        conf = 0.5 + 0.4 * std_norm - 0.2 * density_penalty
        return float(max(self.config.min_confidence, min(0.95, conf)))

    def _find_caption(
        self,
        ocr_results: List[OCRTextResult],
        figure_bbox: BoundingBox,
        figure_id: str,
        page_number: int,
    ) -> Optional[StructuralElement]:
        """Search OCR results above and below the figure for a caption line."""
        margin = float(self.config.caption_search_margin)

        # Build valid search zones (only when they have positive height)
        search_zones: List[BoundingBox] = []
        if figure_bbox.y_min > 1.0:
            above_y_min = max(0.0, figure_bbox.y_min - margin)
            if above_y_min < figure_bbox.y_min:
                search_zones.append(
                    BoundingBox(
                        x_min=figure_bbox.x_min,
                        y_min=above_y_min,
                        x_max=figure_bbox.x_max,
                        y_max=figure_bbox.y_min,
                    )
                )
        below_y_max = figure_bbox.y_max + margin
        search_zones.append(
            BoundingBox(
                x_min=figure_bbox.x_min,
                y_min=figure_bbox.y_max,
                x_max=figure_bbox.x_max,
                y_max=below_y_max,
            )
        )

        best_ocr: Optional[OCRTextResult] = None
        best_cap_num: Optional[str] = None

        for zone in search_zones:
            for ocr in ocr_results:
                if ocr.bbox.intersection(zone) is None:
                    continue
                for pat in _CAPTION_PATTERNS:
                    m = pat.search(ocr.text)
                    if m:
                        best_ocr = ocr
                        best_cap_num = m.group(1) if m.lastindex else None
                        break
                if best_ocr is not None:
                    break
            if best_ocr is not None:
                break

        if best_ocr is None:
            return None

        cap_id = f"cap_{uuid.uuid4().hex[:8]}"
        caption = Caption(
            content=best_ocr.text.strip(),
            caption_type="figure",
            bbox=best_ocr.bbox,
            confidence=best_ocr.confidence,
            referenced_element_id=figure_id,
            caption_number=best_cap_num,
        )
        return StructuralElement(
            element_id=cap_id,
            element_type=ElementType.CAPTION,
            content=caption,
            bbox=best_ocr.bbox,
            confidence=best_ocr.confidence,
            page_number=page_number,
            processing_method="figure_detector_caption_pattern",
        )
