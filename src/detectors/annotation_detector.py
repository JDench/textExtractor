"""
Annotation Detection Module

Detects visual annotations that readers add to document pages:

  HIGHLIGHT     — Colored background swath covering text.
                  Detected via HSV saturation threshold restricted to OCR-text zones.
  UNDERLINE     — Horizontal line drawn below a text line.
                  Detected with HoughLinesP on a thin strip just beneath each OCR bbox.
  STRIKETHROUGH — Horizontal line crossing through the vertical midpoint of a text line.
                  Detected with HoughLinesP on a thin strip at ~50 % of text height.

Each detected annotation stores the text it annotates (gathered from overlapping
OCR results) in Annotation.content so that downstream consumers can read the
marked-up text without a second OCR pass.

Follows Config + Detector + Trace pattern.
"""

import uuid
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from data_models import (
    Annotation,
    BoundingBox,
    ElementType,
    OCRTextResult,
    StructuralElement,
)

logger = logging.getLogger(__name__)


@dataclass
class AnnotationDetectorConfig:
    """Configuration for AnnotationDetector."""
    detect_highlights: bool = True
    detect_underlines: bool = True
    detect_strikethroughs: bool = True
    # Highlight detection
    highlight_min_saturation: int = 60     # HSV S-channel threshold (0-255)
    highlight_min_value: int = 50          # HSV V-channel minimum (reject near-black)
    highlight_min_area: int = 200          # minimum pixel area for a highlight blob
    # Underline / strikethrough (HoughLinesP)
    line_search_height: int = 8            # height of the scan strip in pixels
    hough_threshold: int = 30              # accumulator threshold
    hough_min_line_length: int = 20        # minimum line length to accept (px)
    hough_max_line_gap: int = 8            # maximum collinear gap within one line
    min_line_coverage: float = 0.40        # line must span ≥ this fraction of text width
    min_confidence: float = 0.40


@dataclass
class AnnotationDetectionTrace:
    """Records timing and counts for a single AnnotationDetector.detect() call."""
    highlights_found: int = 0
    underlines_found: int = 0
    strikethroughs_found: int = 0
    ocr_results_analyzed: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[AnnotationDetectorConfig] = None


class AnnotationDetector:
    """
    Detects visual annotations (highlights, underlines, strikethroughs).

    Args:
        config: AnnotationDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[AnnotationDetectorConfig] = None) -> None:
        self.config = config or AnnotationDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], AnnotationDetectionTrace]:
        """
        Detect annotations in an image.

        Args:
            image:       BGR (or grayscale) NumPy array.
            ocr_results: Pre-computed OCR results for the same image.
            page_number: Page index assigned to emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        trace = AnnotationDetectionTrace(config=self.config)
        elements: List[StructuralElement] = []

        image_h, image_w = image.shape[:2]
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        trace.ocr_results_analyzed = len(ocr_results)

        if self.config.detect_highlights and len(image.shape) == 3:
            highlights = self._detect_highlights(image, ocr_results, page_number, image_h, image_w)
            elements.extend(highlights)
            trace.highlights_found = len(highlights)

        for ocr in ocr_results:
            x1 = max(0, int(ocr.bbox.x_min))
            y1 = max(0, int(ocr.bbox.y_min))
            x2 = min(image_w, int(ocr.bbox.x_max))
            y2 = min(image_h, int(ocr.bbox.y_max))
            if x2 <= x1 or y2 <= y1:
                continue
            text_h = y2 - y1

            if self.config.detect_underlines:
                ul = self._detect_line_annotation(
                    gray=gray,
                    ocr=ocr,
                    x1=x1,
                    x2=x2,
                    scan_y=y2,
                    page_number=page_number,
                    ann_type="underline",
                    image_h=image_h,
                )
                if ul is not None:
                    elements.append(ul)
                    trace.underlines_found += 1

            if self.config.detect_strikethroughs:
                mid_y = y1 + text_h // 2
                st = self._detect_line_annotation(
                    gray=gray,
                    ocr=ocr,
                    x1=x1,
                    x2=x2,
                    scan_y=mid_y,
                    page_number=page_number,
                    ann_type="strikethrough",
                    image_h=image_h,
                )
                if st is not None:
                    elements.append(st)
                    trace.strikethroughs_found += 1

        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_highlights(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int,
        image_h: int,
        image_w: int,
    ) -> List[StructuralElement]:
        """Find colored highlight regions that overlap OCR text bboxes."""
        cfg = self.config
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # High-saturation, non-dark pixels
        sat_mask = cv2.inRange(
            hsv,
            (0, cfg.highlight_min_saturation, cfg.highlight_min_value),
            (179, 255, 255),
        )

        # Build a proximity mask: pixels inside any OCR bbox
        proximity = np.zeros((image_h, image_w), dtype=np.uint8)
        for r in ocr_results:
            x1 = max(0, int(r.bbox.x_min))
            y1 = max(0, int(r.bbox.y_min))
            x2 = min(image_w, int(r.bbox.x_max))
            y2 = min(image_h, int(r.bbox.y_max))
            if x2 > x1 and y2 > y1:
                proximity[y1:y2, x1:x2] = 255

        highlight_mask = cv2.bitwise_and(sat_mask, proximity)

        # Close small intra-word gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results: List[StructuralElement] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < cfg.highlight_min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            bbox = BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            )
            color_name = self._hsv_to_color_name(hsv[y : y + h, x : x + w])
            text_content = self._get_overlapping_text(ocr_results, bbox)
            annotation = Annotation(
                content=text_content,
                bbox=bbox,
                annotation_type="highlight",
                confidence=0.72,
                color=color_name,
            )
            results.append(
                StructuralElement(
                    element_id=f"ann_{uuid.uuid4().hex[:8]}",
                    element_type=ElementType.ANNOTATION,
                    content=annotation,
                    bbox=bbox,
                    confidence=0.72,
                    page_number=page_number,
                    processing_method="annotation_detector_hsv_highlight",
                )
            )
        return results

    def _detect_line_annotation(
        self,
        gray: np.ndarray,
        ocr: OCRTextResult,
        x1: int,
        x2: int,
        scan_y: int,
        page_number: int,
        ann_type: str,
        image_h: int,
    ) -> Optional[StructuralElement]:
        """
        Look for a near-horizontal line in a thin horizontal strip.

        For underlines: strip is just below the text bbox (scan_y = y2).
        For strikethroughs: strip is at the vertical midpoint (scan_y = y_mid).
        """
        cfg = self.config
        half = cfg.line_search_height // 2
        strip_y1 = max(0, scan_y - half)
        strip_y2 = min(image_h, scan_y + half)
        if strip_y2 <= strip_y1 or x2 <= x1:
            return None

        strip = gray[strip_y1:strip_y2, x1:x2]
        edges = cv2.Canny(strip, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        if lines is None:
            return None

        text_w = x2 - x1
        for ln in lines:
            lx1, ly1, lx2, ly2 = ln[0]
            if abs(ly2 - ly1) > 3:
                continue  # not horizontal enough
            line_len = abs(lx2 - lx1)
            if line_len < cfg.min_line_coverage * text_w:
                continue

            # Build a valid bbox (ensure x_max > x_min and y_max > y_min)
            bx_min = float(x1 + min(lx1, lx2))
            bx_max = float(x1 + max(lx1, lx2))
            by_min = float(strip_y1 + min(ly1, ly2))
            by_max = float(strip_y1 + max(ly1, ly2) + 1)
            if bx_max <= bx_min:
                bx_max = bx_min + 1.0
            bbox = BoundingBox(
                x_min=bx_min,
                y_min=by_min,
                x_max=bx_max,
                y_max=by_max,
            )
            annotation = Annotation(
                content=ocr.text.strip(),
                bbox=bbox,
                annotation_type=ann_type,
                confidence=0.65,
            )
            return StructuralElement(
                element_id=f"ann_{uuid.uuid4().hex[:8]}",
                element_type=ElementType.ANNOTATION,
                content=annotation,
                bbox=bbox,
                confidence=0.65,
                page_number=page_number,
                processing_method=f"annotation_detector_hough_{ann_type}",
            )
        return None

    def _hsv_to_color_name(self, hsv_roi: np.ndarray) -> str:
        if hsv_roi.size == 0:
            return "unknown"
        mean_h = float(np.mean(hsv_roi[:, :, 0]))
        if mean_h < 15 or mean_h >= 165:
            return "red"
        if mean_h < 30:
            return "orange"
        if mean_h < 45:
            return "yellow"
        if mean_h < 85:
            return "green"
        if mean_h < 130:
            return "blue"
        return "purple"

    def _get_overlapping_text(
        self,
        ocr_results: List[OCRTextResult],
        bbox: BoundingBox,
    ) -> str:
        parts = [r.text.strip() for r in ocr_results if r.bbox.intersection(bbox) is not None]
        return " ".join(parts)
