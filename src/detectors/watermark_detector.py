"""
Watermark Detection Module

Detects semi-transparent background text overlays (watermarks) using two
complementary signals from OCR results:

  Signal 1 — Vocabulary match
    Any OCR result whose text contains a known watermark word (DRAFT,
    CONFIDENTIAL, COPY, SAMPLE, …) is flagged regardless of confidence.

  Signal 2 — Large-span + low-confidence + pixel-opacity
    An OCR result whose bounding box spans a large fraction of the page
    AND whose OCR confidence is low AND whose foreground pixels are
    medium-gray (not solid black) is treated as a diagonal or tiled
    watermark.  All three sub-conditions must hold to avoid flagging
    wide headings or low-quality scans as watermarks.

Each detected watermark is wrapped in a Watermark domain object that
records the OCR'd text, an estimated opacity, and a WATERMARK
StructuralElement.

Opacity estimation:
    The 10th-percentile grayscale value inside the bbox approximates the
    "darkest ink" intensity.  Values near 0 → solid black foreground
    (normal text).  Values > config.opacity_pixel_threshold → lighter ink
    (semi-transparent watermark).

Follows Config + Detector + Trace pattern.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from data_models import (
    BoundingBox,
    ElementType,
    OCRTextResult,
    StructuralElement,
    Watermark,
)

logger = logging.getLogger(__name__)


@dataclass
class WatermarkDetectorConfig:
    """Configuration for WatermarkDetector."""

    # Vocabulary that, when matched (case-insensitive), immediately flags the
    # OCR result as a watermark regardless of its confidence or position.
    watermark_vocabulary: List[str] = field(default_factory=lambda: [
        "DRAFT", "CONFIDENTIAL", "COPY", "SAMPLE", "VOID", "CANCELLED",
        "APPROVED", "REJECTED", "RECEIVED", "PAID", "ORIGINAL", "DUPLICATE",
        "PROPRIETARY", "INTERNAL USE ONLY", "NOT FOR DISTRIBUTION",
        "FOR REVIEW ONLY", "PRELIMINARY", "UNCLASSIFIED", "CLASSIFIED",
    ])

    # Signal 2: bbox must span at least this fraction of page width OR height.
    min_span_fraction: float = 0.45

    # Signal 2: OCR confidence must be at or below this for span detection.
    # (Solid body text typically comes back ≥ 0.70; watermarks are lower.)
    max_span_confidence: float = 0.65

    # Signal 2: 10th-percentile grayscale pixel value in the bbox must be
    # above this to indicate light/transparent ink (watermark).
    opacity_pixel_threshold: int = 110

    # Minimum bbox area in pixels to consider (filters tiny OCR fragments).
    min_bbox_area: int = 400

    # Minimum confidence assigned to emitted WATERMARK elements.
    min_output_confidence: float = 0.45


@dataclass
class WatermarkDetectionTrace:
    """Records what WatermarkDetector.detect() found and why."""
    watermarks_found: int = 0
    vocabulary_matches: int = 0
    span_detections: int = 0
    ocr_results_scanned: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[WatermarkDetectorConfig] = None


class WatermarkDetector:
    """
    Detects watermark overlays in document images.

    Args:
        config: WatermarkDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[WatermarkDetectorConfig] = None) -> None:
        self.config = config or WatermarkDetectorConfig()
        # Pre-compile vocabulary patterns for speed
        self._vocab_patterns = [
            __import__("re").compile(
                r"\b" + __import__("re").escape(word) + r"\b",
                __import__("re").IGNORECASE,
            )
            for word in self.config.watermark_vocabulary
        ]

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], WatermarkDetectionTrace]:
        """
        Detect watermarks in an image.

        Args:
            image:       BGR (or grayscale) NumPy array.
            ocr_results: Pre-computed OCR results for the same image.
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        import time
        t0 = time.perf_counter()
        trace = WatermarkDetectionTrace(config=self.config)
        elements: List[StructuralElement] = []

        image_h, image_w = image.shape[:2]
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        trace.ocr_results_scanned = len(ocr_results)

        seen_ids: set = set()
        for ocr in ocr_results:
            bbox_area = (ocr.bbox.x_max - ocr.bbox.x_min) * (ocr.bbox.y_max - ocr.bbox.y_min)
            if bbox_area < self.config.min_bbox_area:
                continue

            detection_method, matched_vocab = self._classify(
                ocr=ocr,
                gray=gray,
                image_h=image_h,
                image_w=image_w,
            )
            if detection_method is None:
                continue

            # Deduplicate: same bounding box may appear multiple times
            key = (round(ocr.bbox.x_min), round(ocr.bbox.y_min),
                   round(ocr.bbox.x_max), round(ocr.bbox.y_max))
            if key in seen_ids:
                continue
            seen_ids.add(key)

            if detection_method == "vocabulary":
                trace.vocabulary_matches += 1
            else:
                trace.span_detections += 1

            opacity = self._estimate_opacity(gray, ocr.bbox)
            confidence = self._score_confidence(ocr, detection_method, opacity)

            watermark = Watermark(
                content=ocr.text.strip(),
                bbox=ocr.bbox,
                confidence=confidence,
                is_background=True,
                opacity_estimate=opacity,
            )
            import uuid
            elem = StructuralElement(
                element_id=f"wm_{uuid.uuid4().hex[:8]}",
                element_type=ElementType.WATERMARK,
                content=watermark,
                bbox=ocr.bbox,
                confidence=confidence,
                page_number=page_number,
                processing_method=f"watermark_detector_{detection_method}",
                metadata={"matched_keyword": matched_vocab},
            )
            elements.append(elem)
            trace.watermarks_found += 1

        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify(
        self,
        ocr: OCRTextResult,
        gray: np.ndarray,
        image_h: int,
        image_w: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (detection_method, matched_keyword) or (None, None) if not a watermark.

        detection_method is "vocabulary" or "span".
        """
        text = ocr.text.strip()
        if not text:
            return None, None

        # Signal 1: vocabulary match (case-insensitive word boundary)
        for pat in self._vocab_patterns:
            if pat.search(text):
                return "vocabulary", pat.pattern

        # Signal 2: large span + low confidence + light pixels
        bbox_w = ocr.bbox.x_max - ocr.bbox.x_min
        bbox_h = ocr.bbox.y_max - ocr.bbox.y_min
        spans_page = (
            bbox_w / image_w >= self.config.min_span_fraction
            or bbox_h / image_h >= self.config.min_span_fraction
        )
        if (
            spans_page
            and ocr.confidence <= self.config.max_span_confidence
            and self._is_light_ink(gray, ocr.bbox)
        ):
            return "span", None

        return None, None

    def _is_light_ink(self, gray: np.ndarray, bbox: BoundingBox) -> bool:
        """Return True when the darkest pixels in the bbox are medium-gray."""
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(gray.shape[1], int(bbox.x_max))
        y2 = min(gray.shape[0], int(bbox.y_max))
        if x2 <= x1 or y2 <= y1:
            return False
        roi = gray[y1:y2, x1:x2]
        # 10th percentile of pixel values ≈ darkest ink intensity
        p10 = float(np.percentile(roi, 10))
        return p10 > self.config.opacity_pixel_threshold

    def _estimate_opacity(self, gray: np.ndarray, bbox: BoundingBox) -> Optional[float]:
        """
        Estimate opacity as a 0–1 value where 1.0 = fully opaque (solid black).
        Uses the 10th-percentile pixel value: darker → more opaque.
        """
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(gray.shape[1], int(bbox.x_max))
        y2 = min(gray.shape[0], int(bbox.y_max))
        if x2 <= x1 or y2 <= y1:
            return None
        p10 = float(np.percentile(gray[y1:y2, x1:x2], 10))
        # p10 = 0 (black) → opacity = 1.0; p10 = 255 (white) → opacity = 0.0
        return round(1.0 - p10 / 255.0, 3)

    def _score_confidence(
        self,
        ocr: OCRTextResult,
        detection_method: str,
        opacity: Optional[float],
    ) -> float:
        """Assign a confidence score to the watermark detection."""
        if detection_method == "vocabulary":
            base = 0.82
        else:
            base = 0.55

        # Boost when opacity is very low (faint text → strong watermark evidence)
        if opacity is not None and opacity < 0.5:
            base += 0.08

        return float(max(self.config.min_output_confidence, min(0.95, base)))
