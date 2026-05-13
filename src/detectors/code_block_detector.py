"""
Code Block Detection Module

Identifies pre-formatted code or monospaced text regions and extracts their
verbatim content, preserving line structure.

Two detection passes are combined:

  Pass 1 — Visual (gray-background box)
    Finds rectangular regions whose background is a uniform gray (not white,
    not black).  Common in technical documents and programming books where
    code is set in a shaded box.  For each box, all OCR results within the
    bbox are gathered, sorted into reading order, and joined with newlines to
    reconstruct the verbatim code text faithfully.

  Pass 2 — Structural (consistent left-alignment)
    Finds groups of ≥ N consecutive OCR results that share a similar left
    margin AND contain at least one code-like token (bracket, operator, or
    language keyword).  These are collapsed into a single CodeBlock even
    when there is no visible background box.

Both passes attempt simple language detection from keyword presence.

Content fidelity:
    Text is gathered from pre-computed OCR results in reading order
    (top → bottom, then left → right within the same line).  No extra OCR
    pass is performed.  Confidence is the mean of the contributing OCR
    result confidences.

Follows Config + Detector + Trace pattern.
"""

import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_models import (
    BoundingBox,
    CodeBlock,
    ElementType,
    OCRTextResult,
    StructuralElement,
)

import logging
logger = logging.getLogger(__name__)


# ── Language fingerprints (keyword → language label) ──────────────────────────

_LANGUAGE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("python",     re.compile(r"\b(?:def|class|import|from|elif|lambda|yield|self)\b")),
    ("javascript", re.compile(r"\b(?:function|const|let|var|=>|async|await|typeof)\b")),
    ("typescript", re.compile(r"\b(?:interface|type\s+\w|readonly|namespace|enum)\b")),
    ("java",       re.compile(r"\b(?:public\s+class|private\s+|protected\s+|static\s+void|@Override)\b")),
    ("csharp",     re.compile(r"\b(?:namespace|using\s+System|public\s+class|private\s+|protected\s+)\b")),
    ("cpp",        re.compile(r"\b(?:#include|std::|cout|cin|nullptr|template<|typename)\b")),
    ("c",          re.compile(r"\b(?:#include|printf|scanf|malloc|free|NULL|sizeof)\b")),
    ("sql",        re.compile(r"\b(?:SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|INSERT INTO|UPDATE)\b", re.IGNORECASE)),
    ("bash",       re.compile(r"(?:^\s*#!|\$\s*\w|\becho\b|\bexport\b|\bsource\b|\bgrep\b)")),
    ("html",       re.compile(r"<(?:html|head|body|div|span|p|script|style)[>\s/]", re.IGNORECASE)),
    ("css",        re.compile(r"\{[^}]*:\s*[^;]+;[^}]*\}|@media|@keyframes")),
]

# Code-like tokens that indicate (but do not prove) a region is code
_CODE_TOKEN_RE = re.compile(
    r"""
    (?:
        [{}()\[\]]          # brackets
      | ==|!=|>=|<=|\+=|-=|\*=|/=|->|=>|::|  # operators
      | \b(?:if|else|for|while|return|import|def|class|var|let|const|function)\b
    )
    """,
    re.VERBOSE,
)


@dataclass
class CodeBlockDetectorConfig:
    """Configuration for CodeBlockDetector."""

    # Pass 1: visual (gray background box) detection
    detect_visual_boxes: bool = True
    # Pixel value lower bound for "gray background" (above → not black text/line)
    box_gray_min: int = 160
    # Pixel value upper bound for "gray background" (below → not pure white)
    box_gray_max: int = 240
    # Minimum area (pixels²) for a gray region to be considered a code box
    min_box_area: int = 2000
    # Minimum fraction of the box that must be the uniform background color
    min_box_fill_fraction: float = 0.30

    # Pass 2: structural (alignment + token) detection
    detect_structural: bool = True
    # Minimum number of consecutive aligned OCR results to form a code block
    min_aligned_lines: int = 3
    # Maximum x-min deviation (pixels) between lines to be considered "aligned"
    max_alignment_deviation: float = 8.0
    # Each group must contain at least this many code token hits
    min_code_tokens: int = 1

    # Language detection: disable to skip per-block keyword scan
    detect_language: bool = True

    min_confidence: float = 0.35


@dataclass
class CodeBlockDetectionTrace:
    """Records what CodeBlockDetector.detect() found and how."""
    code_blocks_found: int = 0
    visual_detections: int = 0
    structural_detections: int = 0
    ocr_results_analyzed: int = 0
    languages_detected: Dict[str, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    config: Optional[CodeBlockDetectorConfig] = None


class CodeBlockDetector:
    """
    Detects code / pre-formatted text regions and extracts their content.

    Args:
        config: CodeBlockDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[CodeBlockDetectorConfig] = None) -> None:
        self.config = config or CodeBlockDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], CodeBlockDetectionTrace]:
        """
        Detect code blocks in an image.

        Args:
            image:       BGR (or grayscale) NumPy array.
            ocr_results: Pre-computed OCR results for the same image.
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        trace = CodeBlockDetectionTrace(config=self.config)
        trace.ocr_results_analyzed = len(ocr_results)
        elements: List[StructuralElement] = []

        image_h, image_w = image.shape[:2]
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )

        # Track which OCR results have already been claimed by Pass 1
        claimed: set = set()

        # ── Pass 1: visual gray-background box ────────────────────────────────
        if self.config.detect_visual_boxes:
            box_bboxes = self._find_gray_boxes(gray, image_h, image_w)
            for bbox in box_bboxes:
                ocrs_in_box, indices = self._ocrs_in_bbox(ocr_results, bbox)
                if not ocrs_in_box:
                    continue
                content, confidence = self._gather_text(ocrs_in_box)
                if not content.strip():
                    continue
                language = self._detect_language(content)
                code_block = CodeBlock(
                    content=content,
                    bbox=bbox,
                    confidence=confidence,
                    language=language,
                )
                elements.append(StructuralElement(
                    element_id=f"cb_{uuid.uuid4().hex[:8]}",
                    element_type=ElementType.CODE_BLOCK,
                    content=code_block,
                    bbox=bbox,
                    confidence=confidence,
                    page_number=page_number,
                    processing_method="code_block_detector_visual",
                ))
                claimed.update(indices)
                trace.visual_detections += 1
                if language:
                    trace.languages_detected[language] = (
                        trace.languages_detected.get(language, 0) + 1
                    )

        # ── Pass 2: structural alignment ──────────────────────────────────────
        if self.config.detect_structural:
            unclaimed = [r for i, r in enumerate(ocr_results) if i not in claimed]
            groups = self._find_aligned_groups(unclaimed)
            for group_ocrs in groups:
                content, confidence = self._gather_text(group_ocrs)
                if not content.strip():
                    continue
                token_count = len(_CODE_TOKEN_RE.findall(content))
                if token_count < self.config.min_code_tokens:
                    continue
                language = self._detect_language(content)
                bbox = self._union_bbox(group_ocrs)
                if bbox is None:
                    continue
                code_block = CodeBlock(
                    content=content,
                    bbox=bbox,
                    confidence=confidence,
                    language=language,
                )
                elements.append(StructuralElement(
                    element_id=f"cb_{uuid.uuid4().hex[:8]}",
                    element_type=ElementType.CODE_BLOCK,
                    content=code_block,
                    bbox=bbox,
                    confidence=confidence,
                    page_number=page_number,
                    processing_method="code_block_detector_structural",
                ))
                trace.structural_detections += 1
                if language:
                    trace.languages_detected[language] = (
                        trace.languages_detected.get(language, 0) + 1
                    )

        trace.code_blocks_found = len(elements)
        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _find_gray_boxes(
        self,
        gray: np.ndarray,
        image_h: int,
        image_w: int,
    ) -> List[BoundingBox]:
        """Return bounding boxes of uniform gray rectangular regions."""
        cfg = self.config
        # Mask for gray-range pixels
        _, lo = cv2.threshold(gray, cfg.box_gray_min - 1, 255, cv2.THRESH_BINARY)
        _, hi = cv2.threshold(gray, cfg.box_gray_max, 255, cv2.THRESH_BINARY)
        gray_mask = cv2.bitwise_and(lo, cv2.bitwise_not(hi))

        # Close small gaps; erode to avoid fringe pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        gray_mask = cv2.erode(gray_mask, erode_k, iterations=1)

        contours, _ = cv2.findContours(
            gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: List[BoundingBox] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < cfg.min_box_area:
                continue
            # Rectangularity: contour area vs bbox area
            cnt_area = cv2.contourArea(cnt)
            if cnt_area / (w * h) < cfg.min_box_fill_fraction:
                continue
            boxes.append(BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            ))
        return boxes

    def _ocrs_in_bbox(
        self,
        ocr_results: List[OCRTextResult],
        bbox: BoundingBox,
    ) -> Tuple[List[OCRTextResult], List[int]]:
        """Return (matching OCR results, their original indices)."""
        inside: List[OCRTextResult] = []
        indices: List[int] = []
        for i, r in enumerate(ocr_results):
            if r.bbox.intersection(bbox) is not None:
                inside.append(r)
                indices.append(i)
        return inside, indices

    def _gather_text(self, ocr_results: List[OCRTextResult]) -> Tuple[str, float]:
        """
        Join OCR results in reading order (top→bottom, left→right) with
        newlines between vertical-position groups.  Returns (text, avg_confidence).
        """
        if not ocr_results:
            return "", 0.0

        # Sort by y-mid, then x-min
        sorted_ocrs = sorted(
            ocr_results,
            key=lambda r: ((r.bbox.y_min + r.bbox.y_max) / 2, r.bbox.x_min),
        )

        # Group into lines: results within LINE_TOLERANCE pixels vertically
        LINE_TOLERANCE = 6
        lines: List[List[OCRTextResult]] = []
        current_line: List[OCRTextResult] = [sorted_ocrs[0]]
        current_y = (sorted_ocrs[0].bbox.y_min + sorted_ocrs[0].bbox.y_max) / 2

        for r in sorted_ocrs[1:]:
            mid_y = (r.bbox.y_min + r.bbox.y_max) / 2
            if abs(mid_y - current_y) <= LINE_TOLERANCE:
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
                current_y = mid_y
        lines.append(current_line)

        text_lines = [" ".join(r.text for r in line) for line in lines]
        content = "\n".join(text_lines)
        avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
        return content, max(self.config.min_confidence, avg_conf)

    def _find_aligned_groups(
        self,
        ocr_results: List[OCRTextResult],
    ) -> List[List[OCRTextResult]]:
        """Find runs of OCR results with similar x_min (code left margin)."""
        if not ocr_results:
            return []

        # Sort by vertical position
        sorted_ocrs = sorted(ocr_results, key=lambda r: r.bbox.y_min)
        cfg = self.config
        groups: List[List[OCRTextResult]] = []
        current: List[OCRTextResult] = [sorted_ocrs[0]]

        for r in sorted_ocrs[1:]:
            ref_x = current[0].bbox.x_min
            if abs(r.bbox.x_min - ref_x) <= cfg.max_alignment_deviation:
                current.append(r)
            else:
                if len(current) >= cfg.min_aligned_lines:
                    groups.append(current)
                current = [r]

        if len(current) >= cfg.min_aligned_lines:
            groups.append(current)

        return groups

    def _union_bbox(self, ocr_results: List[OCRTextResult]) -> Optional[BoundingBox]:
        """Return the bounding box enclosing all given OCR results."""
        if not ocr_results:
            return None
        x_min = min(r.bbox.x_min for r in ocr_results)
        y_min = min(r.bbox.y_min for r in ocr_results)
        x_max = max(r.bbox.x_max for r in ocr_results)
        y_max = max(r.bbox.y_max for r in ocr_results)
        if x_max <= x_min or y_max <= y_min:
            return None
        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def _detect_language(self, text: str) -> Optional[str]:
        """Return the first matched language label, or None."""
        if not self.config.detect_language:
            return None
        for lang, pat in _LANGUAGE_PATTERNS:
            if pat.search(text):
                return lang
        return None
