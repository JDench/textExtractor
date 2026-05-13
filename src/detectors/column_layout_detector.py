"""
Column Layout Detector — detects multi-column page layouts from OCR results.

Runs before structural detectors to identify pages with 2+ text columns.
When multi-column layout is detected:
  1. ContentTableDetector is suppressed (spatial clustering would false-positive).
  2. TEXT elements are sorted into column reading order (top-to-bottom per column,
     left column before right column).

Algorithm:
  - Project OCR bounding boxes onto the horizontal axis to build a coverage array.
  - Smooth with a box kernel to fill intra-word and inter-word gaps.
  - Find contiguous zero-coverage gaps >= min_gap_fraction * page_width.
  - Identify column regions between gaps with width >= min_column_width_fraction.
  - Confidence is derived from gap clarity (how empty the gap regions are).

Returns ColumnLayoutResult — NOT a StructuralElement; this is a page-level signal,
not a detected document element.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from data_models import OCRTextResult


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ColumnLayoutConfig:
    """Controls column layout detection sensitivity."""

    min_gap_fraction: float = 0.04
    """Gap must be at least this fraction of page width to count as a column separator."""

    min_column_width_fraction: float = 0.15
    """Each column must be at least this wide (fraction of page width)."""

    min_text_density: float = 0.25
    """Fraction of page height that must contain OCR results for detection to run."""

    min_ocr_results_for_detection: int = 8
    """Skip detection if fewer than this many OCR results are present."""

    max_columns: int = 4
    """Ignore results that imply more than this many columns (noise rejection)."""

    smoothing_kernel_fraction: float = 0.015
    """Width of smoothing kernel as a fraction of page width (fills intra-word gaps)."""

    gap_threshold: float = 0.05
    """Coverage fraction below which a region is treated as a gap."""

    multi_column_confidence_threshold: float = 0.50
    """Minimum confidence to declare a page multi-column."""


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class ColumnLayoutResult:
    """Result of running ColumnLayoutDetector on one page."""

    is_multi_column: bool
    num_columns: int
    columns: List[Tuple[float, float]]
    """List of (x_min, x_max) column regions in pixel coordinates."""

    gap_regions: List[Tuple[float, float]]
    """List of (x_min, x_max) gap regions between columns."""

    confidence: float
    page_width: float
    page_height: float

    def column_index_for_x(self, x: float) -> int:
        """Return which column index an x-coordinate falls in (0-based). -1 if in gap."""
        for i, (x_min, x_max) in enumerate(self.columns):
            if x_min <= x <= x_max:
                return i
        return 0  # default to first column for elements not cleanly in a column


# Sentinel for when detection was skipped (single column assumed).
SINGLE_COLUMN = ColumnLayoutResult(
    is_multi_column=False,
    num_columns=1,
    columns=[],
    gap_regions=[],
    confidence=1.0,
    page_width=0.0,
    page_height=0.0,
)


# ── Detector ───────────────────────────────────────────────────────────────────

class ColumnLayoutDetector:
    """Detects multi-column layouts using OCR bounding-box projection."""

    def __init__(self, config: Optional[ColumnLayoutConfig] = None) -> None:
        self.config = config or ColumnLayoutConfig()

    def detect(
        self,
        ocr_results: List[OCRTextResult],
        page_width: float,
        page_height: float,
    ) -> ColumnLayoutResult:
        """
        Detect column layout from OCR bounding boxes.

        Parameters
        ----------
        ocr_results:
            All OCR text results for the page.
        page_width, page_height:
            Dimensions of the page in pixels.
        """
        cfg = self.config

        if not ocr_results or len(ocr_results) < cfg.min_ocr_results_for_detection:
            return SINGLE_COLUMN

        if page_width <= 0 or page_height <= 0:
            return SINGLE_COLUMN

        # Build coverage array
        width = int(page_width)
        coverage = np.zeros(width, dtype=np.float32)
        for r in ocr_results:
            x0 = max(0, int(r.bbox.x_min))
            x1 = min(width, int(r.bbox.x_max))
            if x0 < x1:
                coverage[x0:x1] += 1.0

        # Smooth to fill intra-word gaps
        kernel_w = max(3, int(cfg.smoothing_kernel_fraction * width))
        kernel = np.ones(kernel_w, dtype=np.float32) / kernel_w
        smoothed = np.convolve(coverage, kernel, mode="same")

        # Normalize so values represent relative coverage density
        max_val = smoothed.max()
        if max_val == 0:
            return SINGLE_COLUMN
        normalized = smoothed / max_val

        # Find gap regions (coverage below threshold)
        min_gap_px = int(cfg.min_gap_fraction * width)
        min_col_px = int(cfg.min_column_width_fraction * width)
        gaps = _find_gaps(normalized, cfg.gap_threshold, min_gap_px)

        if not gaps:
            return ColumnLayoutResult(
                is_multi_column=False,
                num_columns=1,
                columns=[(0.0, page_width)],
                gap_regions=[],
                confidence=1.0,
                page_width=page_width,
                page_height=page_height,
            )

        # Build column regions as intervals between gaps and page edges
        columns = _build_columns(gaps, width, min_col_px)

        if len(columns) < 2 or len(columns) > cfg.max_columns:
            return ColumnLayoutResult(
                is_multi_column=False,
                num_columns=1,
                columns=[(0.0, page_width)],
                gap_regions=gaps,
                confidence=1.0,
                page_width=page_width,
                page_height=page_height,
            )

        # Confidence: average "emptiness" of gap regions
        confidence = _gap_confidence(normalized, gaps)

        if confidence < cfg.multi_column_confidence_threshold:
            return ColumnLayoutResult(
                is_multi_column=False,
                num_columns=1,
                columns=[(0.0, page_width)],
                gap_regions=gaps,
                confidence=confidence,
                page_width=page_width,
                page_height=page_height,
            )

        return ColumnLayoutResult(
            is_multi_column=True,
            num_columns=len(columns),
            columns=columns,
            gap_regions=gaps,
            confidence=confidence,
            page_width=page_width,
            page_height=page_height,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_gaps(
    normalized: np.ndarray,
    threshold: float,
    min_gap_px: int,
) -> List[Tuple[float, float]]:
    """Return list of (x_start, x_end) gap regions wider than min_gap_px."""
    below = normalized < threshold
    gaps: List[Tuple[float, float]] = []
    in_gap = False
    gap_start = 0

    for i, is_below in enumerate(below):
        if is_below and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_below and in_gap:
            gap_width = i - gap_start
            if gap_width >= min_gap_px:
                gaps.append((float(gap_start), float(i)))
            in_gap = False

    # Handle gap that extends to end of array
    if in_gap:
        gap_width = len(normalized) - gap_start
        if gap_width >= min_gap_px:
            gaps.append((float(gap_start), float(len(normalized))))

    return gaps


def _build_columns(
    gaps: List[Tuple[float, float]],
    page_width: int,
    min_col_px: int,
) -> List[Tuple[float, float]]:
    """Build column regions from gap list. Returns empty list if any column is too narrow.

    Margin gaps (those touching x=0 or x=page_width) are treated as part of the
    adjacent column rather than as column separators. Only interior gaps divide columns.
    """
    pw = float(page_width)
    # Only gaps that lie strictly inside the page create column boundaries
    inner_gaps = [(s, e) for s, e in gaps if s > 0 and e < pw]

    if not inner_gaps:
        return []  # No interior separators → not multi-column

    edges = [0.0]
    for g_start, g_end in inner_gaps:
        edges.append(g_start)
        edges.append(g_end)
    edges.append(pw)

    columns: List[Tuple[float, float]] = []
    i = 0
    while i + 1 < len(edges):
        col_start = edges[i]
        col_end = edges[i + 1]
        if col_end - col_start < min_col_px:
            return []  # Column too narrow — not a real multi-column layout
        columns.append((col_start, col_end))
        i += 2  # Skip the gap pair

    return columns


def _gap_confidence(
    normalized: np.ndarray,
    gaps: List[Tuple[float, float]],
) -> float:
    """Confidence that gaps are real: 1 - mean coverage in gap regions."""
    total_coverage = 0.0
    total_px = 0
    for g_start, g_end in gaps:
        s, e = int(g_start), int(g_end)
        if e > s:
            total_coverage += float(normalized[s:e].sum())
            total_px += e - s
    if total_px == 0:
        return 0.0
    mean_gap_coverage = total_coverage / total_px
    return 1.0 - mean_gap_coverage


# ── Reading order sort ─────────────────────────────────────────────────────────

def sort_elements_by_column_order(
    elements: list,
    layout: ColumnLayoutResult,
) -> list:
    """
    Sort StructuralElements into column reading order.

    For single-column layouts: sort by y_min (top to bottom).
    For multi-column layouts: sort by (column_index, y_min) so all elements in
    column 0 come before column 1, in top-to-bottom order within each column.
    """
    if not layout.is_multi_column or not elements:
        return sorted(elements, key=lambda e: (e.bbox.y_min if e.bbox else 0))

    def sort_key(elem):
        if elem.bbox is None:
            return (0, 0.0)
        x_center = (elem.bbox.x_min + elem.bbox.x_max) / 2.0
        col_idx = layout.column_index_for_x(x_center)
        return (col_idx, elem.bbox.y_min)

    return sorted(elements, key=sort_key)
