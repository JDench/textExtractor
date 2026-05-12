"""
Header/Footer Detection Module

Detects page-structure elements from OCR results using spatial zone analysis:
- HEADER      : Text in the top zone of the page
- FOOTER      : Text in the bottom zone of the page
- PAGE_NUMBER : Numeric page indicator found inside a header or footer zone

Additionally scans header/footer text for date values and attempts to
classify each date as a published date, a modified/revised date, or an
unclassified date.  Classification is optional and missing context is
handled gracefully — a date without surrounding keywords is still reported
with date_type="unknown".

Design:
- Re-uses pre-computed OCR results when provided (no extra OCR pass)
- Spatial thresholds are configurable fractions of image height
- Date classification uses a bidirectional keyword window search
- Follows Config + Detector + Trace pattern
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from data_models import (
    BoundingBox,
    ElementType,
    OCRTextResult,
    PageFooter,
    PageHeader,
    PSMMode,
    StructuralElement,
)
from ocr_engine import OCREngine, OCREngineConfig


logger = logging.getLogger(__name__)


# ── Date regex building blocks ─────────────────────────────────────────────────

_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
_DAY   = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH = r"(?:0?[1-9]|1[0-2])"
_YEAR4 = r"(?:19|20)\d{2}"

# Ordered from most-specific to least-specific to avoid premature short matches
_DATE_PATTERNS: List[re.Pattern] = [
    re.compile(rf"\b{_YEAR4}[/\-]{_MONTH}[/\-]{_DAY}\b"),         # ISO: 2024-01-15
    re.compile(rf"\b{_MONTH}/{_DAY}/{_YEAR4}\b"),                   # US:  01/15/2024
    re.compile(rf"\b{_DAY}\.{_MONTH}\.{_YEAR4}\b"),                # EU:  15.01.2024
    re.compile(rf"\b{_MONTH_NAME}\s+{_DAY},?\s+{_YEAR4}\b"),       # Jan 15, 2024
    re.compile(rf"\b{_DAY}\s+{_MONTH_NAME}\s+{_YEAR4}\b"),         # 15 Jan 2024
    re.compile(rf"\b{_MONTH_NAME}\s+{_YEAR4}\b"),                   # Jan 2024
]

_PUBLISHED_RE = re.compile(
    r"\b(?:published|pub\.?|publication|issued|released|effective|printed|date\s*:)\b",
    re.IGNORECASE,
)
_MODIFIED_RE = re.compile(
    r"\b(?:modified|updated|revised|last\s+updated|last\s+modified|"
    r"version|amended|edited|last\s+revised)\b",
    re.IGNORECASE,
)

# Page number: optional "Page" prefix, mandatory digits, optional "of N" suffix
_PAGE_NUM_RE = re.compile(
    r"(?:(?:page|pg)\.?\s+)?(?P<num>\d{1,4})(?:\s*(?:of|/)\s*\d{1,4})?",
    re.IGNORECASE,
)

# Context window size (chars) searched around a date match for keywords
_DATE_CONTEXT_WINDOW = 70


# ── Local domain types ─────────────────────────────────────────────────────────

@dataclass
class DateInfo:
    """
    A date value detected within a header or footer.

    Attributes:
        date_str        : Raw matched date string from OCR text.
        date_type       : "published", "modified", or "unknown".
                          "unknown" means a date was found but no keyword
                          context was present to classify it.
        confidence      : Detection confidence (0-1).
                          Higher when a classification keyword is nearby.
        context_keyword : The keyword that triggered classification, if any.
                          None when date_type is "unknown".
    """
    date_str: str
    date_type: str
    confidence: float
    context_keyword: Optional[str] = None


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class HeaderFooterDetectorConfig:
    """
    Configuration for header/footer/date detection.

    Attributes:
        header_zone_ratio : Fraction of image height that defines the header
                            zone (measured from the top). Default 0.10 (10%).
        footer_zone_ratio : Fraction of image height that defines the footer
                            zone (measured from the bottom). Default 0.10.
        detect_headers    : Enable HEADER element detection.
        detect_footers    : Enable FOOTER element detection.
        detect_page_numbers : Detect and emit PAGE_NUMBER elements.
        detect_dates      : Scan header/footer text for date values.
        min_confidence    : Minimum OCR confidence for included results.
        language          : Tesseract language string.
        enable_preprocessing : Apply preprocessing before OCR (when re-running).
    """
    header_zone_ratio: float = 0.10
    footer_zone_ratio: float = 0.10
    detect_headers: bool = True
    detect_footers: bool = True
    detect_page_numbers: bool = True
    detect_dates: bool = True
    min_confidence: float = 0.3
    language: str = "eng"
    enable_preprocessing: bool = True

    def __post_init__(self):
        if not 0.0 < self.header_zone_ratio <= 0.5:
            raise ValueError("header_zone_ratio must be in (0, 0.5]")
        if not 0.0 < self.footer_zone_ratio <= 0.5:
            raise ValueError("footer_zone_ratio must be in (0, 0.5]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class HeaderFooterDetectionTrace:
    """Processing trace for header/footer detection."""
    config: HeaderFooterDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]

    headers_found: int = 0
    footers_found: int = 0
    page_numbers_found: int = 0
    dates_found: int = 0
    dates_classified: int = 0   # dates with a non-"unknown" type
    ocr_results_analyzed: int = 0

    @property
    def total_processing_time_ms(self) -> float:
        return (self.processing_end - self.processing_start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "header_zone_ratio": self.config.header_zone_ratio,
                "footer_zone_ratio": self.config.footer_zone_ratio,
                "detect_dates": self.config.detect_dates,
            },
            "results": {
                "headers": self.headers_found,
                "footers": self.footers_found,
                "page_numbers": self.page_numbers_found,
                "dates_found": self.dates_found,
                "dates_classified": self.dates_classified,
            },
            "analysis": {"ocr_results_analyzed": self.ocr_results_analyzed},
            "timing_ms": {"total": self.total_processing_time_ms},
        }


# ── Detector ───────────────────────────────────────────────────────────────────

class HeaderFooterDetector:
    """
    Detects HEADER, FOOTER, PAGE_NUMBER, and date annotations from OCR results.

    Spatial strategy:
    - OCR results whose bottom edge falls within the top ``header_zone_ratio``
      fraction of the image become HEADER candidates.
    - OCR results whose top edge falls within the bottom ``footer_zone_ratio``
      fraction of the image become FOOTER candidates.

    Date strategy:
    - The combined text of each zone is searched for date patterns.
    - A ±``_DATE_CONTEXT_WINDOW``-char window around each match is scanned for
      published/modified keywords to classify the date.
    - Detected dates are stored in the StructuralElement's ``metadata`` dict
      so that the domain model (PageHeader/PageFooter) stays unchanged.

    Usage::

        config   = HeaderFooterDetectorConfig(header_zone_ratio=0.08)
        detector = HeaderFooterDetector(config)
        elements, trace = detector.detect(image, page_number=1,
                                          ocr_results=ocr_results)
    """

    def __init__(self, config: Optional[HeaderFooterDetectorConfig] = None):
        self.config = config or HeaderFooterDetectorConfig()
        self._ocr_engine: Optional[OCREngine] = None   # created lazily

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        page_number: int = 1,
        ocr_results: Optional[List[OCRTextResult]] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], HeaderFooterDetectionTrace]:
        """
        Detect headers, footers, page numbers, and dates.

        Args:
            image       : Image as numpy array (BGR).
            page_number : Page number for element metadata.
            ocr_results : Pre-extracted OCR results.  When None the detector
                          runs its own OCR pass (PSM SINGLE_COLUMN then AUTO).
            image_path  : Optional path string used for log messages only.

        Returns:
            Tuple of (List[StructuralElement], HeaderFooterDetectionTrace).
        """
        start_time = datetime.now()
        image_h, image_w = image.shape[:2]

        if ocr_results is None:
            logger.debug("No OCR results provided; running internal OCR pass")
            ocr_results, _ = self._get_ocr_engine().extract_text(
                image, page_number, image_path=image_path
            )

        header_cutoff = image_h * self.config.header_zone_ratio
        footer_cutoff = image_h * (1.0 - self.config.footer_zone_ratio)

        header_results = [r for r in ocr_results if r.bbox.y_max <= header_cutoff
                          and r.confidence >= self.config.min_confidence]
        footer_results = [r for r in ocr_results if r.bbox.y_min >= footer_cutoff
                          and r.confidence >= self.config.min_confidence]

        elements: List[StructuralElement] = []
        elem_counter = 0
        total_dates = 0
        total_classified = 0
        total_page_numbers = 0

        def _process_zone(zone_results: List[OCRTextResult], zone: str) -> None:
            nonlocal elem_counter, total_dates, total_classified, total_page_numbers
            if not zone_results:
                return

            combined_text = " ".join(r.text for r in zone_results)
            avg_conf = sum(r.confidence for r in zone_results) / len(zone_results)

            bbox = BoundingBox(
                x_min=min(r.bbox.x_min for r in zone_results),
                y_min=min(r.bbox.y_min for r in zone_results),
                x_max=max(r.bbox.x_max for r in zone_results),
                y_max=max(r.bbox.y_max for r in zone_results),
            )

            # ── Page number detection ──────────────────────────────────────────
            page_num_val: Optional[int] = None
            pn_result: Optional[OCRTextResult] = None
            if self.config.detect_page_numbers:
                page_num_val = self._detect_page_number(combined_text)
                if page_num_val is not None:
                    pn_result = self._find_page_number_result(zone_results, page_num_val)
                    total_page_numbers += 1

            # ── Date detection ─────────────────────────────────────────────────
            dates: List[DateInfo] = []
            if self.config.detect_dates:
                dates = self._detect_dates(combined_text)
                total_dates += len(dates)
                total_classified += sum(1 for d in dates if d.date_type != "unknown")

            # ── Build metadata ─────────────────────────────────────────────────
            metadata: Dict[str, Any] = {"zone": zone}
            if page_num_val is not None:
                metadata["page_number_value"] = page_num_val
            if dates:
                metadata["dates"] = [
                    {
                        "date_str": d.date_str,
                        "date_type": d.date_type,
                        "confidence": d.confidence,
                        "context_keyword": d.context_keyword,
                    }
                    for d in dates
                ]

            # ── Domain object and StructuralElement ────────────────────────────
            if zone == "header":
                domain_obj = PageHeader(
                    content=combined_text,
                    bbox=bbox,
                    page_number=page_number,
                    confidence=avg_conf,
                    includes_page_number=(page_num_val is not None),
                )
                el_type = ElementType.HEADER
                el_id = f"header_{page_number}_{elem_counter}"
            else:
                domain_obj = PageFooter(
                    content=combined_text,
                    bbox=bbox,
                    page_number=page_number,
                    confidence=avg_conf,
                    includes_page_number=(page_num_val is not None),
                )
                el_type = ElementType.FOOTER
                el_id = f"footer_{page_number}_{elem_counter}"

            elements.append(StructuralElement(
                element_id=el_id,
                element_type=el_type,
                content=domain_obj,
                bbox=bbox,
                confidence=avg_conf,
                page_number=page_number,
                metadata=metadata,
                processing_method="header_footer_detector_spatial",
            ))
            elem_counter += 1

            # ── PAGE_NUMBER child element ──────────────────────────────────────
            if pn_result is not None:
                elements.append(StructuralElement(
                    element_id=f"page_number_{page_number}_{elem_counter}",
                    element_type=ElementType.PAGE_NUMBER,
                    content=str(page_num_val),
                    bbox=pn_result.bbox,
                    confidence=pn_result.confidence,
                    page_number=page_number,
                    parent_id=el_id,
                    metadata={"page_number_value": page_num_val, "zone": zone},
                    processing_method="header_footer_detector_page_number",
                ))
                elem_counter += 1

        if self.config.detect_headers:
            _process_zone(header_results, "header")
        if self.config.detect_footers:
            _process_zone(footer_results, "footer")

        end_time = datetime.now()
        trace = HeaderFooterDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions=(image_w, image_h),
            headers_found=sum(1 for e in elements if e.element_type == ElementType.HEADER),
            footers_found=sum(1 for e in elements if e.element_type == ElementType.FOOTER),
            page_numbers_found=total_page_numbers,
            dates_found=total_dates,
            dates_classified=total_classified,
            ocr_results_analyzed=len(ocr_results),
        )

        logger.info(
            "Header/footer detection complete: %d header(s), %d footer(s), "
            "%d page number(s), %d date(s) (%d classified) in %.1fms",
            trace.headers_found, trace.footers_found,
            trace.page_numbers_found, trace.dates_found,
            trace.dates_classified, trace.total_processing_time_ms,
        )
        return elements, trace

    # ── Date helpers ───────────────────────────────────────────────────────────

    def _detect_dates(self, text: str) -> List[DateInfo]:
        """
        Find all date strings in *text* and classify each as published,
        modified, or unknown.

        Classification uses a bidirectional keyword window: the
        ``_DATE_CONTEXT_WINDOW`` characters before and after each match are
        scanned for published/modified keywords.  When both keyword types
        appear in the window the nearer one wins; when neither is present
        date_type is "unknown" and context_keyword is None.
        """
        results: List[DateInfo] = []
        seen_spans: List[Tuple[int, int]] = []   # avoid duplicate matches

        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                span = match.span()
                # Skip if this span overlaps a match already captured by a
                # more-specific pattern
                if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                    continue
                seen_spans.append(span)

                date_str = match.group(0)
                w_start = max(0, match.start() - _DATE_CONTEXT_WINDOW)
                w_end   = min(len(text), match.end() + _DATE_CONTEXT_WINDOW)
                window  = text[w_start:w_end]

                pub_m = _PUBLISHED_RE.search(window)
                mod_m = _MODIFIED_RE.search(window)

                if pub_m and not mod_m:
                    date_type  = "published"
                    keyword    = pub_m.group(0)
                    confidence = 0.88
                elif mod_m and not pub_m:
                    date_type  = "modified"
                    keyword    = mod_m.group(0)
                    confidence = 0.88
                elif pub_m and mod_m:
                    # Both found; pick the nearer keyword relative to the date
                    match_pos_in_window = match.start() - w_start
                    pub_dist = abs(pub_m.start() - match_pos_in_window)
                    mod_dist = abs(mod_m.start() - match_pos_in_window)
                    if pub_dist <= mod_dist:
                        date_type, keyword = "published", pub_m.group(0)
                    else:
                        date_type, keyword = "modified", mod_m.group(0)
                    confidence = 0.72   # slightly lower — ambiguous context
                else:
                    date_type  = "unknown"
                    keyword    = None
                    confidence = 0.58

                results.append(DateInfo(
                    date_str=date_str,
                    date_type=date_type,
                    confidence=confidence,
                    context_keyword=keyword,
                ))

        return results

    def _detect_page_number(self, text: str) -> Optional[int]:
        """Return the numeric page number if *text* contains one, else None."""
        match = _PAGE_NUM_RE.search(text)
        if match:
            try:
                return int(match.group("num"))
            except (ValueError, AttributeError):
                pass
        return None

    def _find_page_number_result(
        self,
        results: List[OCRTextResult],
        page_num_val: int,
    ) -> Optional[OCRTextResult]:
        """Return the OCR result whose text contains the page number string."""
        target = str(page_num_val)
        for r in results:
            if target in r.text:
                return r
        return results[0] if results else None   # fallback: first result in zone

    # ── Internal OCR (lazy) ────────────────────────────────────────────────────

    def _get_ocr_engine(self) -> OCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(OCREngineConfig(
                psm_modes=[PSMMode.SINGLE_COLUMN, PSMMode.FULLY_AUTOMATIC],
                languages=self.config.language,
                enable_preprocessing=self.config.enable_preprocessing,
                min_confidence=self.config.min_confidence,
            ))
        return self._ocr_engine
