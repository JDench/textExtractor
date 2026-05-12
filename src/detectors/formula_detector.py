"""
Formula/Equation Detection Module

Detects mathematical content from OCR results and (optionally) image regions:
- FORMULA  : Mathematical expression identified by math-character density.
- EQUATION : Numbered formula with a reference label (e.g. "(2.5)").

Detection layers (applied in order, each adding information):
1. Text-pattern layer (always active)
   - Counts Unicode math operators and Greek letters in each OCR result.
   - Detects equation-number patterns at line boundaries.
   - Classifies display-style vs. inline by horizontal centering.
   - Extracts single-letter variable names heuristically.

2. pix2tex layer (active when ``pix2tex`` is installed)
   - Crops the formula region from the original image.
   - Runs the pre-trained LatexOCR model to produce a LaTeX string.
   - Stored in FormulaExpression.latex.

3. sympy validation layer (active when ``sympy`` is installed)
   - Attempts to parse the pix2tex LaTeX output with sympy.
   - If parsing succeeds, replaces the heuristic variable list with
     sympy's free-symbol set (more accurate).
   - If parsing fails, the LaTeX string is discarded rather than stored
     in an invalid state.

Install optional dependencies:
    pip install pix2tex[cli]   # image → LaTeX OCR
    pip install sympy          # LaTeX parsing and validation

Follows Config + Detector + Trace pattern (ARCHITECTURAL_DECISIONS.md).
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging

from data_models import (
    BoundingBox,
    ElementType,
    EquationReference,
    FormulaExpression,
    OCRTextResult,
    PSMMode,
    StructuralElement,
)
from ocr_engine import OCREngine, OCREngineConfig


logger = logging.getLogger(__name__)


# ── Optional heavy dependencies ────────────────────────────────────────────────

try:
    from pix2tex.cli import LatexOCR as _LatexOCR   # type: ignore[import]
    import PIL.Image as _PILImage                    # type: ignore[import]
    _PIX2TEX_AVAILABLE = True
    logger.info("pix2tex available — image→LaTeX conversion enabled")
except ImportError:
    _PIX2TEX_AVAILABLE = False
    logger.debug("pix2tex not installed; image→LaTeX conversion disabled")

try:
    from sympy.parsing.latex import parse_latex as _parse_latex  # type: ignore[import]
    _SYMPY_AVAILABLE = True
    logger.info("sympy available — LaTeX validation enabled")
except ImportError:
    _SYMPY_AVAILABLE = False
    logger.debug("sympy not installed; LaTeX validation disabled")


# ── Math character sets ────────────────────────────────────────────────────────

# Core Unicode math operators and relations
_MATH_OPERATORS: Set[str] = set(
    "=+−×÷"          # = + − × ÷
    "∑∏∫∂∞" # ∑ ∏ ∫ ∂ ∞
    "√±≤≥≠" # √ ± ≤ ≥ ≠
    "≈∝∈∉"       # ≈ ∝ ∈ ∉
    "⊂⊃∩∪"       # ⊂ ⊃ ∩ ∪
)

# Greek lowercase and uppercase letters
_GREEK_LOWER: Set[str] = set("αβγδεζηθ"
                               "ιλμνξπρσ"
                               "τυφχψω")
_GREEK_UPPER: Set[str] = set("ΑΒΓΔΕΖΗΘ"
                               "ΙΛΜΝΞΠΡΣ"
                               "ΤΥΦΧΨΩ")

_MATH_SYMBOLS: Set[str] = _MATH_OPERATORS | _GREEK_LOWER | _GREEK_UPPER

# Equation number patterns: checked at or near line boundaries
_EQ_NUM_PATTERNS: List[re.Pattern] = [
    re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*$"),            # (2.5) at end
    re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\]\s*$"),            # [2.5] at end
    re.compile(r"\bEq(?:uation)?\.?\s*\(?(\d+(?:\.\d+)?)\)?"),# Eq. 3 / Eq.(3)
]

# Heuristic variable extraction: isolated single-letter tokens
_VAR_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")

# Padding (pixels) added around a formula bbox before pix2tex crop
_CROP_PADDING = 8


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class FormulaDetectorConfig:
    """
    Configuration for formula and equation detection.

    Attributes:
        detect_formulas          : Emit FORMULA elements.
        detect_equations         : Emit EQUATION elements for numbered formulas.
        min_math_chars           : Minimum number of math-symbol characters an
                                   OCR result must contain to be a formula
                                   candidate. Default 2.
        attempt_latex_conversion : When pix2tex is installed, crop the image
                                   region and run LaTeX OCR. Default True.
        validate_latex           : When sympy is installed, parse and validate
                                   the pix2tex output; discard on failure.
                                   Default True.
        display_style_margin     : Maximum fraction of image width by which the
                                   formula's horizontal midpoint may differ from
                                   the page centre and still be considered
                                   display-style (centred). Default 0.15.
        crop_padding             : Extra pixels added on each side when cropping
                                   a formula region for pix2tex. Default 8.
        min_confidence           : Minimum OCR confidence for candidates.
        language                 : Tesseract language string.
        enable_preprocessing     : Apply image preprocessing when re-running OCR.
    """
    detect_formulas: bool = True
    detect_equations: bool = True
    min_math_chars: int = 2
    attempt_latex_conversion: bool = True
    validate_latex: bool = True
    display_style_margin: float = 0.15
    crop_padding: int = _CROP_PADDING
    min_confidence: float = 0.3
    language: str = "eng"
    enable_preprocessing: bool = True

    def __post_init__(self):
        if self.min_math_chars < 1:
            raise ValueError("min_math_chars must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.display_style_margin <= 0.5:
            raise ValueError("display_style_margin must be in [0, 0.5]")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be >= 0")


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class FormulaDetectionTrace:
    """Processing trace for formula/equation detection."""
    config: FormulaDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]

    formulas_found: int = 0
    equations_found: int = 0
    latex_converted: int = 0    # pix2tex conversions that produced output
    latex_validated: int = 0    # sympy parses that succeeded
    latex_rejected: int = 0     # sympy parses that failed (LaTeX discarded)
    ocr_results_analyzed: int = 0
    pix2tex_available: bool = _PIX2TEX_AVAILABLE
    sympy_available: bool = _SYMPY_AVAILABLE

    @property
    def total_processing_time_ms(self) -> float:
        return (self.processing_end - self.processing_start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "detect_formulas": self.config.detect_formulas,
                "detect_equations": self.config.detect_equations,
                "min_math_chars": self.config.min_math_chars,
                "attempt_latex_conversion": self.config.attempt_latex_conversion,
                "validate_latex": self.config.validate_latex,
            },
            "capabilities": {
                "pix2tex_available": self.pix2tex_available,
                "sympy_available": self.sympy_available,
            },
            "results": {
                "formulas": self.formulas_found,
                "equations": self.equations_found,
                "latex_converted": self.latex_converted,
                "latex_validated": self.latex_validated,
                "latex_rejected": self.latex_rejected,
            },
            "analysis": {"ocr_results_analyzed": self.ocr_results_analyzed},
            "timing_ms": {"total": self.total_processing_time_ms},
        }


# ── Detector ───────────────────────────────────────────────────────────────────

class FormulaDetector:
    """
    Detects FORMULA and EQUATION elements from OCR results.

    The pix2tex model is loaded lazily on first use; the first call that
    triggers it will be slower due to model weight loading.

    Usage::

        config   = FormulaDetectorConfig(min_math_chars=2)
        detector = FormulaDetector(config)
        elements, trace = detector.detect(image, page_number=1,
                                          ocr_results=ocr_results)

        for elem in elements:
            expr = elem.content   # FormulaExpression or EquationReference
            print(elem.element_type, expr.raw_text, expr.latex)
    """

    def __init__(self, config: Optional[FormulaDetectorConfig] = None):
        self.config = config or FormulaDetectorConfig()
        self._latex_ocr = None          # pix2tex model, loaded on first use
        self._ocr_engine: Optional[OCREngine] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        page_number: int = 1,
        ocr_results: Optional[List[OCRTextResult]] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], FormulaDetectionTrace]:
        """
        Detect formulas and equations in *image*.

        Args:
            image       : Image as numpy array (BGR).
            page_number : Page number for element metadata.
            ocr_results : Pre-extracted OCR results.  When None the detector
                          runs its own OCR pass.
            image_path  : Optional path string for log messages.

        Returns:
            Tuple of (List[StructuralElement], FormulaDetectionTrace).
        """
        start_time = datetime.now()
        image_h, image_w = image.shape[:2]

        if ocr_results is None:
            logger.debug("No OCR results provided; running internal OCR pass")
            ocr_results, _ = self._get_ocr_engine().extract_text(
                image, page_number, image_path=image_path
            )

        elements: List[StructuralElement] = []
        latex_converted = 0
        latex_validated = 0
        latex_rejected  = 0
        elem_counter    = 0

        for result in ocr_results:
            if result.confidence < self.config.min_confidence:
                continue

            text = result.text
            math_count = sum(1 for ch in text if ch in _MATH_SYMBOLS)
            if math_count < self.config.min_math_chars:
                continue

            # ── Equation number ────────────────────────────────────────────────
            eq_number = self._detect_equation_number(text)

            # ── Display-style classification ───────────────────────────────────
            bbox_cx = (result.bbox.x_min + result.bbox.x_max) / 2.0
            offset   = abs(bbox_cx - image_w / 2.0) / image_w
            is_display = offset <= self.config.display_style_margin

            # ── Variable extraction (heuristic) ───────────────────────────────
            variables = self._extract_variables(text)

            # ── pix2tex: image → LaTeX ─────────────────────────────────────────
            latex: Optional[str] = None
            if self.config.attempt_latex_conversion and _PIX2TEX_AVAILABLE:
                latex = self._extract_latex_from_region(image, result.bbox)
                if latex:
                    latex_converted += 1

            # ── sympy: validate and improve variable list ──────────────────────
            if latex and self.config.validate_latex and _SYMPY_AVAILABLE:
                sympy_vars = self._validate_and_extract(latex)
                if sympy_vars is not None:
                    latex_validated += 1
                    variables = sympy_vars   # more accurate than heuristic
                else:
                    latex_rejected += 1
                    logger.debug("sympy rejected LaTeX (discarding): %.80s", latex)
                    latex = None

            # ── Build FormulaExpression ────────────────────────────────────────
            formula_expr = FormulaExpression(
                raw_text=text,
                bbox=result.bbox,
                confidence=result.confidence,
                latex=latex,
                is_displaystyle=is_display,
                variables=variables,
            )

            proc_method = "formula_detector_text_pattern"
            if latex:
                proc_method += "+pix2tex"
                if self.config.validate_latex and _SYMPY_AVAILABLE:
                    proc_method += "+sympy"

            # ── EQUATION vs FORMULA ────────────────────────────────────────────
            if eq_number and self.config.detect_equations:
                content = EquationReference(
                    formula=formula_expr,
                    equation_number=eq_number,
                )
                el_type = ElementType.EQUATION
                el_id   = f"equation_{page_number}_{elem_counter}"
                metadata: Dict[str, Any] = {
                    "equation_number": eq_number,
                    "is_displaystyle": is_display,
                    "latex_available": latex is not None,
                    "math_char_count": math_count,
                }
            elif self.config.detect_formulas:
                content = formula_expr
                el_type = ElementType.FORMULA
                el_id   = f"formula_{page_number}_{elem_counter}"
                metadata = {
                    "is_displaystyle": is_display,
                    "latex_available": latex is not None,
                    "math_char_count": math_count,
                    "variables": variables,
                }
            else:
                continue   # neither type is enabled

            elements.append(StructuralElement(
                element_id=el_id,
                element_type=el_type,
                content=content,
                bbox=result.bbox,
                confidence=result.confidence,
                page_number=page_number,
                metadata=metadata,
                processing_method=proc_method,
            ))
            elem_counter += 1

        end_time = datetime.now()
        trace = FormulaDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions=(image_w, image_h),
            formulas_found=sum(1 for e in elements if e.element_type == ElementType.FORMULA),
            equations_found=sum(1 for e in elements if e.element_type == ElementType.EQUATION),
            latex_converted=latex_converted,
            latex_validated=latex_validated,
            latex_rejected=latex_rejected,
            ocr_results_analyzed=len(ocr_results),
        )

        logger.info(
            "Formula detection complete: %d formula(s), %d equation(s), "
            "%d LaTeX converted, %d validated, %d rejected in %.1fms",
            trace.formulas_found, trace.equations_found,
            trace.latex_converted, trace.latex_validated,
            trace.latex_rejected, trace.total_processing_time_ms,
        )
        return elements, trace

    # ── Text-pattern helpers ───────────────────────────────────────────────────

    def _detect_equation_number(self, text: str) -> Optional[str]:
        """Return equation number string if text ends with a reference tag."""
        for pattern in _EQ_NUM_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1)
        return None

    def _extract_variables(self, text: str) -> List[str]:
        """
        Heuristically extract single-letter variable names from OCR text.

        Only isolated letters (not adjacent to other letters) are considered,
        which filters out common English words while capturing math variables.
        """
        return sorted(set(m.group(1) for m in _VAR_RE.finditer(text)))

    # ── pix2tex helpers ────────────────────────────────────────────────────────

    def _get_latex_ocr(self):
        """Lazy-load the pix2tex LatexOCR model (downloads weights on first call)."""
        if self._latex_ocr is None and _PIX2TEX_AVAILABLE:
            logger.info("Loading pix2tex LatexOCR model (first-use download may take a moment)…")
            self._latex_ocr = _LatexOCR()
        return self._latex_ocr

    def _extract_latex_from_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> Optional[str]:
        """
        Crop *bbox* from *image*, pad it, convert to PIL, and run pix2tex.

        Returns the LaTeX string on success, None on any failure.
        """
        model = self._get_latex_ocr()
        if model is None:
            return None

        try:
            import cv2   # already a project dependency

            pad = self.config.crop_padding
            h, w = image.shape[:2]
            x1 = max(0, int(bbox.x_min) - pad)
            y1 = max(0, int(bbox.y_min) - pad)
            x2 = min(w, int(bbox.x_max) + pad)
            y2 = min(h, int(bbox.y_max) + pad)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = image[y1:y2, x1:x2]
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            pil_img = _PILImage.fromarray(crop)
            latex = model(pil_img)
            return latex.strip() if latex else None

        except Exception as exc:
            logger.debug("pix2tex extraction failed: %s", exc)
            return None

    # ── sympy helpers ──────────────────────────────────────────────────────────

    def _validate_and_extract(self, latex: str) -> Optional[List[str]]:
        """
        Parse *latex* with sympy.

        Returns sorted list of free-symbol names on success, None on failure.
        """
        try:
            expr = _parse_latex(latex)
            return sorted(str(s) for s in expr.free_symbols)
        except Exception:
            return None

    # ── Internal OCR (lazy) ────────────────────────────────────────────────────

    def _get_ocr_engine(self) -> OCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(OCREngineConfig(
                psm_modes=[PSMMode.SPARSE_TEXT, PSMMode.FULLY_AUTOMATIC],
                languages=self.config.language,
                enable_preprocessing=self.config.enable_preprocessing,
                min_confidence=self.config.min_confidence,
            ))
        return self._ocr_engine
