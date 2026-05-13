"""
Barcode / QR-Code Detection Module

Detects machine-readable markers in document images and decodes their
values.  Two decoding backends are supported, tried in priority order:

  1. pyzbar  (optional)  — Decodes QR codes, Code128, EAN-13, UPC-A, PDF417,
                           DataMatrix, and dozens of other formats.
                           Install with: pip install pyzbar
                           (Requires the system libzbar shared library.)

  2. cv2.QRCodeDetector (built-in) — Decodes QR codes only, using OpenCV's
                           bundled QR decoder.  Used when pyzbar is absent
                           or returns no results for a region.

If neither backend successfully decodes a symbol, no element is emitted for
that region.  The trace records which backends are available so callers can
understand coverage.

Barcode confidence is treated as binary: 1.0 for a successful decode (the
value IS the decoded content, which is either right or wrong), 0.0 for a
failed decode.  No BARCODE element is emitted for failed decodes.

Follows Config + Detector + Trace pattern.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np

from data_models import (
    Barcode,
    BoundingBox,
    ElementType,
    OCRTextResult,
    StructuralElement,
)

logger = logging.getLogger(__name__)


# ── Optional pyzbar ────────────────────────────────────────────────────────────

try:
    from pyzbar import pyzbar as _pyzbar  # type: ignore[import]
    _PYZBAR_AVAILABLE = True
except ImportError:
    _pyzbar = None  # type: ignore[assignment]
    _PYZBAR_AVAILABLE = False


# ── Lazy cv2 QR detector ───────────────────────────────────────────────────────

_CV2_QR: Optional[cv2.QRCodeDetector] = None


def _get_cv2_qr() -> cv2.QRCodeDetector:
    global _CV2_QR
    if _CV2_QR is None:
        _CV2_QR = cv2.QRCodeDetector()
    return _CV2_QR


@dataclass
class BarcodeDetectorConfig:
    """Configuration for BarcodeDetector."""

    # Use pyzbar when available (handles many barcode formats beyond QR)
    use_pyzbar: bool = True

    # Fall back to cv2.QRCodeDetector when pyzbar is absent or finds nothing
    use_cv2_qr_fallback: bool = True

    # Minimum bounding box area (pixels²) for a candidate decode region.
    # Very small regions are likely OCR noise, not real barcodes.
    min_barcode_area: int = 400

    # Confidence assigned to successfully decoded barcodes.
    decode_confidence: float = 0.95

    # Whether to store the raw cropped barcode image bytes in Barcode.raw_image.
    store_raw_image: bool = False


@dataclass
class BarcodeDetectionTrace:
    """Records what BarcodeDetector.detect() found and how."""
    barcodes_found: int = 0
    pyzbar_available: bool = False
    cv2_qr_available: bool = True
    pyzbar_decodes: int = 0
    cv2_qr_decodes: int = 0
    decode_failures: int = 0
    processing_time_seconds: float = 0.0
    config: Optional[BarcodeDetectorConfig] = None


class BarcodeDetector:
    """
    Detects and decodes barcodes and QR codes in document images.

    Args:
        config: BarcodeDetectorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[BarcodeDetectorConfig] = None) -> None:
        self.config = config or BarcodeDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int = 1,
    ) -> Tuple[List[StructuralElement], BarcodeDetectionTrace]:
        """
        Detect and decode barcodes in an image.

        Args:
            image:       BGR (or grayscale) NumPy array.
            ocr_results: Pre-computed OCR results (used for fallback region
                         hints, not for text content — barcode values come
                         from the decode backend).
            page_number: Page index for emitted StructuralElements.

        Returns:
            Tuple of (elements, trace).
        """
        t0 = time.perf_counter()
        cfg = self.config
        trace = BarcodeDetectionTrace(
            pyzbar_available=_PYZBAR_AVAILABLE,
            config=cfg,
        )
        elements: List[StructuralElement] = []

        # Ensure BGR for color-sensitive decoders
        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image

        # ── Primary: pyzbar ───────────────────────────────────────────────────
        if cfg.use_pyzbar and _PYZBAR_AVAILABLE:
            decoded_list = _pyzbar.decode(bgr)  # type: ignore[union-attr]
            for decoded in decoded_list:
                elem = self._make_element_from_pyzbar(decoded, bgr, page_number, cfg)
                if elem is not None:
                    elements.append(elem)
                    trace.pyzbar_decodes += 1

        # ── Fallback: cv2.QRCodeDetector ──────────────────────────────────────
        if cfg.use_cv2_qr_fallback and (not elements):
            qr_elem = self._try_cv2_qr(bgr, page_number, cfg)
            if qr_elem is not None:
                elements.append(qr_elem)
                trace.cv2_qr_decodes += 1

        trace.barcodes_found = len(elements)
        trace.processing_time_seconds = time.perf_counter() - t0
        return elements, trace

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_element_from_pyzbar(
        self,
        decoded: Any,
        bgr: np.ndarray,
        page_number: int,
        cfg: BarcodeDetectorConfig,
    ) -> Optional[StructuralElement]:
        """Build a StructuralElement from a pyzbar Decoded object."""
        try:
            data_str = decoded.data.decode("utf-8", errors="replace").strip()
        except Exception:
            data_str = repr(decoded.data)

        if not data_str:
            return None

        rect = decoded.rect  # Rect(left, top, width, height)
        x, y, w, h = rect.left, rect.top, rect.width, rect.height

        if w * h < cfg.min_barcode_area:
            return None

        if w <= 0 or h <= 0:
            return None

        bbox = BoundingBox(
            x_min=float(x),
            y_min=float(y),
            x_max=float(x + w),
            y_max=float(y + h),
        )

        raw_bytes: Optional[bytes] = None
        if cfg.store_raw_image:
            roi = bgr[y : y + h, x : x + w]
            ok, buf = cv2.imencode(".png", roi)
            raw_bytes = bytes(buf) if ok else None

        barcode_type = str(decoded.type)
        barcode = Barcode(
            barcode_type=barcode_type,
            decoded_value=data_str,
            bbox=bbox,
            confidence=cfg.decode_confidence,
            raw_image=raw_bytes,
            metadata={"quality": getattr(decoded, "quality", None)},
        )
        return StructuralElement(
            element_id=f"bc_{uuid.uuid4().hex[:8]}",
            element_type=ElementType.BARCODE,
            content=barcode,
            bbox=bbox,
            confidence=cfg.decode_confidence,
            page_number=page_number,
            processing_method="barcode_detector_pyzbar",
        )

    def _try_cv2_qr(
        self,
        bgr: np.ndarray,
        page_number: int,
        cfg: BarcodeDetectorConfig,
    ) -> Optional[StructuralElement]:
        """Attempt QR-code decode using cv2.QRCodeDetector."""
        try:
            detector = _get_cv2_qr()
            decoded_text, points, _ = detector.detectAndDecode(bgr)
        except Exception as exc:
            logger.debug("cv2.QRCodeDetector failed: %s", exc)
            return None

        if not decoded_text or points is None:
            return None

        pts = points.reshape(-1, 2)
        x_min = float(pts[:, 0].min())
        y_min = float(pts[:, 1].min())
        x_max = float(pts[:, 0].max())
        y_max = float(pts[:, 1].max())

        if x_max <= x_min or y_max <= y_min:
            return None
        if (x_max - x_min) * (y_max - y_min) < cfg.min_barcode_area:
            return None

        bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        barcode = Barcode(
            barcode_type="QRCODE",
            decoded_value=decoded_text.strip(),
            bbox=bbox,
            confidence=cfg.decode_confidence,
        )
        return StructuralElement(
            element_id=f"bc_{uuid.uuid4().hex[:8]}",
            element_type=ElementType.BARCODE,
            content=barcode,
            bbox=bbox,
            confidence=cfg.decode_confidence,
            page_number=page_number,
            processing_method="barcode_detector_cv2_qr",
        )
