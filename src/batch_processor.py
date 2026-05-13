"""
Batch Processor

Orchestrates all detectors to process one or many document images and produce
DocumentResult / BatchResult objects ready for export.

Design
------
- One BatchProcessor instance holds all detector instances; they are created
  once in __init__ to avoid repeated startup cost.
- OCR is run once per image with OCREngine (PSM 3) to produce a shared set of
  OCRTextResult objects passed to all detectors that accept them.
- TextDetector, ListDetector, and TableDetector pre-date the standard
  detect(image, ocr_results, page_number) signature; the batch processor
  calls each with its correct API.
- Every detector call is wrapped in try/except so a single-detector failure
  does not abort processing of the rest of the image.
- HierarchyBuilder is run as a post-processing step when build_hierarchy=True.

Usage
-----
    from batch_processor import BatchProcessor, BatchProcessorConfig

    processor = BatchProcessor()
    result = processor.process_image("scan.png")          # -> DocumentResult
    batch  = processor.process_batch(["a.png", "b.png"])  # -> BatchResult
"""

import uuid
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from data_models import (
    BatchResult,
    DocumentMetadata,
    DocumentResult,
    ElementType,
    OCRTextResult,
    ProcessingStatus,
    StructuralElement,
)
from ocr_engine import OCREngine, OCREngineConfig, load_image
from hierarchy_builder import HierarchyBuilder, HierarchyConfig

# ── Detector imports ──────────────────────────────────────────────────────────
from detectors import (
    TextDetector, TextDetectorConfig,
    ListDetector, ListDetectorConfig,
    TableDetector, TableDetectorConfig,
    ContentTableDetector, ContentTableDetectorConfig,
    HeaderFooterDetector, HeaderFooterDetectorConfig,
    FormulaDetector, FormulaDetectorConfig,
    FigureDetector, FigureDetectorConfig,
    AnnotationDetector, AnnotationDetectorConfig,
    WatermarkDetector, WatermarkDetectorConfig,
    BarcodeDetector, BarcodeDetectorConfig,
    CodeBlockDetector, CodeBlockDetectorConfig,
    ReferenceDetector, ReferenceDetectorConfig,
    TOCDetector, TOCDetectorConfig,
    IndexDetector, IndexDetectorConfig,
)
from detectors.layout_detector import LayoutDetector, LayoutDetectorConfig
from language_detector import LanguageDetector, LanguageDetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessorConfig:
    """
    Master configuration for BatchProcessor.

    Each ``detect_*`` flag enables or disables that detector entirely.
    Each ``*_config`` field holds the per-detector configuration object
    (pass None to use that detector's built-in defaults).
    """

    # ── Detection flags ───────────────────────────────────────────────────────
    detect_text: bool = True
    detect_lists: bool = True
    detect_tables_line: bool = True       # bordered tables (Hough transform)
    detect_tables_content: bool = True   # borderless tables (spatial analysis)
    detect_headers_footers: bool = True
    detect_formulas: bool = True
    detect_figures: bool = True
    detect_annotations: bool = True
    detect_watermarks: bool = True
    detect_barcodes: bool = True
    detect_code_blocks: bool = True
    detect_references: bool = True
    detect_toc: bool = True
    detect_index: bool = True
    detect_layout: bool = False           # ML-based layout regions (Sprint 9)

    # ── Post-processing ────────────────────────────────────────────────────────
    build_hierarchy: bool = True

    # ── Per-detector configs (None → use each detector's own defaults) ────────
    text_config: Optional[TextDetectorConfig] = None
    list_config: Optional[ListDetectorConfig] = None
    table_config: Optional[TableDetectorConfig] = None
    content_table_config: Optional[ContentTableDetectorConfig] = None
    header_footer_config: Optional[HeaderFooterDetectorConfig] = None
    formula_config: Optional[FormulaDetectorConfig] = None
    figure_config: Optional[FigureDetectorConfig] = None
    annotation_config: Optional[AnnotationDetectorConfig] = None
    watermark_config: Optional[WatermarkDetectorConfig] = None
    barcode_config: Optional[BarcodeDetectorConfig] = None
    code_block_config: Optional[CodeBlockDetectorConfig] = None
    reference_config: Optional[ReferenceDetectorConfig] = None
    toc_config: Optional[TOCDetectorConfig] = None
    index_config: Optional[IndexDetectorConfig] = None
    layout_config: Optional[LayoutDetectorConfig] = None       # Sprint 9
    language_detector_config: Optional[LanguageDetectorConfig] = None  # Sprint 9
    hierarchy_config: Optional[HierarchyConfig] = None
    ocr_config: Optional[OCREngineConfig] = None

    # ── Global settings ────────────────────────────────────────────────────────
    language: str = "eng"
    # Auto-detect document language from OCR results (Sprint 9)
    auto_detect_language: bool = False
    # Global post-filter: elements with confidence < this are dropped
    min_confidence: float = 0.30
    # When True, raw OCRTextResult objects are stored in DocumentResult
    store_raw_ocr_results: bool = False
    # GPU acceleration for image preprocessing (Sprint 9)
    use_gpu: bool = False


class BatchProcessor:
    """
    Orchestrates all detectors to process document images.

    Args:
        config: BatchProcessorConfig; defaults used when None.
    """

    def __init__(self, config: Optional[BatchProcessorConfig] = None) -> None:
        self.config = config or BatchProcessorConfig()
        self._ocr_engine: Optional[OCREngine] = None
        self._hierarchy_builder: Optional[HierarchyBuilder] = None
        self._language_detector: Optional[LanguageDetector] = None
        self._gpu_available: bool = self._check_gpu_availability() if self.config.use_gpu else False
        self._init_detectors()

    # ── Public API ─────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Pre-initialise the OCR engine, HierarchyBuilder, and LanguageDetector
        so the first call to process_image() does not pay the startup cost.

        Call this once after constructing a BatchProcessor that will be
        reused across many images (e.g. in a long-running service).
        """
        self._get_ocr_engine()
        if self.config.build_hierarchy:
            self._get_hierarchy_builder()
        if self.config.auto_detect_language:
            self._get_language_detector()

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        page_number: int = 1,
    ) -> DocumentResult:
        """
        Process a single image and return a DocumentResult.

        Args:
            image_input: File path (str or Path) or BGR NumPy array.
            page_number: Page number assigned to all extracted elements.

        Returns:
            DocumentResult with all detected elements and metadata.
        """
        t0 = time.perf_counter()

        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            try:
                image = load_image(image_path)
            except Exception as exc:
                return self._make_failed_result(image_path, str(exc), 0.001)
        else:
            image = image_input
            image_path = "in_memory"

        image_h, image_w = image.shape[:2]
        errors: List[str] = []

        # ── GPU preprocessing (when enabled and available) ────────────────────
        if self.config.use_gpu and self._gpu_available:
            image = self._preprocess_image_gpu(image)

        # ── Shared OCR pass ───────────────────────────────────────────────────
        ocr_results: List[OCRTextResult] = []
        try:
            ocr_results, _ = self._get_ocr_engine().extract_text(
                image, page_number=page_number, image_path=image_path
            )
        except Exception as exc:
            errors.append(f"OCR: {exc}")
            logger.error("OCR failed for %s: %s", image_path, exc)

        # ── Language detection ────────────────────────────────────────────────
        detected_language = self.config.language
        if self.config.auto_detect_language and ocr_results:
            try:
                detected_language, _ = self._get_language_detector().detect(ocr_results)
            except Exception as exc:
                errors.append(f"language_detector: {exc}")
                logger.warning("Language detection failed: %s", exc)

        # ── Run all enabled detectors ─────────────────────────────────────────
        all_elements: List[StructuralElement] = []

        all_elements, errors = self._run_all_detectors(
            image, ocr_results, page_number, image_path, errors
        )

        # ── Global confidence filter ──────────────────────────────────────────
        min_conf = self.config.min_confidence
        if min_conf > 0:
            all_elements = [e for e in all_elements if e.confidence >= min_conf]

        # ── Hierarchy building ────────────────────────────────────────────────
        if self.config.build_hierarchy and all_elements:
            try:
                all_elements, _ = self._get_hierarchy_builder().build(all_elements)
            except Exception as exc:
                errors.append(f"hierarchy_builder: {exc}")
                logger.error("HierarchyBuilder failed: %s", exc)

        # ── Build result ──────────────────────────────────────────────────────
        duration = max(time.perf_counter() - t0, 1e-4)
        avg_conf = (
            sum(e.confidence for e in all_elements) / len(all_elements)
            if all_elements else 0.0
        )
        status = (
            ProcessingStatus.COMPLETED if not errors
            else ProcessingStatus.PARTIAL if all_elements
            else ProcessingStatus.FAILED
        )
        doc_id = Path(image_path).stem + "_" + uuid.uuid4().hex[:6]
        metadata = DocumentMetadata(
            source_file=image_path,
            document_id=doc_id,
            processing_timestamp=datetime.now(),
            processing_duration=duration,
            image_dimensions=(image_w, image_h),
            detected_language=detected_language,
            total_elements_extracted=len(all_elements),
            average_confidence=avg_conf,
            processing_status=status,
            errors_encountered=errors,
        )
        raw_ocr = ocr_results if self.config.store_raw_ocr_results else []
        return DocumentResult(
            metadata=metadata,
            elements=all_elements,
            raw_ocr_results=raw_ocr,
        )

    def process_batch(
        self,
        inputs: List[Union[str, Path, np.ndarray]],
        batch_id: Optional[str] = None,
    ) -> BatchResult:
        """
        Process a list of images and return a BatchResult.

        Args:
            inputs:   List of file paths or BGR NumPy arrays.
            batch_id: Optional identifier for this batch; auto-generated if None.

        Returns:
            BatchResult aggregating all DocumentResult objects plus statistics.
        """
        bid = batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        documents: List[DocumentResult] = []

        for i, inp in enumerate(inputs):
            label = str(inp) if isinstance(inp, (str, Path)) else f"image[{i}]"
            logger.info("Processing %s (%d/%d)", label, i + 1, len(inputs))
            try:
                doc = self.process_image(inp, page_number=i + 1)
            except Exception as exc:
                logger.error("Unhandled error processing %s: %s", label, exc)
                doc = self._make_failed_result(label, str(exc), 0.001)
            documents.append(doc)

        return BatchResult(
            batch_id=bid,
            created_at=datetime.now(),
            documents=documents,
        )

    # ── Detector initialisation ────────────────────────────────────────────────

    def _init_detectors(self) -> None:
        cfg = self.config
        # Store each detector or None when disabled
        self._text_detector = TextDetector(cfg.text_config) if cfg.detect_text else None
        self._list_detector = ListDetector(cfg.list_config) if cfg.detect_lists else None
        self._table_detector = TableDetector(cfg.table_config) if cfg.detect_tables_line else None
        self._content_table_detector = (
            ContentTableDetector(cfg.content_table_config) if cfg.detect_tables_content else None
        )
        self._header_footer_detector = (
            HeaderFooterDetector(cfg.header_footer_config) if cfg.detect_headers_footers else None
        )
        self._formula_detector = FormulaDetector(cfg.formula_config) if cfg.detect_formulas else None
        self._figure_detector = FigureDetector(cfg.figure_config) if cfg.detect_figures else None
        self._annotation_detector = (
            AnnotationDetector(cfg.annotation_config) if cfg.detect_annotations else None
        )
        self._watermark_detector = (
            WatermarkDetector(cfg.watermark_config) if cfg.detect_watermarks else None
        )
        self._barcode_detector = BarcodeDetector(cfg.barcode_config) if cfg.detect_barcodes else None
        self._code_block_detector = (
            CodeBlockDetector(cfg.code_block_config) if cfg.detect_code_blocks else None
        )
        self._reference_detector = (
            ReferenceDetector(cfg.reference_config) if cfg.detect_references else None
        )
        self._toc_detector = TOCDetector(cfg.toc_config) if cfg.detect_toc else None
        self._index_detector = IndexDetector(cfg.index_config) if cfg.detect_index else None
        self._layout_detector = (
            LayoutDetector(cfg.layout_config) if cfg.detect_layout else None
        )

    # ── Lazy singletons ────────────────────────────────────────────────────────

    def _get_ocr_engine(self) -> OCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(self.config.ocr_config)
        return self._ocr_engine

    def _get_hierarchy_builder(self) -> HierarchyBuilder:
        if self._hierarchy_builder is None:
            self._hierarchy_builder = HierarchyBuilder(self.config.hierarchy_config)
        return self._hierarchy_builder

    def _get_language_detector(self) -> LanguageDetector:
        if self._language_detector is None:
            self._language_detector = LanguageDetector(self.config.language_detector_config)
        return self._language_detector

    # ── Detector dispatch ──────────────────────────────────────────────────────

    def _run_all_detectors(
        self,
        image: np.ndarray,
        ocr_results: List[OCRTextResult],
        page_number: int,
        image_path: str,
        errors: List[str],
    ) -> Tuple[List[StructuralElement], List[str]]:
        """Call each enabled detector and collect elements."""
        all_elements: List[StructuralElement] = []

        # TextDetector has its own internal OCR; different method signature
        self._call(
            "text", self._text_detector,
            lambda d: d.detect_text_elements(image, page_number, image_path),
            all_elements, errors,
        )

        # ListDetector accepts optional ocr_results; pass them to avoid a second OCR pass
        self._call(
            "list", self._list_detector,
            lambda d: d.detect_lists(image, page_number, ocr_results, image_path),
            all_elements, errors,
        )

        # TableDetector (line-based) runs its own internal OCR for cell text
        self._call(
            "table_line", self._table_detector,
            lambda d: d.detect_tables(image, page_number, image_path),
            all_elements, errors,
        )

        # ContentTableDetector requires ocr_results
        self._call(
            "table_content", self._content_table_detector,
            lambda d: d.detect_tables(image, page_number, ocr_results, image_path),
            all_elements, errors,
        )

        # Sprint 4+ detectors: standard detect(image, ocr_results, page_number)
        for name, detector in [
            ("header_footer",  self._header_footer_detector),
            ("formula",        self._formula_detector),
            ("figure",         self._figure_detector),
            ("annotation",     self._annotation_detector),
            ("watermark",      self._watermark_detector),
            ("barcode",        self._barcode_detector),
            ("code_block",     self._code_block_detector),
            ("reference",      self._reference_detector),
            ("toc",            self._toc_detector),
            ("index",          self._index_detector),
            ("layout",         self._layout_detector),   # Sprint 9
        ]:
            self._call(
                name, detector,
                lambda d: d.detect(image, ocr_results, page_number),
                all_elements, errors,
            )

        return all_elements, errors

    @staticmethod
    def _call(
        name: str,
        detector: Any,
        fn,
        all_elements: List[StructuralElement],
        errors: List[str],
    ) -> None:
        """Call a detector function, appending results and logging on failure."""
        if detector is None:
            return
        try:
            result = fn(detector)
            # All detectors return (elements, trace) tuples
            elements, _ = result
            all_elements.extend(elements)
            logger.debug("Detector '%s' found %d elements", name, len(elements))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.error("Detector '%s' failed: %s", name, exc)

    # ── GPU helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _check_gpu_availability() -> bool:
        """Return True if an OpenCV CUDA-capable GPU is present."""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            return False

    @staticmethod
    def _preprocess_image_gpu(image: np.ndarray) -> np.ndarray:
        """
        Apply GPU-accelerated grayscale conversion and denoising.
        Falls back to CPU automatically if CUDA operations fail.
        """
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(image)
            # Grayscale conversion on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
            # Upload back to BGR (detectors expect 3-channel or handle gray)
            gpu_bgr = cv2.cuda.cvtColor(gpu_gray, cv2.COLOR_GRAY2BGR)
            return gpu_bgr.download()
        except Exception as exc:
            logger.debug("GPU preprocessing unavailable, using CPU fallback: %s", exc)
            return image

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_failed_result(image_path: str, error: str, duration: float) -> DocumentResult:
        doc_id = Path(image_path).stem + "_" + uuid.uuid4().hex[:6]
        metadata = DocumentMetadata(
            source_file=image_path,
            document_id=doc_id,
            processing_timestamp=datetime.now(),
            processing_duration=duration,
            image_dimensions=(1, 1),
            detected_language="",
            total_elements_extracted=0,
            average_confidence=0.0,
            processing_status=ProcessingStatus.FAILED,
            errors_encountered=[error],
        )
        return DocumentResult(metadata=metadata)
