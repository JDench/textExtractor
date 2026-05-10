"""
OCR Engine Wrapper - Basic Tesseract Integration

This module provides a parameterized wrapper around Tesseract OCR that:
1. Supports multiple PSM (Page Segmentation) and OEM (Engine) modes
2. Returns normalized OCRTextResult objects
3. Handles preprocessing and postprocessing
4. Records processing metadata for reproducibility

Design Alignment:
- Follows parameterized strategy pattern (ARCHITECTURAL_DECISIONS.md #3)
- Supports multiple detection modes (PSM modes) for different layouts
- Records configuration and timing for reproducibility
- Normalizes confidence scores to 0-1 range for consistency
"""

import cv2
import pytesseract
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging

from data_models import (
    OCRTextResult,
    BoundingBox,
    PSMMode,
    OEMMode,
    Coordinates,
)


logger = logging.getLogger(__name__)


@dataclass
class OCREngineConfig:
    """
    Configuration for OCR engine behavior.
    
    Follows the parameterized strategy pattern:
    - Select which PSM/OEM modes to use
    - Control preprocessing options
    - Set confidence thresholds
    - Enable additional metrics/metadata
    
    Attributes:
        psm_modes: List of PSMMode enums to try (in order)
        oem_mode: Which OCR engine mode to use
        languages: Tesseract language codes (e.g., "eng", "eng+fra")
        min_confidence: Filter results below this threshold (0-1)
        
        # Preprocessing options
        enable_preprocessing: Apply preprocessing steps
        target_dpi: Resize image to target DPI if provided
        use_binary: Convert to binary before OCR
        
        # Output options
        include_font_properties: Extract font name/size if available
        detect_language: Attempt to detect language
        
        # Advanced
        tesseract_path: Explicit path to tesseract executable (if not in PATH)
        extra_config: Additional Tesseract config string
    """
    
    psm_modes: List[PSMMode] = field(
        default_factory=lambda: [PSMMode.FULLY_AUTOMATIC]
    )
    oem_mode: OEMMode = OEMMode.DEFAULT
    languages: str = "eng"  # Tesseract language codes
    min_confidence: float = 0.3  # 30% minimum
    
    # Preprocessing
    enable_preprocessing: bool = True
    target_dpi: Optional[int] = None
    use_binary: bool = False
    
    # Output options
    include_font_properties: bool = False
    detect_language: bool = True
    
    # Advanced
    tesseract_path: Optional[str] = None
    extra_config: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.psm_modes:
            raise ValueError("At least one PSM mode must be specified")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1")
        if self.target_dpi and self.target_dpi < 50:
            raise ValueError("target_dpi must be >= 50")


@dataclass
class OCRProcessingTrace:
    """
    Records metadata about OCR processing for reproducibility and debugging.
    
    Allows re-running OCR with exact same parameters or understanding why
    results vary between runs.
    
    Attributes:
        config: The OCREngineConfig used for this OCR run
        psm_mode_used: Which PSM mode actually produced results
        processing_start: When OCR started
        processing_end: When OCR completed
        processing_duration_ms: How long it took
        image_dimensions_before: Image size before preprocessing
        image_dimensions_after: Image size after preprocessing (if changed)
        preprocessing_applied: Which preprocessing steps were applied
        total_results: How many text regions were detected
        average_confidence: Mean confidence across results
        languages_detected: Language codes detected
    """
    config: OCREngineConfig
    psm_mode_used: PSMMode
    processing_start: datetime
    processing_end: datetime
    image_dimensions_before: Tuple[int, int]
    image_dimensions_after: Tuple[int, int]
    preprocessing_applied: List[str] = field(default_factory=list)
    total_results: int = 0
    average_confidence: float = 0.0
    languages_detected: List[str] = field(default_factory=list)
    
    @property
    def processing_duration_ms(self) -> float:
        """Calculate processing duration in milliseconds."""
        delta = self.processing_end - self.processing_start
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for logging/serialization."""
        return {
            "config": {
                "psm_modes": [m.name for m in self.config.psm_modes],
                "oem_mode": self.config.oem_mode.name,
                "languages": self.config.languages,
                "min_confidence": self.config.min_confidence,
            },
            "psm_mode_used": self.psm_mode_used.name,
            "processing_duration_ms": self.processing_duration_ms,
            "image_dimensions_before": self.image_dimensions_before,
            "image_dimensions_after": self.image_dimensions_after,
            "preprocessing_applied": self.preprocessing_applied,
            "total_results": self.total_results,
            "average_confidence": self.average_confidence,
            "languages_detected": self.languages_detected,
        }


class OCREngine:
    """
    Tesseract OCR wrapper supporting multiple PSM modes and configurations.
    
    This is the basic OCR engine wrapper implementing Sprint 1 requirements.
    
    Design:
    - Parameterized via OCREngineConfig (strategy pattern)
    - Tries multiple PSM modes to find best results
    - Returns OCRTextResult objects with normalized confidence (0-1)
    - Records processing trace for reproducibility
    - Handles preprocessing (binary conversion, resize, etc.)
    
    Typical usage:
        config = OCREngineConfig(
            psm_modes=[PSMMode.FULLY_AUTOMATIC, PSMMode.SPARSE_TEXT],
            languages="eng"
        )
        engine = OCREngine(config)
        
        image = cv2.imread("document.png")
        results = engine.extract_text(image, page_number=1)
        
        for result in results:
            print(f"{result.text}: {result.confidence:.2%}")
    """
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        """
        Initialize OCR engine with configuration.
        
        Args:
            config: OCREngineConfig with engine parameters. Uses defaults if None.
        
        Raises:
            ValueError: If Tesseract is not installed or config is invalid
        """
        self.config = config or OCREngineConfig()
        
        # Set up Tesseract path if specified
        if self.config.tesseract_path:
            pytesseract.pytesseract.pytesseract_cmd = self.config.tesseract_path
        
        # Verify Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not found or not accessible: {e}")
            raise ValueError(
                "Tesseract OCR must be installed. Install via: "
                "https://github.com/UB-Mannheim/tesseract/wiki"
            ) from e
    
    def extract_text(
        self,
        image: np.ndarray,
        page_number: int = 1,
        image_path: Optional[str] = None,
    ) -> Tuple[List[OCRTextResult], OCRProcessingTrace]:
        """
        Extract text from image using Tesseract.
        
        Tries PSM modes in order until successful or all fail.
        
        Args:
            image: Image as numpy array (BGR format from cv2)
            page_number: Page number for document (default 1)
            image_path: Optional path to image file for logging
        
        Returns:
            Tuple of (OCRTextResult list, OCRProcessingTrace)
            - Results in order of text detection
            - Trace with metadata about processing
        
        Raises:
            ValueError: If image is invalid
            RuntimeError: If OCR fails on all PSM modes
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        start_time = datetime.now()
        image_dims_before = tuple(image.shape[1::-1])  # (width, height)
        
        logger.info(
            f"Starting OCR on image {image_path or '(unnamed)'} "
            f"(size: {image_dims_before}, page: {page_number})"
        )
        
        # Try each PSM mode in order
        best_results = None
        best_psm_mode = None
        preprocessing_steps = []
        
        # Preprocess if enabled
        if self.config.enable_preprocessing:
            image = self._preprocess_image(image, preprocessing_steps)
        
        image_dims_after = tuple(image.shape[1::-1])
        
        # Try each PSM mode
        for psm_mode in self.config.psm_modes:
            try:
                logger.debug(f"Trying PSM mode: {psm_mode.name}")
                results = self._extract_with_psm(image, psm_mode, page_number)
                
                if results:
                    best_results = results
                    best_psm_mode = psm_mode
                    logger.info(
                        f"Successfully extracted {len(results)} text regions "
                        f"with PSM {psm_mode.name}"
                    )
                    break
            except Exception as e:
                logger.warning(f"PSM mode {psm_mode.name} failed: {e}")
                continue
        
        if best_results is None:
            raise RuntimeError(
                f"OCR extraction failed on all {len(self.config.psm_modes)} "
                f"PSM modes for image {image_path or '(unnamed)'}"
            )
        
        end_time = datetime.now()
        
        # Compute statistics
        confidences = [r.confidence for r in best_results if r.confidence > 0]
        avg_confidence = (
            np.mean(confidences) if confidences else 0.0
        )
        
        # Create processing trace
        trace = OCRProcessingTrace(
            config=self.config,
            psm_mode_used=best_psm_mode,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions_before=image_dims_before,
            image_dimensions_after=image_dims_after,
            preprocessing_applied=preprocessing_steps,
            total_results=len(best_results),
            average_confidence=avg_confidence,
        )
        
        logger.info(
            f"OCR completed in {trace.processing_duration_ms:.1f}ms: "
            f"{len(best_results)} results, avg confidence {avg_confidence:.2%}"
        )
        
        return best_results, trace
    
    def _extract_with_psm(
        self,
        image: np.ndarray,
        psm_mode: PSMMode,
        page_number: int,
    ) -> List[OCRTextResult]:
        """
        Extract text using specific PSM mode with word-level data.
        
        Uses Tesseract's output_type=Output.DICT to get detailed
        word-level information including confidence and bounding boxes.
        
        Args:
            image: Preprocessed image
            psm_mode: PSM mode to use
            page_number: Page number for results
        
        Returns:
            List of OCRTextResult objects
        """
        # Build Tesseract config
        config = self._build_tesseract_config(psm_mode)
        
        # Run Tesseract with detailed output
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.error(f"Tesseract failed with config: {config}\nError: {e}")
            raise
        
        # Convert Tesseract output to OCRTextResult objects
        results = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            
            # Skip empty results
            if not text:
                continue
            
            # Get confidence (0-100 from Tesseract)
            confidence_raw = int(data["conf"][i])
            confidence_normalized = confidence_raw / 100.0  # Normalize to 0-1
            
            # Filter by minimum confidence
            if confidence_normalized < self.config.min_confidence:
                logger.debug(f"Filtering low-confidence result: '{text}' ({confidence_raw}%)")
                continue
            
            # Extract bounding box
            bbox = BoundingBox(
                x_min=float(data["left"][i]),
                y_min=float(data["top"][i]),
                x_max=float(data["left"][i] + data["width"][i]),
                y_max=float(data["top"][i] + data["height"][i]),
                confidence=confidence_normalized,
            )
            
            # Create OCRTextResult
            result = OCRTextResult(
                text=text,
                confidence=confidence_normalized,
                bbox=bbox,
                language=self.config.languages.split("+")[0],  # First language in list
                page_number=page_number,
                is_numeric=self._is_numeric(text),
            )
            
            results.append(result)
            logger.debug(
                f"Extracted: '{text}' @ ({bbox.x_min:.0f},{bbox.y_min:.0f}) "
                f"conf={confidence_normalized:.2%}"
            )
        
        return results
    
    def _preprocess_image(
        self,
        image: np.ndarray,
        steps_applied: List[str],
    ) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Applied steps (if enabled):
        1. Convert to grayscale
        2. Upscale if below target DPI (if target_dpi set)
        3. Apply binary threshold (if use_binary set)
        4. Denoise (optional)
        
        Args:
            image: Input image (BGR from cv2)
            steps_applied: List to record which steps were applied
        
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            steps_applied.append("grayscale_conversion")
        
        # Upscale if target DPI specified
        # Note: This is a simplified approach. Real DPI handling would require
        # image metadata, which we may not have from numpy arrays
        if self.config.target_dpi and self.config.target_dpi > 0:
            # Estimate current DPI based on image size (heuristic)
            current_estimated_dpi = 96  # Assume 96 DPI default
            if current_estimated_dpi < self.config.target_dpi:
                scale_factor = self.config.target_dpi / current_estimated_dpi
                new_size = (
                    int(processed.shape[1] * scale_factor),
                    int(processed.shape[0] * scale_factor),
                )
                processed = cv2.resize(processed, new_size, interpolation=cv2.INTER_CUBIC)
                steps_applied.append(f"upscale_to_{self.config.target_dpi}dpi")
        
        # Binary threshold
        if self.config.use_binary:
            _, processed = cv2.threshold(
                processed, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            steps_applied.append("binary_threshold")
        
        return processed
    
    def _build_tesseract_config(self, psm_mode: PSMMode) -> str:
        """
        Build Tesseract configuration string.
        
        Args:
            psm_mode: PSM mode to use
        
        Returns:
            Config string for pytesseract
        """
        config_parts = [
            f"--psm {psm_mode.value}",
            f"--oem {self.config.oem_mode.value}",
            f"-l {self.config.languages}",
        ]
        
        # Add any extra config
        if self.config.extra_config:
            config_parts.append(self.config.extra_config)
        
        return " ".join(config_parts)
    
    @staticmethod
    def _is_numeric(text: str) -> bool:
        """Detect if text is primarily numeric."""
        # Simple heuristic: mostly digits and common numeric chars
        numeric_chars = sum(1 for c in text if c.isdigit() or c in ".,-%")
        return numeric_chars > len(text) * 0.7 if text else False
    
    def extract_text_simple(
        self,
        image: np.ndarray,
        page_number: int = 1,
    ) -> str:
        """
        Extract all text from image as simple string (convenience method).
        
        Args:
            image: Image as numpy array
            page_number: Page number (for metadata)
        
        Returns:
            All extracted text as single string
        """
        results, _ = self.extract_text(image, page_number=page_number)
        return " ".join(r.text for r in results)


# ============================================================================
# Utility Functions
# ============================================================================


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (BGR format)
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image can't be loaded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load image from file in RGB format.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image as numpy array (RGB format)
    """
    image = load_image(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
