"""
Text Detection Module - Heading and Paragraph Extraction

This module implements text-based element detection:
1. HEADING - Titles and section headers with level hierarchy (H1-H6)
2. TEXT - General body text paragraphs
3. BLOCK_QUOTE - Indented or quoted text blocks

Design:
- Uses PSM mode 11 for heading detection (sparse text, better segmentation)
- Uses PSM mode 6 for paragraph detection (uniform blocks, dense text)
- Analyzes text properties (size, position, indentation) to classify elements
- Builds hierarchical relationships between headings and paragraphs
- Returns StructuralElement objects with appropriate types

Follows parameterized strategy pattern (ARCHITECTURAL_DECISIONS.md #3)
Implements processing traces for reproducibility (#5)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging

from data_models import (
    OCRTextResult,
    StructuralElement,
    ElementType,
    BoundingBox,
    PSMMode,
)
from ocr_engine import OCREngine, OCREngineConfig


logger = logging.getLogger(__name__)


class HeadingLevel(Enum):
    """Heading hierarchy levels (H1-H6)."""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


@dataclass
class TextDetectorConfig:
    """
    Configuration for text detection (headings, paragraphs, block quotes).
    
    Attributes:
        # Detection flags
        detect_headings: Enable heading detection
        detect_paragraphs: Enable paragraph detection
        detect_block_quotes: Enable block quote detection
        
        # Heading detection
        heading_levels: Which heading levels to detect (1-6)
        heading_min_confidence: Minimum confidence for headings
        min_heading_size_ratio: Ratio of heading size vs average text
        
        # Paragraph detection
        min_paragraph_words: Minimum words to constitute a paragraph
        min_paragraph_confidence: Minimum confidence for paragraphs
        
        # Block quote detection
        block_quote_min_indentation: Minimum indentation (pixels)
        block_quote_min_confidence: Minimum confidence for block quotes
        
        # General
        min_confidence: Global minimum confidence threshold
        language: Language for OCR
        enable_preprocessing: Apply image preprocessing
    """
    
    # Detection flags
    detect_headings: bool = True
    detect_paragraphs: bool = True
    detect_block_quotes: bool = True
    
    # Heading detection
    heading_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    heading_min_confidence: float = 0.5
    min_heading_size_ratio: float = 1.2  # Heading text must be 1.2x average size
    
    # Paragraph detection
    min_paragraph_words: int = 1
    min_paragraph_confidence: float = 0.3
    
    # Block quote detection
    block_quote_min_indentation: float = 20.0  # pixels
    block_quote_min_confidence: float = 0.3
    
    # General
    min_confidence: float = 0.3
    language: str = "eng"
    enable_preprocessing: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 <= self.heading_min_confidence <= 1.0):
            raise ValueError("heading_min_confidence must be 0-1")
        if not (0.0 <= self.min_paragraph_confidence <= 1.0):
            raise ValueError("min_paragraph_confidence must be 0-1")
        if not (0.0 <= self.block_quote_min_confidence <= 1.0):
            raise ValueError("block_quote_min_confidence must be 0-1")
        if self.min_heading_size_ratio < 1.0:
            raise ValueError("min_heading_size_ratio must be >= 1.0")
        if self.block_quote_min_indentation < 0:
            raise ValueError("block_quote_min_indentation must be >= 0")


@dataclass
class TextDetectionTrace:
    """
    Processing trace for text detection.
    
    Attributes:
        config: TextDetectorConfig used
        processing_start: When detection started
        processing_end: When detection completed
        image_dimensions: Image size in pixels
        
        # Detection results
        headings_found: Number of headings detected
        paragraphs_found: Number of paragraphs detected
        block_quotes_found: Number of block quotes detected
        
        # Analysis
        total_results: Total structural elements created
        ocr_results_analyzed: How many OCRTextResult objects were analyzed
        average_text_size: Average font size detected
        heading_sizes: Font sizes used for headings
        
        # Timing
        heading_detection_time_ms: Time spent on heading detection
        paragraph_detection_time_ms: Time spent on paragraph detection
        block_quote_detection_time_ms: Time spent on block quote detection
    """
    config: TextDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]
    
    headings_found: int = 0
    paragraphs_found: int = 0
    block_quotes_found: int = 0
    total_results: int = 0
    ocr_results_analyzed: int = 0
    average_text_size: float = 0.0
    heading_sizes: List[Tuple[int, float]] = field(default_factory=list)  # (level, size)
    
    heading_detection_time_ms: float = 0.0
    paragraph_detection_time_ms: float = 0.0
    block_quote_detection_time_ms: float = 0.0
    
    @property
    def total_processing_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        delta = self.processing_end - self.processing_start
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "config": {
                "detect_headings": self.config.detect_headings,
                "detect_paragraphs": self.config.detect_paragraphs,
                "detect_block_quotes": self.config.detect_block_quotes,
                "heading_levels": self.config.heading_levels,
                "min_confidence": self.config.min_confidence,
            },
            "results": {
                "headings": self.headings_found,
                "paragraphs": self.paragraphs_found,
                "block_quotes": self.block_quotes_found,
                "total": self.total_results,
            },
            "analysis": {
                "ocr_results_analyzed": self.ocr_results_analyzed,
                "average_text_size": self.average_text_size,
                "heading_sizes": self.heading_sizes,
            },
            "timing_ms": {
                "total": self.total_processing_time_ms,
                "headings": self.heading_detection_time_ms,
                "paragraphs": self.paragraph_detection_time_ms,
                "block_quotes": self.block_quote_detection_time_ms,
            },
        }


class TextDetector:
    """
    Detector for text-based structural elements (headings, paragraphs, block quotes).
    
    Strategy:
    - Runs OCR with different PSM modes optimized for each element type
    - Analyzes text properties (size, position, indentation) to classify
    - Groups text into hierarchical structures (headings → paragraphs)
    - Returns StructuralElement objects ready for document assembly
    
    Usage:
        config = TextDetectorConfig(detect_headings=True, detect_paragraphs=True)
        detector = TextDetector(config)
        
        elements, trace = detector.detect_text_elements(
            image=image,
            page_number=1,
            image_path="document.png"
        )
        
        for element in elements:
            print(f"{element.element_type.value}: {element.content}")
    """
    
    def __init__(self, config: Optional[TextDetectorConfig] = None):
        """
        Initialize text detector with configuration.
        
        Args:
            config: TextDetectorConfig. Uses defaults if None.
        """
        self.config = config or TextDetectorConfig()
        
        # Initialize OCR engines for different detection modes
        self.heading_ocr_config = OCREngineConfig(
            psm_modes=[PSMMode.SPARSE_TEXT, PSMMode.FULLY_AUTOMATIC],
            languages=self.config.language,
            enable_preprocessing=self.config.enable_preprocessing,
            min_confidence=self.config.heading_min_confidence,
        )
        self.heading_engine = OCREngine(self.heading_ocr_config)
        
        self.paragraph_ocr_config = OCREngineConfig(
            psm_modes=[PSMMode.SINGLE_COLUMN, PSMMode.FULLY_AUTOMATIC],
            languages=self.config.language,
            enable_preprocessing=self.config.enable_preprocessing,
            min_confidence=self.config.min_paragraph_confidence,
        )
        self.paragraph_engine = OCREngine(self.paragraph_ocr_config)
    
    def detect_text_elements(
        self,
        image: np.ndarray,
        page_number: int = 1,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], TextDetectionTrace]:
        """
        Detect all text-based elements in image.
        
        Args:
            image: Image as numpy array (BGR from cv2)
            page_number: Page number for document (default 1)
            image_path: Optional path to image for logging
        
        Returns:
            Tuple of (StructuralElement list, TextDetectionTrace)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        start_time = datetime.now()
        image_dims = tuple(image.shape[1::-1])  # (width, height)
        
        logger.info(
            "Starting text detection on %s (size: %s, page: %d)",
            image_path or "unnamed image", image_dims, page_number
        )
        
        all_elements = []
        trace = TextDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=start_time,  # Will be updated
            image_dimensions=image_dims,
        )
        
        # Detect headings
        if self.config.detect_headings:
            heading_start = datetime.now()
            headings, ocr_results = self._detect_headings(image, page_number)
            all_elements.extend(headings)
            trace.headings_found = len(headings)
            trace.ocr_results_analyzed += len(ocr_results)
            trace.heading_detection_time_ms = (
                (datetime.now() - heading_start).total_seconds() * 1000
            )
            logger.info("Detected %d headings", len(headings))
        
        # Detect paragraphs
        if self.config.detect_paragraphs:
            para_start = datetime.now()
            paragraphs, ocr_results = self._detect_paragraphs(image, page_number)
            all_elements.extend(paragraphs)
            trace.paragraphs_found = len(paragraphs)
            trace.ocr_results_analyzed += len(ocr_results)
            trace.paragraph_detection_time_ms = (
                (datetime.now() - para_start).total_seconds() * 1000
            )
            logger.info(f"Detected {len(paragraphs)} paragraphs")
        
        # Detect block quotes
        if self.config.detect_block_quotes:
            quote_start = datetime.now()
            quotes = self._detect_block_quotes(all_elements, image, page_number)
            all_elements.extend(quotes)
            trace.block_quotes_found = len(quotes)
            trace.block_quote_detection_time_ms = (
                (datetime.now() - quote_start).total_seconds() * 1000
            )
            logger.info(f"Detected {len(quotes)} block quotes")
        
        # Build hierarchy and compute statistics
        self._build_hierarchy(all_elements)
        self._compute_statistics(all_elements, trace)
        
        trace.total_results = len(all_elements)
        trace.processing_end = datetime.now()
        
        logger.info(
            f"Text detection complete: {trace.total_results} elements found "
            f"in {trace.total_processing_time_ms:.1f}ms"
        )
        
        return all_elements, trace
    
    def _detect_headings(
        self,
        image: np.ndarray,
        page_number: int,
    ) -> Tuple[List[StructuralElement], List[OCRTextResult]]:
        """
        Detect headings using sparse text segmentation (PSM mode 11).
        
        Strategy:
        1. Extract text with PSM 11 (sparse text, better segmentation)
        2. Analyze text sizes to identify relative size changes
        3. Text above paragraphs → likely heading
        4. Classify heading level (H1-H6) by relative size
        5. Filter by confidence threshold
        
        Args:
            image: Input image
            page_number: Page number
        
        Returns:
            Tuple of (heading StructuralElement list, OCRTextResult list used)
        """
        logger.debug("Starting heading detection")
        
        # Run OCR with sparse text mode
        ocr_results, _ = self.heading_engine.extract_text(image, page_number)
        
        if not ocr_results:
            logger.debug("No text found for heading detection")
            return [], []
        
        # Analyze text sizes
        text_heights = [r.bbox.y_max - r.bbox.y_min for r in ocr_results]
        avg_height = np.mean(text_heights) if text_heights else 10
        
        headings = []
        element_id_counter = 0
        
        for result in ocr_results:
            text_height = result.bbox.y_max - result.bbox.y_min
            size_ratio = text_height / avg_height if avg_height > 0 else 1.0
            
            # Heading candidates: significantly larger than average text
            if size_ratio >= self.config.min_heading_size_ratio:
                if result.confidence < self.config.heading_min_confidence:
                    logger.debug(
                        f"Skipping low-confidence heading: '{result.text}' "
                        f"({result.confidence:.2%})"
                    )
                    continue
                
                # Determine heading level (1-6) based on size ratio
                # Larger ratio = higher priority = lower level (H1)
                level = self._classify_heading_level(size_ratio)
                
                # Check if level is in allowed levels
                if level not in self.config.heading_levels:
                    continue
                
                element = StructuralElement(
                    element_id=f"heading_{page_number}_{element_id_counter}",
                    element_type=ElementType.HEADING,
                    content=result.text,
                    bbox=result.bbox,
                    confidence=result.confidence,
                    page_number=page_number,
                    nesting_level=level - 1,  # H1 = level 0, H2 = level 1, etc.
                    metadata={
                        "heading_level": level,
                        "size_ratio": size_ratio,
                        "text_height": text_height,
                    },
                    processing_method="heading_detector_psm11",
                )
                headings.append(element)
                element_id_counter += 1
                
                logger.debug(
                    f"Detected heading (H{level}): '{result.text}' "
                    f"(size_ratio: {size_ratio:.2f}, conf: {result.confidence:.2%})"
                )
        
        return headings, ocr_results
    
    def _detect_paragraphs(
        self,
        image: np.ndarray,
        page_number: int,
    ) -> Tuple[List[StructuralElement], List[OCRTextResult]]:
        """
        Detect paragraphs using uniform block segmentation (PSM mode 6).
        
        Strategy:
        1. Extract text with PSM 6 (uniform blocks, dense text)
        2. Group words by vertical position (same baseline = same line)
        3. Group lines by horizontal alignment (same indent level = same paragraph)
        4. Calculate confidence as average of constituent words
        5. Return as TEXT elements
        
        Args:
            image: Input image
            page_number: Page number
        
        Returns:
            Tuple of (paragraph StructuralElement list, OCRTextResult list used)
        """
        logger.debug("Starting paragraph detection")
        
        # Run OCR with uniform block mode
        ocr_results, _ = self.paragraph_engine.extract_text(image, page_number)
        
        if not ocr_results:
            logger.debug("No text found for paragraph detection")
            return [], []
        
        # Group results into paragraphs
        # Simple approach: group by vertical proximity and horizontal alignment
        paragraphs = self._group_into_paragraphs(ocr_results)
        
        elements = []
        for para_idx, para_results in enumerate(paragraphs):
            if len(para_results) < self.config.min_paragraph_words:
                continue
            
            # Calculate paragraph properties
            para_text = " ".join(r.text for r in para_results)
            para_confidences = [r.confidence for r in para_results]
            para_confidence = np.mean(para_confidences) if para_confidences else 0.0
            
            if para_confidence < self.config.min_paragraph_confidence:
                logger.debug(
                    f"Skipping low-confidence paragraph: "
                    f"'{para_text[:50]}...' ({para_confidence:.2%})"
                )
                continue
            
            # Compute bounding box (union of all words in paragraph)
            x_mins = [r.bbox.x_min for r in para_results]
            y_mins = [r.bbox.y_min for r in para_results]
            x_maxs = [r.bbox.x_max for r in para_results]
            y_maxs = [r.bbox.y_max for r in para_results]
            
            para_bbox = BoundingBox(
                x_min=min(x_mins),
                y_min=min(y_mins),
                x_max=max(x_maxs),
                y_max=max(y_maxs),
                confidence=para_confidence,
            )
            
            element = StructuralElement(
                element_id=f"paragraph_{page_number}_{para_idx}",
                element_type=ElementType.TEXT,
                content=para_text,
                bbox=para_bbox,
                confidence=para_confidence,
                page_number=page_number,
                metadata={
                    "word_count": len(para_results),
                    "indentation_level": 0,
                },
                processing_method="paragraph_detector_psm6",
            )
            elements.append(element)
            
            logger.debug(
                f"Detected paragraph: '{para_text[:40]}...' "
                f"({len(para_results)} words, conf: {para_confidence:.2%})"
            )
        
        return elements, ocr_results
    
    def _detect_block_quotes(
        self,
        existing_elements: List[StructuralElement],
        image: np.ndarray,
        page_number: int,
    ) -> List[StructuralElement]:
        """
        Detect block quotes (indented or highlighted text).
        
        Strategy:
        1. Analyze indentation of detected text elements
        2. Elements with significant left margin → likely block quotes
        3. Optionally detect visual styling (borders, background)
        4. Mark as BLOCK_QUOTE element type
        
        Args:
            existing_elements: Already detected TEXT/HEADING elements
            image: Input image
            page_number: Page number
        
        Returns:
            List of block quote StructuralElement objects
        """
        logger.debug("Starting block quote detection")
        
        if not existing_elements:
            logger.debug("No elements to analyze for block quotes")
            return []
        
        # Get image width to calculate relative indentation
        image_width = image.shape[1]
        
        block_quotes = []
        element_id_counter = 0
        
        for element in existing_elements:
            # Skip already-classified block quotes and headings
            if element.element_type in [ElementType.BLOCK_QUOTE, ElementType.HEADING]:
                continue
            
            # Calculate indentation (left margin in pixels)
            indentation = element.bbox.x_min
            indentation_ratio = indentation / image_width if image_width > 0 else 0
            
            # Check if indented enough to be considered a block quote
            if indentation >= self.config.block_quote_min_indentation:
                logger.debug(
                    f"Detected indented text as block quote: "
                    f"'{str(element.content)[:40]}...' (indent: {indentation:.0f}px)"
                )
                
                # Create block quote element (copy of original with different type)
                quote = StructuralElement(
                    element_id=f"blockquote_{page_number}_{element_id_counter}",
                    element_type=ElementType.BLOCK_QUOTE,
                    content=element.content,
                    bbox=element.bbox,
                    confidence=element.confidence,
                    page_number=page_number,
                    nesting_level=1,
                    metadata={
                        "indentation_pixels": indentation,
                        "indentation_ratio": indentation_ratio,
                        "original_element_id": element.element_id,
                    },
                    processing_method="block_quote_detector_spatial",
                )
                block_quotes.append(quote)
                element_id_counter += 1
        
        return block_quotes
    
    def _classify_heading_level(self, size_ratio: float) -> int:
        """
        Classify heading level (1-6) based on size ratio.
        
        Mapping:
        - Ratio >= 3.0: H1 (largest)
        - Ratio >= 2.5: H2
        - Ratio >= 2.0: H3
        - Ratio >= 1.5: H4
        - Ratio >= 1.2: H5
        - Ratio >= 1.0: H6 (smallest)
        
        Args:
            size_ratio: Text size relative to average
        
        Returns:
            Heading level (1-6)
        """
        if size_ratio >= 3.0:
            return 1
        elif size_ratio >= 2.5:
            return 2
        elif size_ratio >= 2.0:
            return 3
        elif size_ratio >= 1.5:
            return 4
        elif size_ratio >= 1.2:
            return 5
        else:
            return 6
    
    def _group_into_paragraphs(
        self,
        ocr_results: List[OCRTextResult],
    ) -> List[List[OCRTextResult]]:
        """
        Group OCR results into paragraphs based on spatial proximity.
        
        Strategy:
        1. Sort by vertical position (top to bottom)
        2. Group by vertical proximity (same line)
        3. Sort within each line by horizontal position
        4. Collect consecutive lines as paragraphs
        
        Args:
            ocr_results: List of OCRTextResult from OCR
        
        Returns:
            List of paragraph groups (each group is list of OCRTextResult)
        """
        if not ocr_results:
            return []
        
        # Sort by position (top-left corner)
        sorted_results = sorted(ocr_results, key=lambda r: (r.bbox.y_min, r.bbox.x_min))
        
        # Group by lines (same vertical position)
        lines = []
        current_line = [sorted_results[0]]
        line_y_threshold = 5  # pixels
        
        for result in sorted_results[1:]:
            # Check if same line (similar y position)
            if abs(result.bbox.y_min - current_line[0].bbox.y_min) < line_y_threshold:
                current_line.append(result)
            else:
                # New line
                lines.append(current_line)
                current_line = [result]
        
        if current_line:
            lines.append(current_line)
        
        # Group lines into paragraphs
        # Simple approach: consecutive lines with similar left margin = same paragraph
        paragraphs = []
        current_paragraph = []
        prev_left_margin = None
        margin_threshold = 10  # pixels
        
        for line in lines:
            # Get left margin of line
            left_margin = min(r.bbox.x_min for r in line)
            
            # Check if same paragraph (similar left margin)
            if (prev_left_margin is None or
                abs(left_margin - prev_left_margin) < margin_threshold):
                current_paragraph.extend(line)
                prev_left_margin = left_margin
            else:
                # New paragraph
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = line
                prev_left_margin = left_margin
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def _build_hierarchy(self, elements: List[StructuralElement]) -> None:
        """
        Build parent-child relationships between elements.
        
        Strategy:
        - Headings become parents of following paragraphs/block quotes
        - H1 heading > H2 subheading > paragraphs
        - Stop parent relationship when encountering same or higher-level heading
        
        Args:
            elements: List of StructuralElement to organize (modified in-place)
        """
        if not elements:
            return
        
        # Find parents for each element
        for i, element in enumerate(elements):
            if element.element_type == ElementType.HEADING:
                # Find next heading of equal or higher level
                element_level = element.metadata.get("heading_level", 1)
                
                # Paragraphs following this heading are children until next heading
                for j in range(i + 1, len(elements)):
                    next_elem = elements[j]
                    
                    if next_elem.element_type == ElementType.HEADING:
                        next_level = next_elem.metadata.get("heading_level", 1)
                        # Stop if next heading is same level or higher (lower number)
                        if next_level <= element_level:
                            break
                    
                    # Make element a child of this heading
                    if next_elem.element_type in [ElementType.TEXT, ElementType.BLOCK_QUOTE]:
                        next_elem.parent_id = element.element_id
                        element.child_ids.append(next_elem.element_id)
    
    def _compute_statistics(
        self,
        elements: List[StructuralElement],
        trace: TextDetectionTrace,
    ) -> None:
        """
        Compute and record statistics about detected elements.
        
        Args:
            elements: Detected elements
            trace: TextDetectionTrace to populate (modified in-place)
        """
        # Calculate average text size from headings
        heading_sizes = []
        for elem in elements:
            if elem.element_type == ElementType.HEADING:
                height = elem.bbox.y_max - elem.bbox.y_min
                level = elem.metadata.get("heading_level", 1)
                heading_sizes.append((level, height))
        
        if heading_sizes:
            trace.heading_sizes = heading_sizes
            avg_height = np.mean([h for _, h in heading_sizes])
            trace.average_text_size = avg_height
        else:
            # Use paragraph sizes if no headings
            for elem in elements:
                if elem.element_type == ElementType.TEXT:
                    height = elem.bbox.y_max - elem.bbox.y_min
                    trace.average_text_size = height
                    break


# ============================================================================
# Utility Functions
# ============================================================================


def create_text_detection_pipeline(
    config: Optional[TextDetectorConfig] = None,
) -> TextDetector:
    """
    Create a fully configured text detection pipeline.
    
    Args:
        config: Optional TextDetectorConfig. Uses defaults if None.
    
    Returns:
        Initialized TextDetector ready for use
    """
    return TextDetector(config)
