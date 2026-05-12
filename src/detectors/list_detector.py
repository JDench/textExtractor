"""
List Detection Module - Marker-Based List Extraction

This module implements list-based element detection:
1. LIST - Bullet points, numbered lists, lettered lists with nesting
2. LIST_ITEM - Individual items with hierarchy preservation

Design:
- Detects markers: bullets (•, -, *), numbers (1., 2., etc.), letters (a., b., etc.)
- Analyzes indentation to build hierarchy (nesting levels)
- Groups consecutive items into ListStructure objects
- Preserves parent-child relationships via parent_item_id/child_item_ids
- Returns StructuralElement objects with ListStructure content

Follows parameterized strategy pattern (ARCHITECTURAL_DECISIONS.md #3)
Implements processing traces for reproducibility (#5)
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Pattern
from datetime import datetime
from enum import Enum
import logging

from data_models import (
    OCRTextResult,
    StructuralElement,
    ElementType,
    BoundingBox,
    ListItem,
    ListStructure,
    PSMMode,
)
from ocr_engine import OCREngine, OCREngineConfig


logger = logging.getLogger(__name__)


class ListMarkerType(Enum):
    """Types of list markers."""
    BULLET = "bullet"          # •, -, *, +
    NUMBER = "number"          # 1., 2., (1), 1)
    LETTER = "letter"          # a., b., (a), a)
    ROMAN = "roman"            # i., ii., I., II.
    DASH = "dash"              # —, --, ---
    MIXED = "mixed"            # Multiple types in same list


@dataclass
class ListDetectorConfig:
    """
    Configuration for list detection (bullet, numbered, lettered).
    
    Attributes:
        # Detection flags
        detect_lists: Enable list detection
        detect_nested_lists: Detect indentation-based nesting
        
        # Marker detection
        marker_types: Which marker types to detect
        bullet_chars: Characters to treat as bullets
        min_items_for_list: Minimum items to constitute a list (default 1)
        
        # Indentation analysis
        indentation_threshold: Pixels to detect nesting level change
        indentation_unit: Expected indentation per level (pixels)
        
        # List item detection
        min_item_confidence: Minimum confidence for list items
        min_item_text_length: Minimum characters in item content
        
        # General
        min_confidence: Global minimum confidence threshold
        language: Language for OCR
        enable_preprocessing: Apply image preprocessing
    """
    
    # Detection flags
    detect_lists: bool = True
    detect_nested_lists: bool = True
    
    # Marker detection
    marker_types: List[ListMarkerType] = field(
        default_factory=lambda: [ListMarkerType.BULLET, ListMarkerType.NUMBER, ListMarkerType.LETTER]
    )
    bullet_chars: str = "•-*+"
    min_items_for_list: int = 1
    
    # Indentation analysis
    indentation_threshold: float = 5.0  # pixels
    indentation_unit: float = 20.0      # pixels per level
    
    # List item detection
    min_item_confidence: float = 0.3
    min_item_text_length: int = 1
    
    # General
    min_confidence: float = 0.3
    language: str = "eng"
    enable_preprocessing: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 <= self.min_item_confidence <= 1.0):
            raise ValueError("min_item_confidence must be 0-1")
        if self.indentation_threshold < 0:
            raise ValueError("indentation_threshold must be >= 0")
        if self.indentation_unit <= 0:
            raise ValueError("indentation_unit must be > 0")
        if self.min_items_for_list < 1:
            raise ValueError("min_items_for_list must be >= 1")


@dataclass
class ListDetectionTrace:
    """
    Processing trace for list detection.
    
    Attributes:
        config: ListDetectorConfig used
        processing_start: When detection started
        processing_end: When detection completed
        image_dimensions: Image size in pixels
        
        # Detection results
        lists_found: Number of lists detected
        items_found: Number of list items detected
        
        # Analysis
        ocr_results_analyzed: How many OCRTextResult objects were analyzed
        marker_types_detected: Set of marker types found
        nesting_levels_found: Maximum nesting depth detected
        marker_detection_matches: Count of successful marker matches
        
        # Timing
        detection_time_ms: Time spent on detection
        hierarchy_building_time_ms: Time spent building parent-child relationships
    """
    config: ListDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]
    
    lists_found: int = 0
    items_found: int = 0
    ocr_results_analyzed: int = 0
    marker_types_detected: set = field(default_factory=set)
    nesting_levels_found: int = 0
    marker_detection_matches: int = 0
    
    detection_time_ms: float = 0.0
    hierarchy_building_time_ms: float = 0.0
    
    @property
    def total_processing_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        delta = self.processing_end - self.processing_start
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "config": {
                "detect_lists": self.config.detect_lists,
                "detect_nested_lists": self.config.detect_nested_lists,
                "marker_types": [mt.value for mt in self.config.marker_types],
                "min_confidence": self.config.min_confidence,
            },
            "results": {
                "lists": self.lists_found,
                "items": self.items_found,
                "marker_types_detected": list(self.marker_types_detected),
                "max_nesting_level": self.nesting_levels_found,
            },
            "analysis": {
                "ocr_results_analyzed": self.ocr_results_analyzed,
                "marker_matches": self.marker_detection_matches,
            },
            "timing_ms": {
                "total": self.total_processing_time_ms,
                "detection": self.detection_time_ms,
                "hierarchy_building": self.hierarchy_building_time_ms,
            },
        }


class ListDetector:
    """
    Detector for list-based structural elements (bullet, numbered, lettered).
    
    Strategy:
    - Runs OCR with PSM 6 (single line) to extract individual list items
    - Detects markers (bullets, numbers, letters) at start of each line
    - Analyzes indentation to build hierarchy (nesting levels)
    - Groups consecutive items into ListStructure objects
    - Returns StructuralElement objects with ListStructure content
    
    Usage:
        config = ListDetectorConfig(detect_lists=True)
        detector = ListDetector(config)
        
        elements, trace = detector.detect_lists(
            image=image,
            page_number=1,
            ocr_results=ocr_results
        )
        
        for element in elements:
            list_struct = element.content
            print(f"Found list with {len(list_struct.items)} items")
    """
    
    def __init__(self, config: Optional[ListDetectorConfig] = None):
        """
        Initialize list detector with configuration.
        
        Args:
            config: ListDetectorConfig. Uses defaults if None.
        """
        self.config = config or ListDetectorConfig()
        
        # Initialize OCR engine for line-by-line detection
        self.ocr_config = OCREngineConfig(
            psm_modes=[PSMMode.SINGLE_LINE, PSMMode.SINGLE_BLOCK],
            languages=self.config.language,
            enable_preprocessing=self.config.enable_preprocessing,
            min_confidence=self.config.min_item_confidence,
        )
        self.ocr_engine = OCREngine(self.ocr_config)
        
        # Compile regex patterns for marker detection
        self._compile_marker_patterns()
    
    def _compile_marker_patterns(self) -> None:
        """Compile regex patterns for all marker types."""
        # Bullet pattern: •, -, *, +
        self.bullet_pattern = re.compile(r"^[\s]*([•\-*+])\s+(.+)")
        
        # Number pattern: 1., (1), 1), 1-
        self.number_pattern = re.compile(r"^[\s]*(\(?(\d+)[.):-])\s+(.+)")
        
        # Letter pattern: a., (a), a), a-
        self.letter_pattern = re.compile(r"^[\s]*(\(?([a-z])[.):-])\s+(.+)")
        
        # Roman numeral pattern: i., I., (i), (I), i), I)
        self.roman_pattern = re.compile(r"^[\s]*(([ivxlcdm]+|[IVXLCDM]+)[.):-])\s+(.+)")
    
    def _detect_marker(self, line: str) -> Tuple[Optional[ListMarkerType], Optional[str], str]:
        """
        Detect marker type in a line of text.
        
        Returns:
            (marker_type, marker_text, item_content) or (None, None, line) if no marker found
        """
        # Try bullet first (fastest)
        match = self.bullet_pattern.match(line)
        if match and ListMarkerType.BULLET in self.config.marker_types:
            return ListMarkerType.BULLET, match.group(1), match.group(2)
        
        # Try number
        match = self.number_pattern.match(line)
        if match and ListMarkerType.NUMBER in self.config.marker_types:
            return ListMarkerType.NUMBER, match.group(1), match.group(3)
        
        # Try letter
        match = self.letter_pattern.match(line)
        if match and ListMarkerType.LETTER in self.config.marker_types:
            return ListMarkerType.LETTER, match.group(1), match.group(3)
        
        # Try roman numeral
        match = self.roman_pattern.match(line)
        if match and ListMarkerType.ROMAN in self.config.marker_types:
            return ListMarkerType.ROMAN, match.group(1), match.group(3)
        
        return None, None, line
    
    def _get_indentation_level(self, line: str) -> int:
        """
        Detect indentation level from leading whitespace.
        
        Returns:
            Nesting level (0 = no indentation, 1+ = nested)
        """
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces == 0:
            return 0
        level = int(leading_spaces / self.config.indentation_unit)
        return max(0, level)
    
    def _extract_item_number(self, marker_text: str, marker_type: ListMarkerType) -> Optional[int]:
        """
        Extract numeric value from marker (for numbered/lettered lists).
        
        Args:
            marker_text: The marker string (e.g., "1.", "a.", "i.")
            marker_type: Type of marker
        
        Returns:
            Numeric position or None
        """
        if marker_type == ListMarkerType.NUMBER:
            # Extract digits from marker
            digits = ''.join(c for c in marker_text if c.isdigit())
            if digits:
                return int(digits)
        elif marker_type == ListMarkerType.LETTER:
            # Extract letter and convert to position
            letter = ''.join(c for c in marker_text if c.isalpha())
            if letter and letter.islower():
                return ord(letter) - ord('a') + 1
            elif letter and letter.isupper():
                return ord(letter) - ord('A') + 1
        
        return None
    
    def _build_list_hierarchy(self, items_with_metadata: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
        """
        Build parent-child hierarchy based on indentation levels.
        
        Args:
            items_with_metadata: List of dicts with item_id, level, etc.
        
        Returns:
            (root_item_ids, parent_map) where parent_map is {child_id: parent_id}
        """
        root_item_ids = []
        parent_map = {}
        level_stack = []  # Stack of (level, item_id) to track hierarchy
        
        for item_meta in items_with_metadata:
            item_id = item_meta["item_id"]
            level = item_meta["level"]
            
            # Find parent based on indentation
            parent_id = None
            while level_stack and level_stack[-1][0] >= level:
                level_stack.pop()
            
            if level_stack:
                parent_id = level_stack[-1][1]
                parent_map[item_id] = parent_id
            else:
                root_item_ids.append(item_id)
            
            level_stack.append((level, item_id))
        
        return root_item_ids, parent_map
    
    def detect_lists(
        self,
        image: np.ndarray,
        page_number: int = 1,
        ocr_results: Optional[List[OCRTextResult]] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], ListDetectionTrace]:
        """
        Detect list structures in image.
        
        Args:
            image: Image as numpy array
            page_number: Page number for this image
            ocr_results: Pre-extracted OCR results (optional, will re-run if not provided)
            image_path: Path to image file (for reference/debugging)
        
        Returns:
            (elements, trace) where:
            - elements: List of StructuralElement objects with ListStructure content
            - trace: ListDetectionTrace with detailed results
        """
        start_time = datetime.now()
        
        # Get or extract OCR results
        if ocr_results is None:
            ocr_results = self.ocr_engine.extract_text(image, page_number=page_number)
        
        # If no OCR results, return early
        if not ocr_results:
            trace = ListDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
            )
            return [], trace
        
        # Sort OCR results by Y position (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda r: (r.bbox.y_min, r.bbox.x_min))
        
        # Phase 1: Detect markers and extract items
        items_with_metadata = []
        marker_types_found = set()
        
        for ocr_result in sorted_results:
            if ocr_result.confidence < self.config.min_item_confidence:
                continue
            
            text = ocr_result.text.strip()
            if not text or len(text) < self.config.min_item_text_length:
                continue
            
            marker_type, marker_text, item_content = self._detect_marker(text)
            
            if marker_type is None:
                continue  # Skip lines without markers
            
            marker_types_found.add(marker_type.value)
            
            # Get indentation level
            level = self._get_indentation_level(text)
            
            # Extract item number if applicable
            item_number = self._extract_item_number(marker_text, marker_type)
            
            # Create ListItem
            item_id = f"list_item_{len(items_with_metadata)}"
            list_item = ListItem(
                content=item_content,
                level=level,
                bbox=ocr_result.bbox,
                confidence=ocr_result.confidence,
                list_type=marker_type.value,
                number=item_number,
                parent_item_id=None,  # Will be filled in hierarchy building
                child_item_ids=[]
            )
            
            items_with_metadata.append({
                "item_id": item_id,
                "list_item": list_item,
                "level": level,
                "marker_type": marker_type,
            })
        
        # If no items found, return early
        if len(items_with_metadata) < self.config.min_items_for_list:
            trace = ListDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                ocr_results_analyzed=len(sorted_results),
                marker_detection_matches=len(items_with_metadata),
            )
            return [], trace
        
        hierarchy_start = datetime.now()
        
        # Phase 2: Build hierarchy
        root_item_ids, parent_map = self._build_list_hierarchy(items_with_metadata)
        
        # Update parent/child relationships
        item_map = {item_meta["item_id"]: item_meta["list_item"] for item_meta in items_with_metadata}
        
        for child_id, parent_id in parent_map.items():
            if parent_id in item_map and child_id in item_map:
                child = item_map[child_id]
                parent = item_map[parent_id]
                child.parent_item_id = parent_id
                parent.child_item_ids.append(child_id)
        
        hierarchy_time = (datetime.now() - hierarchy_start).total_seconds() * 1000
        
        # Phase 3: Group items into ListStructure objects
        # For now, we treat all items as a single list (can extend later for multi-list detection)
        all_items = [item_meta["list_item"] for item_meta in items_with_metadata]
        
        # Compute list statistics
        avg_confidence = sum(item.confidence for item in all_items) / len(all_items) if all_items else 0.0
        
        # Compute bounding box
        if all_items:
            x_min = min(item.bbox.x_min for item in all_items)
            y_min = min(item.bbox.y_min for item in all_items)
            x_max = max(item.bbox.x_max for item in all_items)
            y_max = max(item.bbox.y_max for item in all_items)
            list_bbox = BoundingBox(x_min, y_min, x_max, y_max)
        else:
            list_bbox = BoundingBox(0, 0, image.shape[1], image.shape[0])
        
        # Determine list type
        list_type = "mixed" if len(marker_types_found) > 1 else (list(marker_types_found)[0] if marker_types_found else "bullet")
        
        # Create ListStructure
        list_structure = ListStructure(
            items=all_items,
            root_item_ids=root_item_ids,
            bbox=list_bbox,
            confidence=avg_confidence,
            list_type=list_type,
        )
        
        # Create StructuralElement wrapping the ListStructure
        element = StructuralElement(
            element_id=f"list_{len(items_with_metadata)}_items",
            element_type=ElementType.LIST,
            content=list_structure,
            bbox=list_bbox,
            confidence=avg_confidence,
            page_number=page_number,
            nesting_level=0,
            metadata={
                "list_type": list_type,
                "item_count": len(all_items),
                "max_nesting_level": max(item.level for item in all_items) if all_items else 0,
                "marker_types": list(marker_types_found),
            },
            processing_method="list_detector_marker_based",
            source_ocr_results=[f"ocr_result_{i}" for i in range(len(ocr_results))],
        )
        
        end_time = datetime.now()
        
        # Create trace
        max_nesting = max(item.level for item in all_items) if all_items else 0
        trace = ListDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions=image.shape[:2],
            lists_found=1 if items_with_metadata else 0,
            items_found=len(all_items),
            ocr_results_analyzed=len(sorted_results),
            marker_types_detected=marker_types_found,
            nesting_levels_found=max_nesting,
            marker_detection_matches=len(items_with_metadata),
            detection_time_ms=(hierarchy_start - start_time).total_seconds() * 1000,
            hierarchy_building_time_ms=hierarchy_time,
        )
        
        logger.info(f"List detection complete: {len(all_items)} items in {len(items_with_metadata)} groups, trace: {trace.to_dict()}")
        
        return [element] if items_with_metadata else [], trace
