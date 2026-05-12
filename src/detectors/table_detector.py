"""
Table Detection Module (Line-Based) - Grid Detection via Hough Transform

This module implements table detection using line-based approaches:
1. TABLE - Structured table data with cells, rows, columns, merged cells
2. TABLE_CELL - Individual cells with row/col indices, content, styling

Design:
- Uses Hough transform to detect horizontal and vertical lines
- Intersects lines to identify cell boundaries
- Extracts text from each cell via Tesseract
- Builds TableStructure with cell grid
- Detects merged cells (colspan/rowspan)
- Returns StructuralElement objects with TableStructure content

Follows parameterized strategy pattern (ARCHITECTURAL_DECISIONS.md #3)
Implements processing traces for reproducibility (#5)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

from data_models import (
    OCRTextResult,
    StructuralElement,
    ElementType,
    BoundingBox,
    TableCell,
    TableStructure,
)


logger = logging.getLogger(__name__)


@dataclass
class TableDetectorConfig:
    """
    Configuration for table detection (line-based via Hough transform).
    
    Attributes:
        # Detection flags
        detect_tables: Enable table detection
        detect_merged_cells: Enable merged cell detection
        detect_table_headers: Detect header rows
        
        # Line detection (Hough transform)
        hough_rho_resolution: Distance resolution in pixels
        hough_theta_resolution: Angle resolution in radians
        hough_threshold: Minimum votes to detect line
        hough_line_gap_pixels: Maximum gap between line segments
        hough_min_line_length: Minimum line length in pixels
        
        # Grid processing
        line_clustering_threshold: Distance to merge similar lines (pixels)
        min_intersection_distance: Minimum distance between line intersections
        
        # Cell extraction
        min_cell_width: Minimum cell width in pixels
        min_cell_height: Minimum cell height in pixels
        min_cell_confidence: Minimum confidence for cell content
        
        # Table structure
        min_cells_for_table: Minimum cells to constitute a table (default 4)
        min_table_area: Minimum bounding box area in pixels
        
        # General
        language: Language for OCR
        enable_preprocessing: Apply image preprocessing
    """
    
    # Detection flags
    detect_tables: bool = True
    detect_merged_cells: bool = True
    detect_table_headers: bool = True
    
    # Line detection (Hough transform)
    hough_rho_resolution: float = 1.0           # pixels
    hough_theta_resolution: float = np.pi / 180  # 1 degree
    hough_threshold: int = 50                   # minimum votes
    hough_line_gap_pixels: int = 10             # max gap between segments
    hough_min_line_length: int = 50             # minimum line length
    
    # Grid processing
    line_clustering_threshold: float = 5.0      # pixels
    min_intersection_distance: float = 2.0      # pixels
    
    # Cell extraction
    min_cell_width: float = 10.0                # pixels
    min_cell_height: float = 10.0               # pixels
    min_cell_confidence: float = 0.1            # very low threshold (OCR may fail in cells)
    
    # Table structure
    min_cells_for_table: int = 4                # at least 2x2
    min_table_area: float = 100.0               # square pixels
    
    # General
    language: str = "eng"
    enable_preprocessing: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.hough_threshold < 0:
            raise ValueError("hough_threshold must be >= 0")
        if self.line_clustering_threshold < 0:
            raise ValueError("line_clustering_threshold must be >= 0")
        if self.min_cell_width <= 0 or self.min_cell_height <= 0:
            raise ValueError("min_cell_width and min_cell_height must be > 0")
        if self.min_cells_for_table < 1:
            raise ValueError("min_cells_for_table must be >= 1")
        if self.min_table_area <= 0:
            raise ValueError("min_table_area must be > 0")


@dataclass
class TableDetectionTrace:
    """
    Processing trace for table detection.
    
    Attributes:
        config: TableDetectorConfig used
        processing_start: When detection started
        processing_end: When detection completed
        image_dimensions: Image size in pixels
        
        # Detection results
        tables_found: Number of tables detected
        cells_found: Total cells across all tables
        merged_cells_detected: Number of merged cells
        
        # Line detection analysis
        horizontal_lines_found: Number of horizontal lines detected
        vertical_lines_found: Number of vertical lines detected
        lines_after_clustering: After merging similar lines
        line_intersections_found: Grid intersection points
        
        # Cell extraction
        cells_with_text: Cells that contained readable text
        cells_without_text: Empty cells
        average_cell_confidence: Mean confidence of cell text
        
        # Timing
        line_detection_time_ms: Time for Hough transform
        grid_extraction_time_ms: Time for intersection detection
        cell_extraction_time_ms: Time for OCR and cell grouping
        merged_cell_detection_time_ms: Time for colspan/rowspan detection
    """
    config: TableDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]
    
    tables_found: int = 0
    cells_found: int = 0
    merged_cells_detected: int = 0
    
    horizontal_lines_found: int = 0
    vertical_lines_found: int = 0
    lines_after_clustering: int = 0
    line_intersections_found: int = 0
    
    cells_with_text: int = 0
    cells_without_text: int = 0
    average_cell_confidence: float = 0.0
    
    line_detection_time_ms: float = 0.0
    grid_extraction_time_ms: float = 0.0
    cell_extraction_time_ms: float = 0.0
    merged_cell_detection_time_ms: float = 0.0
    
    @property
    def total_processing_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        delta = self.processing_end - self.processing_start
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "config": {
                "detect_tables": self.config.detect_tables,
                "detect_merged_cells": self.config.detect_merged_cells,
                "hough_threshold": self.config.hough_threshold,
                "min_cells_for_table": self.config.min_cells_for_table,
            },
            "results": {
                "tables": self.tables_found,
                "cells": self.cells_found,
                "merged_cells": self.merged_cells_detected,
            },
            "line_detection": {
                "horizontal_lines": self.horizontal_lines_found,
                "vertical_lines": self.vertical_lines_found,
                "after_clustering": self.lines_after_clustering,
                "intersections": self.line_intersections_found,
            },
            "cell_extraction": {
                "with_text": self.cells_with_text,
                "without_text": self.cells_without_text,
                "average_confidence": self.average_cell_confidence,
            },
            "timing_ms": {
                "total": self.total_processing_time_ms,
                "line_detection": self.line_detection_time_ms,
                "grid_extraction": self.grid_extraction_time_ms,
                "cell_extraction": self.cell_extraction_time_ms,
                "merged_cell_detection": self.merged_cell_detection_time_ms,
            },
        }


class TableDetector:
    """
    Detector for table-based structural elements using line-based detection.
    
    Strategy:
    - Applies Hough transform to detect horizontal and vertical lines
    - Clusters similar lines and removes duplicates
    - Finds line intersections to identify cell grid
    - Extracts text from each cell via OCR
    - Builds TableStructure from grid and cell content
    - Detects merged cells via colspan/rowspan analysis
    - Returns StructuralElement objects with TableStructure content
    
    Limitations:
    - Requires visible grid lines (doesn't work on layout tables)
    - May fail on rotated/skewed tables (future: add skew detection)
    - Requires sufficient spacing between cells
    
    Fallback:
    - See ContentBasedTableDetector for tables without visible grid lines
    
    Usage:
        config = TableDetectorConfig(detect_tables=True)
        detector = TableDetector(config)
        
        elements, trace = detector.detect_tables(
            image=image,
            page_number=1
        )
        
        for element in elements:
            table_struct = element.content
            print(f"Found table: {table_struct.num_rows}x{table_struct.num_cols}")
    """
    
    def __init__(self, config: Optional[TableDetectorConfig] = None):
        """
        Initialize table detector with configuration.
        
        Args:
            config: TableDetectorConfig. Uses defaults if None.
        """
        self.config = config or TableDetectorConfig()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for line detection.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Preprocessed binary image suitable for Hough transform
        """
        if len(image.shape) == 3:
            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def _detect_lines(self, edges: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Detect lines in edge image using Hough transform.
        
        Args:
            edges: Edge-detected binary image
        
        Returns:
            (horizontal_lines, vertical_lines) as lists of (x1, y1, x2, y2)
        """
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges,
            rho=self.config.hough_rho_resolution,
            theta=self.config.hough_theta_resolution,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_line_gap_pixels,
        )
        
        if lines is None or len(lines) == 0:
            return [], []
        
        # Classify lines as horizontal or vertical
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # Horizontal line: more horizontal extent than vertical
            if dx > dy:
                horizontal_lines.append((x1, y1, x2, y2))
            # Vertical line: more vertical extent than horizontal
            else:
                vertical_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines, vertical_lines
    
    def _cluster_lines(self, lines: List[Tuple[int, int, int, int]], vertical: bool = False) -> List[float]:
        """
        Cluster similar lines and return their consolidated positions.
        
        Args:
            lines: List of line endpoints (x1, y1, x2, y2)
            vertical: Whether these are vertical lines (else horizontal)
        
        Returns:
            List of consolidated line positions (y-coords for horizontal, x-coords for vertical)
        """
        if not lines:
            return []
        
        # Extract relevant coordinate (y for horizontal, x for vertical)
        coords = []
        for x1, y1, x2, y2 in lines:
            if vertical:
                # For vertical lines, use average x coordinate
                avg_x = (x1 + x2) / 2
                coords.append(avg_x)
            else:
                # For horizontal lines, use average y coordinate
                avg_y = (y1 + y2) / 2
                coords.append(avg_y)
        
        # Sort coordinates
        coords.sort()
        
        # Cluster by grouping close coordinates
        clustered = []
        current_cluster = [coords[0]]
        
        for coord in coords[1:]:
            if coord - current_cluster[-1] <= self.config.line_clustering_threshold:
                current_cluster.append(coord)
            else:
                # Finalize current cluster and start new one
                clustered.append(np.mean(current_cluster))
                current_cluster = [coord]
        
        # Don't forget last cluster
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    def _find_intersections(self, h_lines: List[float], v_lines: List[float]) -> List[Tuple[float, float]]:
        """
        Find intersections of horizontal and vertical lines.
        
        Args:
            h_lines: List of horizontal line y-coordinates
            v_lines: List of vertical line x-coordinates
        
        Returns:
            List of (x, y) intersection points
        """
        intersections = []
        
        for x in v_lines:
            for y in h_lines:
                intersections.append((x, y))
        
        return intersections
    
    def _build_cell_grid(self, intersections: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
        """
        Build regular grid from intersection points.
        
        Args:
            intersections: List of line intersections
        
        Returns:
            (sorted_x_coords, sorted_y_coords) for grid
        """
        if not intersections:
            return [], []
        
        # Extract unique x and y coordinates
        x_coords = sorted(set(x for x, y in intersections))
        y_coords = sorted(set(y for x, y in intersections))
        
        return x_coords, y_coords
    
    def _extract_cells(
        self,
        image: np.ndarray,
        x_coords: List[float],
        y_coords: List[float],
    ) -> List[TableCell]:
        """
        Extract text from cells defined by grid coordinates.
        
        Args:
            image: Original image for text extraction
            x_coords: Sorted x-coordinates of vertical lines
            y_coords: Sorted y-coordinates of horizontal lines
        
        Returns:
            List of TableCell objects
        """
        cells = []
        cell_id = 0
        
        # Iterate through grid cells
        for row_idx in range(len(y_coords) - 1):
            for col_idx in range(len(x_coords) - 1):
                x_min = int(x_coords[col_idx])
                y_min = int(y_coords[row_idx])
                x_max = int(x_coords[col_idx + 1])
                y_max = int(y_coords[row_idx + 1])
                
                # Check cell dimensions
                width = x_max - x_min
                height = y_max - y_min
                
                if width < self.config.min_cell_width or height < self.config.min_cell_height:
                    continue  # Skip too-small cells
                
                # Extract cell content via OCR (if cv2 and pytesseract available)
                content = self._extract_cell_text(image, x_min, y_min, x_max, y_max)
                confidence = 0.5 if content else 0.0  # Placeholder confidence
                
                # Create TableCell
                cell = TableCell(
                    content=content,
                    row_index=row_idx,
                    col_index=col_idx,
                    bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    confidence=confidence,
                    colspan=1,
                    rowspan=1,
                    is_header=(row_idx == 0),  # Assume first row is header
                )
                
                cells.append(cell)
                cell_id += 1
        
        return cells
    
    def _extract_cell_text(self, image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> str:
        """
        Extract text from a cell region via OCR.
        
        Args:
            image: Original image
            x_min, y_min, x_max, y_max: Cell bounds
        
        Returns:
            Extracted text (empty string if extraction fails or cv2 not available)
        """
        try:
            import pytesseract
        except ImportError:
            logger.debug("Tesseract not available, returning empty cell")
            return ""

        # Extract cell region
        if len(image.shape) == 3:
            cell_image = image[y_min:y_max, x_min:x_max, :]
        else:
            cell_image = image[y_min:y_max, x_min:x_max]
        
        # Apply preprocessing
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image
        
        # Threshold to binary (helps OCR)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Extract text via OCR
        try:
            text = pytesseract.image_to_string(binary, lang=self.config.language).strip()
            return text
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")
            return ""
    
    def _detect_merged_cells(self, cells: List[TableCell]) -> List[TableCell]:
        """
        Detect merged cells (colspan/rowspan) via empty-slot inference.

        Strategy:
        1. Build a grid occupancy map from the extracted cells.
        2. Find every (row, col) slot that has no cell (these are the "holes"
           left when a neighbouring cell physically spans across them).
        3. For each hole, walk left to find the nearest filled cell in the
           same row — that cell gains +1 colspan.  If no left neighbour exists,
           walk up to find the nearest filled cell in the same column — that
           cell gains +1 rowspan.
        4. Repeat until no new holes are found (handles multi-cell spans).

        Args:
            cells: Cells extracted directly from the grid (all colspan=rowspan=1).

        Returns:
            Updated cell list with colspan/rowspan set; empty-slot placeholder
            cells are removed so the list only contains logical cells.
        """
        if not cells:
            return cells

        max_row = max(c.row_index for c in cells)
        max_col = max(c.col_index for c in cells)

        # Index filled slots
        cell_map: Dict[Tuple[int, int], TableCell] = {
            (c.row_index, c.col_index): c for c in cells
        }

        changed = True
        while changed:
            changed = False
            for row in range(max_row + 1):
                for col in range(max_col + 1):
                    if (row, col) in cell_map:
                        continue  # slot already occupied

                    # Try left neighbour first (colspan extension)
                    for left_col in range(col - 1, -1, -1):
                        neighbour = cell_map.get((row, left_col))
                        if neighbour is not None:
                            neighbour.colspan += 1
                            # Mark slot as covered so we don't process it again
                            cell_map[(row, col)] = neighbour
                            changed = True
                            break
                    else:
                        # No left neighbour — try cell above (rowspan extension)
                        for up_row in range(row - 1, -1, -1):
                            neighbour = cell_map.get((up_row, col))
                            if neighbour is not None:
                                neighbour.rowspan += 1
                                cell_map[(row, col)] = neighbour
                                changed = True
                                break

        # Return only the original logical cells (deduplicated by identity)
        seen_ids = set()
        result = []
        for c in cells:
            cid = id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                result.append(c)

        logger.debug(
            "Merged cell detection: %d logical cells from %d grid slots "
            "(colspan/rowspan updated)",
            len(result), (max_row + 1) * (max_col + 1),
        )
        return result
    
    def detect_tables(
        self,
        image: np.ndarray,
        page_number: int = 1,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], TableDetectionTrace]:
        """
        Detect table structures in image via line-based detection.
        
        Args:
            image: Image as numpy array
            page_number: Page number for this image
            image_path: Path to image file (for reference/debugging)
        
        Returns:
            (elements, trace) where:
            - elements: List of StructuralElement objects with TableStructure content
            - trace: TableDetectionTrace with detailed results
        """
        start_time = datetime.now()
        elements = []
        
        if not self.config.detect_tables:
            trace = TableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
            )
            return [], trace
        
        # Phase 1: Preprocess and detect lines
        line_start = datetime.now()
        edges = self._preprocess_image(image)
        h_lines_raw, v_lines_raw = self._detect_lines(edges)
        line_time = (datetime.now() - line_start).total_seconds() * 1000
        
        if not h_lines_raw or not v_lines_raw:
            trace = TableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                line_detection_time_ms=line_time,
                horizontal_lines_found=len(h_lines_raw),
                vertical_lines_found=len(v_lines_raw),
            )
            logger.debug(f"No lines detected: h={len(h_lines_raw)}, v={len(v_lines_raw)}")
            return [], trace
        
        # Phase 2: Cluster lines
        grid_start = datetime.now()
        h_lines = self._cluster_lines(h_lines_raw, vertical=False)
        v_lines = self._cluster_lines(v_lines_raw, vertical=True)
        grid_time = (datetime.now() - grid_start).total_seconds() * 1000
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            trace = TableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                line_detection_time_ms=line_time,
                grid_extraction_time_ms=grid_time,
                horizontal_lines_found=len(h_lines_raw),
                vertical_lines_found=len(v_lines_raw),
                lines_after_clustering=len(h_lines) + len(v_lines),
            )
            logger.debug(f"Too few lines after clustering: h={len(h_lines)}, v={len(v_lines)}")
            return [], trace
        
        # Phase 3: Find intersections and build grid
        intersections = self._find_intersections(h_lines, v_lines)
        x_coords, y_coords = self._build_cell_grid(intersections)
        
        # Phase 4: Extract cells
        cell_start = datetime.now()
        cells = self._extract_cells(image, x_coords, y_coords)
        cell_time = (datetime.now() - cell_start).total_seconds() * 1000
        
        if len(cells) < self.config.min_cells_for_table:
            trace = TableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                line_detection_time_ms=line_time,
                grid_extraction_time_ms=grid_time,
                cell_extraction_time_ms=cell_time,
                horizontal_lines_found=len(h_lines_raw),
                vertical_lines_found=len(v_lines_raw),
                lines_after_clustering=len(h_lines) + len(v_lines),
                line_intersections_found=len(intersections),
                cells_found=len(cells),
            )
            logger.debug(f"Too few cells: {len(cells)} < {self.config.min_cells_for_table}")
            return [], trace
        
        # Phase 5: Detect merged cells (if enabled)
        merged_start = datetime.now()
        if self.config.detect_merged_cells:
            cells = self._detect_merged_cells(cells)
        merged_time = (datetime.now() - merged_start).total_seconds() * 1000
        
        # Phase 6: Build TableStructure
        merged_count = sum(1 for c in cells if c.colspan > 1 or c.rowspan > 1)
        text_count = sum(1 for c in cells if c.content)
        avg_conf = sum(c.confidence for c in cells) / len(cells) if cells else 0.0
        
        x_min = min(x_coords) if x_coords else 0
        y_min = min(y_coords) if y_coords else 0
        x_max = max(x_coords) if x_coords else image.shape[1]
        y_max = max(y_coords) if y_coords else image.shape[0]
        table_bbox = BoundingBox(x_min, y_min, x_max, y_max)
        
        table_struct = TableStructure(
            cells=cells,
            bbox=table_bbox,
            confidence=avg_conf,
            num_rows=len(y_coords) - 1 if len(y_coords) > 1 else 0,
            num_cols=len(x_coords) - 1 if len(x_coords) > 1 else 0,
            table_type="data_table",
            has_irregular_structure=(merged_count > 0),
        )
        
        # Create StructuralElement
        element = StructuralElement(
            element_id=f"table_{len(cells)}_cells",
            element_type=ElementType.TABLE,
            content=table_struct,
            bbox=table_bbox,
            confidence=avg_conf,
            page_number=page_number,
            nesting_level=0,
            metadata={
                "num_cells": len(cells),
                "num_rows": table_struct.num_rows,
                "num_cols": table_struct.num_cols,
                "merged_cells": merged_count,
                "cells_with_text": text_count,
            },
            processing_method="table_detector_hough_lines",
            source_ocr_results=[],
        )
        
        elements.append(element)
        
        # Create trace
        end_time = datetime.now()
        trace = TableDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions=image.shape[:2],
            tables_found=1,
            cells_found=len(cells),
            merged_cells_detected=merged_count,
            horizontal_lines_found=len(h_lines_raw),
            vertical_lines_found=len(v_lines_raw),
            lines_after_clustering=len(h_lines) + len(v_lines),
            line_intersections_found=len(intersections),
            cells_with_text=text_count,
            cells_without_text=len(cells) - text_count,
            average_cell_confidence=avg_conf,
            line_detection_time_ms=line_time,
            grid_extraction_time_ms=grid_time,
            cell_extraction_time_ms=cell_time,
            merged_cell_detection_time_ms=merged_time,
        )
        
        logger.info(f"Table detection complete: {table_struct.num_rows}x{table_struct.num_cols} table, {len(cells)} cells, trace: {trace.to_dict()}")
        
        return elements, trace
