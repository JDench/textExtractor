"""
Table Detection Module (Content-Based) - Fallback Detection via Spatial Analysis

This module implements table detection using content-based approaches (fallback):
1. TABLE - Structured table data detected from spatial distribution of text
2. TABLE_CELL - Cells inferred from text clustering and alignment

Design:
- Uses spatial distribution of OCR results to detect columns and rows
- Analyzes x-alignment to detect column boundaries
- Analyzes y-alignment to detect row boundaries
- Groups text into cells via proximity clustering
- Builds TableStructure from inferred grid
- Less accurate than line-based but works on all documents

Follows parameterized strategy pattern (ARCHITECTURAL_DECISIONS.md #3)
Implements processing traces for reproducibility (#5)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
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
class ContentTableDetectorConfig:
    """
    Configuration for content-based table detection (spatial analysis).
    
    Attributes:
        # Detection flags
        detect_tables: Enable table detection
        detect_columns: Enable column boundary detection
        detect_rows: Enable row boundary detection
        
        # Column/Row detection
        column_alignment_threshold: Max horizontal distance for x-alignment (pixels)
        row_alignment_threshold: Max vertical distance for y-alignment (pixels)
        min_column_width: Minimum column width (pixels)
        min_row_height: Minimum row height (pixels)
        column_gap_threshold: Gap to detect column boundary (pixels)
        row_gap_threshold: Gap to detect row boundary (pixels)
        
        # Cell clustering
        cell_cluster_distance: Max distance to group text into cell (pixels)
        min_texts_per_cell: Minimum OCR results to constitute a cell (usually 1)
        
        # Table structure
        min_cells_for_table: Minimum cells to constitute a table (default 4)
        min_rows: Minimum row count (default 2)
        min_cols: Minimum column count (default 2)
        min_text_alignment_ratio: Percentage of cells that should align (0-1)
        
        # General
        language: Language for OCR
    """
    
    # Detection flags
    detect_tables: bool = True
    detect_columns: bool = True
    detect_rows: bool = True
    
    # Column/Row detection
    column_alignment_threshold: float = 10.0   # pixels
    row_alignment_threshold: float = 5.0       # pixels
    min_column_width: float = 20.0             # pixels
    min_row_height: float = 15.0               # pixels
    column_gap_threshold: float = 30.0         # pixels
    row_gap_threshold: float = 15.0            # pixels
    
    # Cell clustering
    cell_cluster_distance: float = 50.0        # pixels
    min_texts_per_cell: int = 1
    
    # Table structure
    min_cells_for_table: int = 4
    min_rows: int = 2
    min_cols: int = 2
    min_text_alignment_ratio: float = 0.3
    
    # General
    language: str = "eng"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.column_alignment_threshold < 0:
            raise ValueError("column_alignment_threshold must be >= 0")
        if self.row_alignment_threshold < 0:
            raise ValueError("row_alignment_threshold must be >= 0")
        if self.min_column_width <= 0 or self.min_row_height <= 0:
            raise ValueError("min_column_width and min_row_height must be > 0")
        if not 0.0 <= self.min_text_alignment_ratio <= 1.0:
            raise ValueError("min_text_alignment_ratio must be in [0, 1]")


@dataclass
class ContentTableDetectionTrace:
    """
    Processing trace for content-based table detection.
    
    Attributes:
        config: ContentTableDetectorConfig used
        processing_start: When detection started
        processing_end: When detection completed
        image_dimensions: Image size in pixels
        
        # Detection results
        tables_found: Number of tables detected
        cells_found: Total cells across all tables
        text_regions_analyzed: Number of OCR results processed
        
        # Column/Row analysis
        potential_columns_found: Before filtering
        columns_after_filtering: After size/alignment checks
        potential_rows_found: Before filtering
        rows_after_filtering: After size/alignment checks
        
        # Cell creation
        cells_created: Number of cells successfully created
        cells_with_text: Cells that contained text
        cells_empty: Empty cells in structure
        average_cell_confidence: Mean confidence
        
        # Timing
        column_detection_time_ms: Time for column analysis
        row_detection_time_ms: Time for row analysis
        cell_creation_time_ms: Time for cell grouping
        structure_building_time_ms: Time for TableStructure creation
    """
    config: ContentTableDetectorConfig
    processing_start: datetime
    processing_end: datetime
    image_dimensions: Tuple[int, int]
    
    tables_found: int = 0
    cells_found: int = 0
    text_regions_analyzed: int = 0
    
    potential_columns_found: int = 0
    columns_after_filtering: int = 0
    potential_rows_found: int = 0
    rows_after_filtering: int = 0
    
    cells_created: int = 0
    cells_with_text: int = 0
    cells_empty: int = 0
    average_cell_confidence: float = 0.0
    
    column_detection_time_ms: float = 0.0
    row_detection_time_ms: float = 0.0
    cell_creation_time_ms: float = 0.0
    structure_building_time_ms: float = 0.0
    
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
                "detect_columns": self.config.detect_columns,
                "detect_rows": self.config.detect_rows,
                "min_cells_for_table": self.config.min_cells_for_table,
            },
            "results": {
                "tables": self.tables_found,
                "cells": self.cells_found,
                "text_regions": self.text_regions_analyzed,
            },
            "column_row_analysis": {
                "potential_columns": self.potential_columns_found,
                "columns_final": self.columns_after_filtering,
                "potential_rows": self.potential_rows_found,
                "rows_final": self.rows_after_filtering,
            },
            "cell_creation": {
                "cells_created": self.cells_created,
                "with_text": self.cells_with_text,
                "empty": self.cells_empty,
                "average_confidence": self.average_cell_confidence,
            },
            "timing_ms": {
                "total": self.total_processing_time_ms,
                "columns": self.column_detection_time_ms,
                "rows": self.row_detection_time_ms,
                "cells": self.cell_creation_time_ms,
                "structure": self.structure_building_time_ms,
            },
        }


class ContentTableDetector:
    """
    Detector for table-based elements using content-based (spatial) detection.
    
    Strategy:
    - Analyzes spatial distribution of OCR results
    - Detects column boundaries via x-alignment of text
    - Detects row boundaries via y-alignment of text
    - Groups text into cells via proximity
    - Builds TableStructure from inferred grid
    - Returns StructuralElement objects with TableStructure content
    
    Characteristics:
    - Works on all documents (including scanned PDFs)
    - More robust to rotation/skew
    - Less precise than line-based detection
    - Suitable as fallback when line-based fails
    
    Typical Usage (as fallback):
        line_detector = TableDetector(line_config)
        elements, trace = line_detector.detect_tables(image)
        
        if not elements:
            # Fallback to content-based
            content_detector = ContentTableDetector(content_config)
            elements, trace = content_detector.detect_tables(image, ocr_results)
    
    Usage (standalone):
        config = ContentTableDetectorConfig(detect_tables=True)
        detector = ContentTableDetector(config)
        
        elements, trace = detector.detect_tables(
            image=image,
            page_number=1,
            ocr_results=ocr_results
        )
    """
    
    def __init__(self, config: Optional[ContentTableDetectorConfig] = None):
        """
        Initialize content-based table detector with configuration.
        
        Args:
            config: ContentTableDetectorConfig. Uses defaults if None.
        """
        self.config = config or ContentTableDetectorConfig()
    
    def _detect_column_boundaries(self, ocr_results: List[OCRTextResult]) -> List[float]:
        """
        Detect column boundaries via x-coordinate alignment analysis.
        
        Algorithm:
        1. Extract left and right edges of all text regions
        2. Group similar x-coordinates (left edges = column starts)
        3. Filter groups by count (must have multiple texts aligned)
        4. Return sorted x-coordinates of column boundaries
        
        Args:
            ocr_results: List of OCR results
        
        Returns:
            Sorted list of x-coordinates marking column boundaries
        """
        if not ocr_results:
            return []
        
        # Collect x-coordinates from text bounding boxes
        x_starts = []  # Left edges of text
        x_ends = []    # Right edges of text
        
        for result in ocr_results:
            x_starts.append(result.bbox.x_min)
            x_ends.append(result.bbox.x_max)
        
        # Cluster x-coordinates to find column boundaries
        all_x = sorted(set(x_starts + x_ends))
        
        columns = []
        current_cluster = [all_x[0]]
        
        for x in all_x[1:]:
            if x - current_cluster[-1] <= self.config.column_alignment_threshold:
                current_cluster.append(x)
            else:
                # Finalize cluster
                avg_x = np.mean(current_cluster)
                columns.append(avg_x)
                current_cluster = [x]
        
        if current_cluster:
            columns.append(np.mean(current_cluster))
        
        return sorted(columns)
    
    def _detect_row_boundaries(self, ocr_results: List[OCRTextResult]) -> List[float]:
        """
        Detect row boundaries via y-coordinate alignment analysis.
        
        Algorithm:
        1. Extract top and bottom edges of all text regions
        2. Group similar y-coordinates (top edges = row starts)
        3. Filter groups by count (must have multiple texts aligned)
        4. Return sorted y-coordinates of row boundaries
        
        Args:
            ocr_results: List of OCR results
        
        Returns:
            Sorted list of y-coordinates marking row boundaries
        """
        if not ocr_results:
            return []
        
        # Collect y-coordinates from text bounding boxes
        y_tops = []    # Top edges of text
        y_bottoms = [] # Bottom edges of text
        
        for result in ocr_results:
            y_tops.append(result.bbox.y_min)
            y_bottoms.append(result.bbox.y_max)
        
        # Cluster y-coordinates to find row boundaries
        all_y = sorted(set(y_tops + y_bottoms))
        
        rows = []
        current_cluster = [all_y[0]]
        
        for y in all_y[1:]:
            if y - current_cluster[-1] <= self.config.row_alignment_threshold:
                current_cluster.append(y)
            else:
                # Finalize cluster
                avg_y = np.mean(current_cluster)
                rows.append(avg_y)
                current_cluster = [y]
        
        if current_cluster:
            rows.append(np.mean(current_cluster))
        
        return sorted(rows)
    
    def _filter_boundaries(self, coords: List[float], min_gap: float) -> List[float]:
        """
        Filter boundaries to remove redundant/overlapping ones.
        
        Args:
            coords: List of coordinates (x or y)
            min_gap: Minimum gap between consecutive boundaries
        
        Returns:
            Filtered list of non-overlapping boundaries
        """
        if not coords:
            return []
        
        filtered = [coords[0]]
        
        for coord in coords[1:]:
            if coord - filtered[-1] >= min_gap:
                filtered.append(coord)
        
        return filtered
    
    def _assign_cells(
        self,
        ocr_results: List[OCRTextResult],
        x_boundaries: List[float],
        y_boundaries: List[float],
    ) -> List[TableCell]:
        """
        Assign OCR results to cells in inferred grid.
        
        Algorithm:
        1. For each OCR result, find overlapping cell(s)
        2. Assign to cell with maximum overlap
        3. Create TableCell for each grid position
        4. Aggregate text within each cell
        
        Args:
            ocr_results: List of OCR results
            x_boundaries: Column x-coordinates
            y_boundaries: Row y-coordinates
        
        Returns:
            List of TableCell objects
        """
        if not x_boundaries or not y_boundaries:
            return []
        
        # Create empty grid
        num_rows = len(y_boundaries) - 1
        num_cols = len(x_boundaries) - 1
        
        grid = {}  # (row, col) -> list of OCR results
        for row in range(num_rows):
            for col in range(num_cols):
                grid[(row, col)] = []
        
        # Assign OCR results to cells
        for result in ocr_results:
            # Find which cell(s) this result overlaps
            overlapping_cells = []
            
            for row in range(num_rows):
                for col in range(num_cols):
                    x_min = x_boundaries[col]
                    x_max = x_boundaries[col + 1]
                    y_min = y_boundaries[row]
                    y_max = y_boundaries[row + 1]
                    
                    # Check overlap
                    if (result.bbox.x_min < x_max and result.bbox.x_max > x_min and
                        result.bbox.y_min < y_max and result.bbox.y_max > y_min):
                        overlapping_cells.append((row, col))
            
            # Assign to cell with center closest to result center
            if overlapping_cells:
                result_center_x = (result.bbox.x_min + result.bbox.x_max) / 2
                result_center_y = (result.bbox.y_min + result.bbox.y_max) / 2
                
                best_cell = min(
                    overlapping_cells,
                    key=lambda rc: self._distance_to_cell_center(
                        result_center_x, result_center_y,
                        x_boundaries[rc[1]], y_boundaries[rc[0]],
                        x_boundaries[rc[1] + 1], y_boundaries[rc[0] + 1]
                    )
                )
                grid[best_cell].append(result)
        
        # Create TableCell objects
        cells = []
        for row in range(num_rows):
            for col in range(num_cols):
                results_in_cell = grid[(row, col)]
                
                # Combine text from all results in cell
                content = " ".join(r.text for r in results_in_cell)
                
                # Compute cell bounds
                x_min = x_boundaries[col]
                x_max = x_boundaries[col + 1]
                y_min = y_boundaries[row]
                y_max = y_boundaries[row + 1]
                
                # Compute confidence
                if results_in_cell:
                    avg_conf = sum(r.confidence for r in results_in_cell) / len(results_in_cell)
                    avg_conf = avg_conf / 100.0  # Normalize Tesseract 0-100 to 0-1
                else:
                    avg_conf = 0.0
                
                cell = TableCell(
                    content=content,
                    row_index=row,
                    col_index=col,
                    bbox=BoundingBox(x_min, y_min, x_max, y_max),
                    confidence=avg_conf,
                    colspan=1,
                    rowspan=1,
                    is_header=(row == 0),
                )
                
                cells.append(cell)
        
        return cells
    
    def _distance_to_cell_center(self, x: float, y: float, x_min: float, y_min: float, x_max: float, y_max: float) -> float:
        """Compute distance from point to cell center."""
        cell_center_x = (x_min + x_max) / 2
        cell_center_y = (y_min + y_max) / 2
        return np.sqrt((x - cell_center_x)**2 + (y - cell_center_y)**2)
    
    def detect_tables(
        self,
        image: np.ndarray,
        page_number: int = 1,
        ocr_results: Optional[List[OCRTextResult]] = None,
        image_path: Optional[str] = None,
    ) -> Tuple[List[StructuralElement], ContentTableDetectionTrace]:
        """
        Detect table structures via content-based spatial analysis.
        
        Args:
            image: Image as numpy array (used for dimensions only)
            page_number: Page number for this image
            ocr_results: OCR results from document (required)
            image_path: Path to image file (for reference/debugging)
        
        Returns:
            (elements, trace) where:
            - elements: List of StructuralElement objects with TableStructure content
            - trace: ContentTableDetectionTrace with detailed results
        """
        start_time = datetime.now()
        elements = []
        
        if not self.config.detect_tables or not ocr_results:
            trace = ContentTableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                text_regions_analyzed=len(ocr_results) if ocr_results else 0,
            )
            return [], trace
        
        # Phase 1: Detect columns
        col_start = datetime.now()
        columns_raw = self._detect_column_boundaries(ocr_results)
        columns = self._filter_boundaries(columns_raw, self.config.column_gap_threshold)
        col_time = (datetime.now() - col_start).total_seconds() * 1000
        
        # Phase 2: Detect rows
        row_start = datetime.now()
        rows_raw = self._detect_row_boundaries(ocr_results)
        rows = self._filter_boundaries(rows_raw, self.config.row_gap_threshold)
        row_time = (datetime.now() - row_start).total_seconds() * 1000
        
        # Validate minimum grid size
        if len(columns) < self.config.min_cols + 1 or len(rows) < self.config.min_rows + 1:
            trace = ContentTableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                text_regions_analyzed=len(ocr_results),
                potential_columns_found=len(columns_raw),
                columns_after_filtering=len(columns),
                potential_rows_found=len(rows_raw),
                rows_after_filtering=len(rows),
                column_detection_time_ms=col_time,
                row_detection_time_ms=row_time,
            )
            logger.debug(f"Grid too small: {len(columns)}x{len(rows)}")
            return [], trace
        
        # Phase 3: Assign cells
        cell_start = datetime.now()
        cells = self._assign_cells(ocr_results, columns, rows)
        cell_time = (datetime.now() - cell_start).total_seconds() * 1000
        
        # Phase 4: Validate table
        if len(cells) < self.config.min_cells_for_table:
            trace = ContentTableDetectionTrace(
                config=self.config,
                processing_start=start_time,
                processing_end=datetime.now(),
                image_dimensions=image.shape[:2],
                text_regions_analyzed=len(ocr_results),
                potential_columns_found=len(columns_raw),
                columns_after_filtering=len(columns),
                potential_rows_found=len(rows_raw),
                rows_after_filtering=len(rows),
                cells_created=len(cells),
                column_detection_time_ms=col_time,
                row_detection_time_ms=row_time,
                cell_creation_time_ms=cell_time,
            )
            logger.debug(f"Too few cells: {len(cells)} < {self.config.min_cells_for_table}")
            return [], trace
        
        # Phase 5: Build TableStructure
        struct_start = datetime.now()
        
        # Compute table statistics
        cells_with_text = sum(1 for c in cells if c.content.strip())
        avg_conf = sum(c.confidence for c in cells) / len(cells) if cells else 0.0
        
        # Compute table bounding box
        x_min = min(columns) if columns else 0
        y_min = min(rows) if rows else 0
        x_max = max(columns) if columns else image.shape[1]
        y_max = max(rows) if rows else image.shape[0]
        table_bbox = BoundingBox(x_min, y_min, x_max, y_max)
        
        # Create TableStructure
        table_struct = TableStructure(
            cells=cells,
            bbox=table_bbox,
            confidence=avg_conf,
            num_rows=len(rows) - 1,
            num_cols=len(columns) - 1,
            table_type="content_inferred",
            has_irregular_structure=False,
        )
        
        # Create StructuralElement
        element = StructuralElement(
            element_id=f"table_inferred_{len(cells)}_cells",
            element_type=ElementType.TABLE,
            content=table_struct,
            bbox=table_bbox,
            confidence=avg_conf,
            page_number=page_number,
            nesting_level=0,
            metadata={
                "detection_method": "content_based_spatial",
                "num_cells": len(cells),
                "num_rows": table_struct.num_rows,
                "num_cols": table_struct.num_cols,
                "cells_with_text": cells_with_text,
                "column_count": len(columns),
                "row_count": len(rows),
            },
            processing_method="table_detector_content_spatial",
            source_ocr_results=[],
        )
        
        elements.append(element)
        struct_time = (datetime.now() - struct_start).total_seconds() * 1000
        
        # Create trace
        end_time = datetime.now()
        trace = ContentTableDetectionTrace(
            config=self.config,
            processing_start=start_time,
            processing_end=end_time,
            image_dimensions=image.shape[:2],
            tables_found=1,
            cells_found=len(cells),
            text_regions_analyzed=len(ocr_results),
            potential_columns_found=len(columns_raw),
            columns_after_filtering=len(columns),
            potential_rows_found=len(rows_raw),
            rows_after_filtering=len(rows),
            cells_created=len(cells),
            cells_with_text=cells_with_text,
            cells_empty=len(cells) - cells_with_text,
            average_cell_confidence=avg_conf,
            column_detection_time_ms=col_time,
            row_detection_time_ms=row_time,
            cell_creation_time_ms=cell_time,
            structure_building_time_ms=struct_time,
        )
        
        logger.info(f"Content-based table detection complete: {table_struct.num_rows}x{table_struct.num_cols} table, {len(cells)} cells, trace: {trace.to_dict()}")
        
        return elements, trace
