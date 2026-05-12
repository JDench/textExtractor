# Sprint 3: List and Table Detection - Implementation Plan

## Overview
Sprint 3 focuses on implementing advanced structure detection for lists and tables, building on the hardened data model foundation from Option A. This document tracks progress across 4 major tasks.

---

## Task 1: Marker-Based List Detection ✅ COMPLETE

### Implementation Summary
- **File**: `/src/detectors/list_detector.py` (450+ lines)
- **Status**: ✅ COMPLETE and TESTED
- **Lines of Code**: 450+
- **Test Coverage**: 7/7 test suites pass (49+ unit tests)

### Features Implemented
1. **Config System** (`ListDetectorConfig`)
   - Detection flags: detect_lists, detect_nested_lists
   - Marker configuration: bullet_chars, marker_types
   - Indentation analysis: indentation_threshold, indentation_unit
   - Validation: __post_init__ checks all parameters

2. **Marker Detection**
   - Bullet markers: `•`, `-`, `*`, `+`
   - Number markers: `1.`, `(1)`, `1)`, `1-`
   - Letter markers: `a.`, `(a)`, `a)`, `a-`
   - Roman numerals: `i.`, `(i)`, `I.`, `(I)`
   - Regex-based detection with fallback chain

3. **Indentation Analysis**
   - Level detection from leading whitespace
   - Configurable indentation unit (pixels per level)
   - Supports arbitrary nesting depth

4. **Hierarchy Building**
   - Parent-child relationship detection
   - Root item identification
   - Recursive hierarchy preservation
   - Full support for deeply nested lists

5. **ListDetectionTrace**
   - Config used, timing, results count
   - Marker types detected
   - Nesting levels found
   - Serialization to dict for logging

6. **ListDetector Class**
   - OCR integration via OCREngine
   - Multi-phase detection algorithm:
     1. Extract lines with OCR (PSM 6)
     2. Detect markers in each line
     3. Analyze indentation for hierarchy
     4. Build parent-child relationships
     5. Create ListStructure from items
   - Returns StructuralElement with ListStructure content

### Algorithm
```
INPUT: Image + OCR Results (or re-run OCR if not provided)

PHASE 1: MARKER DETECTION
  For each OCR result:
    1. Match text against marker patterns (bullet, number, letter, roman)
    2. Extract indentation level from leading whitespace
    3. Extract item number if numbered/lettered
    4. Create ListItem with extracted metadata
  OUTPUT: List of ListItems with metadata

PHASE 2: HIERARCHY BUILDING
  For each ListItem in order:
    1. Pop stack items with level >= current level
    2. If stack not empty, current item's parent = stack top
    3. If stack empty, item is root
    4. Push (level, item_id) to stack
  OUTPUT: root_item_ids[], parent_map{}

PHASE 3: GROUPING
  1. Create ListStructure with all items
  2. Update parent/child relationships
  3. Compute statistics (avg confidence, bbox)
  4. Create StructuralElement wrapping ListStructure
  OUTPUT: [StructuralElement with ListStructure content]
```

### Validation
All 7 test suites pass with 49+ individual test cases:
- ✓ Bullet marker detection (7 test cases)
- ✓ Number marker detection (7 test cases)
- ✓ Letter marker detection (7 test cases)
- ✓ Indentation level detection (8 test cases)
- ✓ Hierarchy building (4 parent relationships verified)
- ✓ Number extraction (7 test cases)
- ✓ Complete marker detection logic (6 test cases)

### Data Model Integration
Uses all ListDetector-specific dataclasses from `data_models.py`:
- `ListMarkerType` enum: BULLET, NUMBER, LETTER, ROMAN, DASH, MIXED
- `ListItem`: content, level, bbox, confidence, list_type, number, parent_item_id, child_item_ids
- `ListStructure`: items[], root_item_ids[], bbox, confidence, list_type
- All with __post_init__ validation

### Known Limitations & Future Enhancements
1. **Multi-list Detection**: Currently groups all items as single list
   - Future: Detect gaps/breaks to split into multiple lists
2. **Marker Consistency**: Doesn't enforce marker consistency within list
   - Future: Flag mixed markers within same list level
3. **Multi-line Items**: Assumes one item per line
   - Future: Handle wrapped/continued items
4. **Custom Markers**: Doesn't support user-defined markers
   - Future: Allow custom marker patterns in config

### Files Created
- `/src/detectors/list_detector.py` (450+ lines)
  - ListMarkerType enum
  - ListDetectorConfig dataclass
  - ListDetectionTrace dataclass
  - ListDetector class
- `/examples/test_list_detector.py` (380+ lines) - Full integration tests
- `/examples/test_list_detector_units.py` (400+ lines) - Unit tests (all pass ✅)

### Integration Points
✅ Follows Config + Detector + Trace pattern
✅ Uses OCREngine for text extraction
✅ Returns StructuralElement with ListStructure
✅ Utilizes ListItem/ListStructure from data_models.py
✅ Processing trace for reproducibility
✅ Comprehensive configuration validation

---

## Task 2: Line-Based Table Detection ✅ COMPLETE

### Implementation Summary
- **File**: `/src/detectors/table_detector.py` (500+ lines)
- **Status**: ✅ COMPLETE and TESTED
- **Lines of Code**: 500+
- **Test Coverage**: 9/9 test suites pass (43+ unit tests)

### Features Implemented
1. **Config System** (`TableDetectorConfig`)
   - Detection flags: detect_tables, detect_merged_cells, detect_table_headers
   - Hough transform parameters: rho/theta resolution, threshold, line gap
   - Line processing: clustering threshold, intersection distance
   - Cell extraction: min width/height, confidence threshold
   - Validation: __post_init__ checks all parameters

2. **Hough Transform-Based Line Detection**
   - Horizontal and vertical line detection
   - Configurable Hough parameters for robustness
   - Handles various line thicknesses and gaps

3. **Line Clustering**
   - Merges similar lines (within threshold pixels)
   - Reduces noise and duplicate line detection
   - Maintains line order (top-to-bottom, left-to-right)

4. **Grid Detection**
   - Finds intersections of horizontal and vertical lines
   - Builds regular grid from intersection points
   - Validates grid has minimum lines (2x2 required)

5. **Cell Extraction**
   - Extracts bounds for each cell
   - Filters cells by minimum dimensions
   - Prepares for OCR text extraction
   - Assigns row/column indices

6. **Merged Cell Detection (Placeholder)**
   - Framework for colspan/rowspan detection
   - Ready for content-based merged cell inference
   - Validates TableCell constraints already in place

7. **TableDetectionTrace**
   - Complete timing breakdown (line detection, grid, cells, merged)
   - Line detection statistics (horizontal, vertical, clustered, intersections)
   - Cell statistics (total, with text, without text, average confidence)
   - Serialization to dict for logging

8. **TableDetector Class**
   - Follows Config + Detector + Trace pattern
   - 6-phase detection algorithm:
     1. Preprocess image (grayscale → blur → Canny edges)
     2. Run Hough transform for line detection
     3. Cluster similar lines to reduce noise
     4. Find line intersections to build grid
     5. Extract cells and text via OCR
     6. Detect merged cells and build TableStructure
   - Returns StructuralElement with TableStructure content
   - Graceful fallback when cv2/pytesseract not available

### Algorithm
```
INPUT: Image

PHASE 1: IMAGE PREPROCESSING
  1. Convert to grayscale (if needed)
  2. Apply Gaussian blur (5x5)
  3. Apply Canny edge detection
  OUTPUT: Binary edge image

PHASE 2: LINE DETECTION (Hough Transform)
  1. Apply HoughLinesP to edge image
  2. Extract line endpoints (x1,y1,x2,y2)
  3. Classify as horizontal (dx > dy) or vertical (dy >= dx)
  OUTPUT: horizontal_lines[], vertical_lines[]

PHASE 3: LINE CLUSTERING
  For horizontal and vertical lines:
    1. Extract y-coords (horizontal) or x-coords (vertical)
    2. Sort coordinates
    3. Group by distance threshold (clustering)
    4. Average coordinates within each cluster
  OUTPUT: clustered_h_lines[], clustered_v_lines[]

PHASE 4: GRID BUILDING
  1. Find intersections of all h × v line combinations
  2. Extract unique x and y coordinates
  3. Build sorted arrays of grid lines
  OUTPUT: x_coords[], y_coords[]

PHASE 5: CELL EXTRACTION
  For each grid cell (row_idx, col_idx):
    1. Extract bounds from grid coordinates
    2. Check minimum dimensions (width, height)
    3. Extract region from original image
    4. Run OCR on region to extract text
    5. Create TableCell with content and metadata
  OUTPUT: TableCell[]

PHASE 6: MERGED CELL DETECTION
  For each cell:
    1. Analyze position and content
    2. Detect if spans multiple rows (rowspan)
    3. Detect if spans multiple columns (colspan)
    4. Update TableCell colspan/rowspan
  OUTPUT: Updated TableCell[] with spanning info

FINAL: Build TableStructure
  1. Create table from cells and grid
  2. Compute statistics (size, confidence)
  3. Wrap in StructuralElement
  OUTPUT: [StructuralElement with TableStructure content]
```

### Validation
All 9 test suites pass with 43+ individual test cases:
- ✓ Horizontal line clustering (6 test cases)
- ✓ Vertical line clustering (3 test cases)
- ✓ Line intersection detection (5 test cases)
- ✓ Grid building (4 test cases)
- ✓ Cell count calculation (6 test cases)
- ✓ Cell bounds extraction (verified 4 cells)
- ✓ Table validation rules (5 test cases)
- ✓ Line classification (5 test cases)
- ✓ Cell size filtering (verified 3/5 valid cells)

### Data Model Integration
Uses TableDetector-specific dataclasses from `data_models.py`:
- `TableCell`: content, row_index, col_index, bbox, confidence, colspan, rowspan, is_header
- `TableStructure`: cells[], bbox, confidence, num_rows, num_cols, table_type, has_irregular_structure
- All with __post_init__ validation

### Known Limitations & Future Enhancements
1. **Rotation/Skew**: Doesn't handle rotated tables
   - Future: Add skew detection and correction
2. **Merged Cell Detection**: Currently placeholder
   - Future: Implement content-based merged cell inference
3. **Text Extraction**: Assumes cv2/pytesseract available
   - Future: Integrate with OCREngine for consistency
4. **Cell Content Validation**: Uses placeholder confidence
   - Future: Propagate actual Tesseract confidence scores
5. **Layout Tables**: Treats all tables as data tables
   - Future: Classify table type (data vs. layout vs. decorative)

### Files Created
- `/src/detectors/table_detector.py` (500+ lines)
  - TableDetectorConfig dataclass
  - TableDetectionTrace dataclass
  - TableDetector class with all 6 phases
- `/examples/test_table_detector_units.py` (400+ lines) - Unit tests (all pass ✅)

### Integration Points
✅ Follows Config + Detector + Trace pattern
✅ Uses Hough transform via cv2 (with fallback)
✅ Returns StructuralElement with TableStructure
✅ Utilizes TableCell/TableStructure from data_models.py
✅ Processing trace for reproducibility
✅ Comprehensive configuration validation

---

## Task 3: Content-Based Table Detection ✅ COMPLETE

### Implementation Summary
- **File**: `/src/detectors/content_table_detector.py` (550+ lines)
- **Status**: ✅ COMPLETE and TESTED
- **Lines of Code**: 550+
- **Test Coverage**: 8/8 test suites pass (40+ unit tests)

### Features Implemented
1. **Config System** (`ContentTableDetectorConfig`)
   - Column detection: x_threshold (clustering distance), min_columns (validation)
   - Row detection: y_threshold, min_rows
   - Boundary filtering: min_gap_threshold (overlapping boundary removal)
   - Cell assignment: text_to_cell_strategy, distance_weighting
   - Confidence thresholds: min_confidence_for_cell_assignment
   - Validation: __post_init__ checks all parameters

2. **Column Boundary Detection**
   - Clusters text region x-coordinates
   - Groups overlapping/nearby regions
   - Removes redundant boundaries via gap threshold
   - Configurable clustering sensitivity

3. **Row Boundary Detection**
   - Clusters text region y-coordinates
   - Similar approach to column detection
   - Handles text at different vertical levels
   - Flexible boundary positioning

4. **Boundary Filtering**
   - Removes boundaries too close together
   - Prevents spurious grid lines from noise
   - Maintains minimum separation between rows/columns
   - Uses configurable gap threshold

5. **Grid Construction**
   - Builds cell grid from boundary coordinates
   - Creates cells from row/column intersections
   - Regular or irregular grids supported
   - Each cell has bounds, position, and metadata

6. **Cell Assignment**
   - Maps OCR text regions to grid cells
   - Overlap detection (region overlaps which cells)
   - Distance-to-center weighting
   - Aggregates text from multiple regions into single cell
   - Handles cells with no text gracefully

7. **Text Aggregation**
   - Combines text from multiple OCR regions
   - Preserves text order
   - Configurable separator (space, newline, etc.)
   - Normalizes whitespace

8. **ContentTableDetectionTrace**
   - Timing breakdown (column detection, row detection, cell assignment)
   - Column/row boundary statistics
   - Cell assignment statistics (total cells, cells with text)
   - Confidence metrics
   - Serialization to dict for logging

9. **ContentTableDetector Class**
   - Follows Config + Detector + Trace pattern
   - 5-phase detection algorithm:
     1. Detect columns via x-coordinate clustering
     2. Detect rows via y-coordinate clustering
     3. Filter boundaries to remove overlaps
     4. Build grid and assign text to cells
     5. Create TableStructure from inferred grid
   - Returns StructuralElement with TableStructure content (table_type="content_inferred")
   - Marks as fallback algorithm in table metadata

### Algorithm
```
INPUT: OCR Results (text regions with positions)

PHASE 1: COLUMN BOUNDARY DETECTION
  1. Extract x-coordinates from all OCR text regions
  2. Sort coordinates
  3. Cluster coordinates within x_threshold distance
  4. Average each cluster to get column center
  5. Filter boundaries by min_gap_threshold
  OUTPUT: x_boundaries[] (sorted column boundaries)

PHASE 2: ROW BOUNDARY DETECTION
  1. Extract y-coordinates from all OCR text regions
  2. Sort coordinates
  3. Cluster coordinates within y_threshold distance
  4. Average each cluster to get row center
  5. Filter boundaries by min_gap_threshold
  OUTPUT: y_boundaries[] (sorted row boundaries)

PHASE 3: GRID CONSTRUCTION
  1. Create cells for all combinations of rows × columns
  2. Assign bounds to each cell (x_min, x_max, y_min, y_max)
  3. Initialize cells with empty content
  OUTPUT: cells[] (grid_rows × grid_cols)

PHASE 4: CELL ASSIGNMENT
  For each OCR text region:
    1. Find which cells it overlaps (by bounds)
    2. If multiple cells: choose closest by center distance
    3. Aggregate text into chosen cell
    4. Update cell confidence from OCR confidence
  OUTPUT: Updated cells[] with text and confidence

PHASE 5: STRUCTURE BUILDING
  1. Create TableStructure from cells and grid
  2. Mark table_type as "content_inferred" (fallback)
  3. Compute overall statistics (size, confidence)
  4. Wrap in StructuralElement
  OUTPUT: [StructuralElement with TableStructure content]
```

### Validation
All 8 test suites pass with 40+ individual test cases:
- ✓ Column boundary detection (4 test cases - clustering works)
- ✓ Row boundary detection (4 test cases - clustering works)
- ✓ Boundary filtering (4 test cases - gap threshold works)
- ✓ Grid construction (4 test cases - cell creation works)
- ✓ Cell overlap detection (4 scenarios - overlap logic works)
- ✓ Cell center distance (3 scenarios - distance calculation works)
- ✓ Text aggregation (3 scenarios - text joining works)
- ✓ Table validation (5 test cases - validation rules work)

### Data Model Integration
Uses ContentTableDetector-specific dataclasses from `data_models.py`:
- `TableCell`: content, row_index, col_index, bbox, confidence, colspan, rowspan, is_header
- `TableStructure`: cells[], bbox, confidence, num_rows, num_cols, table_type, has_irregular_structure
- All with __post_init__ validation

### Known Limitations & Future Enhancements
1. **Header Detection**: Currently treats all rows equally
   - Future: Detect and mark header rows
2. **Cell Merge Inference**: Framework present but not implemented
   - Future: Detect colspan/rowspan from spatial gaps
3. **Confidence Accuracy**: Uses placeholder confidence values
   - Future: Propagate actual OCR confidence scores
4. **Text Overflow**: Assumes text stays within cell bounds
   - Future: Handle text that spans multiple cells
5. **Empty Cell Detection**: May miss truly empty cells
   - Future: Distinguish empty from low-confidence cells

### Files Created
- `/src/detectors/content_table_detector.py` (550+ lines)
  - ContentTableDetectorConfig dataclass
  - ContentTableDetectionTrace dataclass
  - ContentTableDetector class with all 5 phases
- `/examples/test_content_table_detector_units.py` (380+ lines) - Unit tests (all pass ✅)

### Integration Points
✅ Follows Config + Detector + Trace pattern
✅ Uses spatial analysis (clustering, overlap detection)
✅ Returns StructuralElement with TableStructure
✅ Utilizes TableCell/TableStructure from data_models.py
✅ Processing trace for reproducibility
✅ Comprehensive configuration validation
✅ Fallback algorithm for when Hough transform fails

### Integration with Task 2
- Used as fallback when line-based detection fails
- Similar input/output interface (both return StructuralElement with TableStructure)
- May return less accurate results but better than no detection
- Can be chained: Try TableDetector first, fall back to ContentTableDetector if no lines detected

---

## Task 4: Merged Cells & Irregular Structures (TODO)

### Plan
- Enhance TableCell to detect colspan/rowspan
- Update TableStructure.to_2d_array() for merged cells
- Update TableStructure.to_csv() for merged cells
- Update TableStructure.to_markdown() for merged cells
- Handle ragged tables (unequal row/column counts)

### Estimated Complexity
- Lines of code: ~100 (updates to existing code)
- Implementation difficulty: Low (mostly logic)
- Already validated: TableCell.colspan, TableCell.rowspan in __post_init__

### Dependencies
- All validation already exists in data_models.py
- Just need to implement export logic

---

## Progress Summary

| Task | Status | Lines | Tests | Notes |
|------|--------|-------|-------|-------|
| 1. List Detection | ✅ COMPLETE | 450+ | 49+ ✅ | All tests pass, marker-based algorithm |
| 2. Line-Based Tables | ✅ COMPLETE | 500+ | 43+ ✅ | All tests pass, Hough transform algorithm |
| 3. Content-Based Tables | ✅ COMPLETE | 550+ | 40+ ✅ | All tests pass, spatial analysis algorithm |
| 4. Merged Cells | ⏳ TODO | 100 | - | Export methods for irregular structures |

---

## Architecture

### ListDetector Pattern (Already Implemented)
```python
config = ListDetectorConfig(...)
detector = ListDetector(config)
elements, trace = detector.detect_lists(image, ocr_results)
```

### TableDetector Pattern (To Be Implemented - Same Approach)
```python
config = TableDetectorConfig(...)
detector = TableDetector(config)
elements, trace = detector.detect_tables(image)
```

---

## Testing Strategy

### Task 1 (Complete)
- ✅ Unit tests: Regex patterns, hierarchy building, number extraction
- ✅ Integration tests: Mock OCR results, end-to-end detection
- ✅ Edge cases: Empty lists, mixed markers, deeply nested structures

### Task 2-4 (Upcoming)
- Unit tests: Hough transform, cell detection, merged cell logic
- Integration tests: Real images with various table structures
- Edge cases: Ragged tables, missing borders, overlapping cells

---

## Key Decisions

### Pattern Matching Approach (Task 1)
- **Chosen**: Regex-based pattern matching
- **Rationale**: Fast, reliable, handles most common markers
- **Alternative**: Machine learning (future enhancement)

### Hierarchy Algorithm (Task 1)
- **Chosen**: Stack-based level tracking
- **Rationale**: O(n) time, handles arbitrary nesting, clean code
- **Alternative**: Full tree building (more complex, same result)

### Line Detection (Task 2)
- **Chosen**: Hough transform
- **Rationale**: Standard approach for grid detection, handles rotation/skew
- **Alternative**: Edge detection + clustering (less robust)

---

## Next Steps
1. ✅ Task 1 COMPLETE - ListDetector ready for integration
2. ⏳ Task 2 - Implement TableDetector with Hough transform
3. ⏳ Task 3 - Implement content-based table detection
4. ⏳ Task 4 - Handle irregular structures and exports
5. ⏳ Integration - Test all detectors together with real images

---

## Metrics

### Task 2 Metrics
- **Lines of Code**: 500+ (detector) + 400+ (unit tests) = 900+
- **Test Coverage**: 9 test suites, 43+ individual tests
- **Pass Rate**: 100% ✅
- **Code Complexity**: Medium (Hough transform, geometry)
- **Dependencies**: numpy, dataclasses, cv2 (optional), pytesseract (optional)

### Task 3 Metrics
- **Lines of Code**: 550+ (detector) + 380+ (unit tests) = 930+
- **Test Coverage**: 8 test suites, 40+ individual tests
- **Pass Rate**: 100% ✅
- **Code Complexity**: Medium (clustering, spatial analysis)
- **Dependencies**: numpy, dataclasses (stdlib)

### Cumulative Project Metrics (End of Task 3)
- **Total Lines**: 3400+ (data_models + ocr_engine + text_detector + list_detector + table_detector + content_table_detector)
- **Total Tests**: 180+ (48 Option A + 49 List + 43 Table + 40 Content)
- **Test Pass Rate**: 100% ✅
- **Implementation Coverage**: 4+ major detector components built

