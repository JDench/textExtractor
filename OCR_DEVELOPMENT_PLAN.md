# OCR Text Extraction System - Development Plan

## Project Overview
Build a parameterized Python-based OCR system using Tesseract that extracts text, tables, lists, and structural information from batches of images, outputting structured data with metadata.

---

## Phase 2: Data Models & Output Structure

### 2.1 Core Data Classes (data_models.py)

#### Element Type Enumeration

The system detects and extracts the following structural element types:

**Text-Based Elements:**
- `TEXT` - General body text paragraphs
- `HEADING` - Titles, section headers (with level hierarchy: H1, H2, H3, etc.)

**List Structures:**
- `LIST` - Bullet points, numbered lists, ordered/unordered (with nested structure)

**Tabular Data:**
- `TABLE` - Structured table data with cells, rows, columns, headers, merged cells

**Visual/Layout Elements:**
- `FIGURE` - Images, diagrams, charts, graphs (non-text regions)
- `CAPTION` - Text captions describing figures or tables

**Mathematical/Technical Elements:**
- `FORMULA` - Mathematical expressions (LaTeX, MathML, or rendered in image)
- `EQUATION` - Numbered equations with reference labels and identifiers

**Page Structure:**
- `HEADER` - Page headers with running titles, section names
- `FOOTER` - Page footers with metadata, page numbers
- `PAGE_NUMBER` - Explicit page numbering

**Annotations & Special Content:**
- `ANNOTATION` - Highlighted, underlined, or commented text; handwritten notes
- `WATERMARK` - Detected watermarks or background text overlays
- `BARCODE` - QR codes, barcodes, or other machine-readable markers

**Organizational Elements:**
- `BLOCK_QUOTE` - Quoted or indented text blocks with optional attribution
- `CODE_BLOCK` - Monospaced code, pre-formatted text, or programming snippets

**Reference Elements:**
- `REFERENCE` - Citations, footnotes, endnotes, bibliography entries
- `TABLE_OF_CONTENTS` - Table of contents entries with page numbers
- `INDEX` - Index entries or cross-references with target pages

#### Key Data Classes

**OCRTextResult:**
- Extracted text content from Tesseract
- Confidence scores (0-100, to be normalized)
- Detected language
- Raw coordinates of text regions
- Optional font properties (name, size)

**StructuralElement:**
- Type: One of the ElementType values above
- Content (text string, or structured dict for tables/lists/formulas)
- Bounding box coordinates
- Confidence score (0-1)
- Nesting level (for hierarchical structures like lists, nested tables)
- Parent/child relationships for tree structures
- Page number and location metadata
- Processing method metadata (which detector found this element)

**DocumentMetadata:**
- Source file path
- Processing timestamp and duration
- Image dimensions
- Preprocessing parameters used
- OCR confidence (average)
- Detected languages
- Processing status (success/failure/partial)
- Quality score assessment
- Error and warning logs

**BatchResult:**
- List of DocumentResult objects (one per image)
- Batch-level statistics (total elements, success rate, timing)
- Configuration used for processing
- Export methods:
  - `to_csv()` - Flat structure, one element per row
  - `to_json()` - Hierarchical structure preserving nesting
  - `to_excel()` - Multiple sheets (elements, documents, statistics, details)
  - `to_dataframe()` - Pandas DataFrame for analysis

---

## Phase 3: Structure Detection Strategies

### 3.1 Text Detection (All Text-Based Elements)

**Heading Detection:**
```
1. Run OCR with PSM mode 11 (sparse text) for better segmentation
2. Identify text size changes and font variations
3. Detect text positioned above body paragraphs
4. Classify hierarchy level based on relative size
5. Return StructuralElement with type=HEADING, level=1-6
```

**Paragraph/Text Extraction:**
```
1. Apply PSM mode 6 (uniform block) for dense text
2. Extract word-level bounding boxes
3. Group words into lines, then paragraphs
4. Calculate confidence as average of word confidences
5. Return StructuralElement with type=TEXT
```

**Block Quote Detection:**
```
1. Identify indented text regions
2. Detect border lines on sides (if present)
3. Extract source/attribution if available
4. Return StructuralElement with type=BLOCK_QUOTE, indentation_level
```

### 3.2 List Detection

**Approach 1: Marker-Based Detection (Primary)**
```
1. Scan for bullet points, dashes, or numbering patterns
2. Identify bullet symbols (●, ○, -, *, etc.) or numbers (1, 2, A, I, etc.)
3. Group consecutive items at same indentation level
4. Detect indentation increases for nested lists
5. Build tree structure with parent-child relationships
6. Return StructuralElement with type=LIST containing ListStructure
```

**Approach 2: Spatial-Based Detection (Fallback)**
```
1. Analyze spatial distribution of text paragraphs
2. Detect regular vertical spacing patterns
3. Identify common left-margin alignment
4. Infer list structure from spacing and alignment
```

**Output:**
- ListItem objects with level, content, marker type
- Flat list stored, but with parent_id/child_id references for tree reconstruction
- StructuralElement wrapping ListStructure

### 3.3 Table Detection

**Approach 1: Line-Based Detection (Bordered Tables)**
```
1. Convert to grayscale
2. Detect horizontal and vertical lines using Hough transform
3. Find intersections to identify grid structure
4. Group intersections into table regions
5. Extract text content for each cell
6. Return TableStructure with cell grid
```

**Approach 2: Content-Based Detection (Borderless/Complex Tables)**
```
1. Analyze spatial distribution of OCR text results
2. Identify regular column/row spacing patterns
3. Detect repeating horizontal text at same Y-coordinate (rows)
4. Detect repeating vertical text at same X-coordinate (columns)
5. Build cell structure based on spacing grid
6. Return TableStructure preserving irregular patterns
```

**Approach 3: ML-Based Detection (Advanced Future)**
- Use pre-trained LayoutLM or DETR model
- Fine-tune on document samples if needed
- Handles complex layouts, merged cells, nested tables

**Output:**
- TableCell objects for each cell (row_index, col_index, content, bbox)
- TableStructure with num_rows, num_cols, headers, caption
- Handle merged cells (colspan, rowspan)
- Flag irregular structures (ragged rows, merged cells)

### 3.4 Figure Detection

**Figure Region Detection:**
```
1. Identify large regions without text (or minimal text)
2. Detect contours of potential figure boundaries
3. Classify figure type (chart, photo, diagram) if possible
4. Extract any embedded text within figure
5. Detect and link associated caption
6. Return FigureRegion with type=FIGURE
```

**Caption Linking:**
```
1. Find text near figures that matches caption pattern
2. Extract figure number (e.g., "Figure 2.3") if present
3. Link caption to figure via referenced_element_id
4. Return Caption element with type=CAPTION
```

### 3.5 Formula Detection (Mathematical Content)

**Formula Detection:**
```
1. Identify text that looks like mathematical notation
2. Detect special characters: =, +, -, ×, ÷, √, ∑, ∫, Greek letters, etc.
3. Extract bounding box of formula region
4. OCR formula using specialized math OCR if available
5. Attempt conversion to LaTeX if possible
6. Return FormulaExpression and wrap in StructuralElement
```

**Equation Number Extraction:**
```
1. Identify numbered equations: "Eq. (2.5)", "(2.5)", etc.
2. Extract equation number and reference label
3. Detect surrounding context text
4. Link to formula
5. Return EquationReference for cross-referencing
```

### 3.6 Header/Footer Detection

**Running Header Detection:**
```
1. Analyze text at top of page (within top 10% of image)
2. Detect repeated content across multiple pages
3. Extract text and determine if includes page number
4. Return PageHeader with repeated=true/false
```

**Running Footer Detection:**
```
1. Analyze text at bottom of page (within bottom 10% of image)
2. Detect repeated content across multiple pages
3. Extract text and determine if includes page number
4. Return PageFooter with repeated=true/false
```

### 3.7 Annotation Detection

**Highlight Detection:**
```
1. Identify colored regions (yellow, pink, etc.) in image
2. Extract text under highlight
3. Optionally OCR any handwritten notes near highlight
4. Return Annotation with type=highlight, color, optional_note
```

**Underline/Strikethrough Detection:**
```
1. Detect line patterns under or through text
2. Extract affected text
3. Return Annotation with appropriate type
```

**Handwritten Annotation:**
```
1. Detect handwritten text regions (texture/stroke analysis)
2. Use handwriting recognition if available
3. Return Annotation with note content
```

### 3.8 Watermark Detection

**Watermark Identification:**
```
1. Detect repeated semi-transparent text patterns
2. Analyze opacity and positioning
3. Extract watermark text if readable
4. Determine if in background (behind content)
5. Estimate tilt angle if diagonal
6. Return Watermark with background flag and opacity estimate
```

### 3.9 Barcode/QR Detection

**Barcode Recognition:**
```
1. Use barcode detection library (pyzbar, etc.)
2. Identify barcode type (QR, Code128, UPC, etc.)
3. Decode value and any associated metadata
4. Store both visual location (bbox) and decoded value
5. Return Barcode element
```

**Batch Tracking Application:**
```
- Track which documents have barcodes
- Use decoded values for batch identification/sorting
- Link barcode metadata to document processing results
```

### 3.10 Code Block Detection

**Code Identification:**
```
1. Detect monospaced font usage
2. Identify code block styling (indentation, border)
3. Detect programming language keywords (if determinable)
4. Extract line numbers if present
5. Return CodeBlock with language, line_number info
```

### 3.11 Reference/Citation Detection

**Citation Extraction:**
```
1. Identify citation patterns: [1], (Smith 2020), etc.
2. Match known citation formats (APA, MLA, Chicago)
3. Extract reference metadata (author, year, title)
4. Return Reference with parsed components
```

**Footnote/Endnote Detection:**
```
1. Detect superscript reference markers
2. Locate corresponding footnote/endnote text
3. Link reference to target
4. Return Reference with ref_type=footnote/endnote
```

### 3.12 Table of Contents / Index Detection

**TOC Extraction:**
```
1. Identify TOC page (usually early in document)
2. Extract heading titles and page numbers
3. Detect indentation levels for subsections
4. Return TableOfContents entries with hierarchy
5. Optionally link to actual headings in document
```

**Index Extraction:**
```
1. Identify index section (usually end of document)
2. Extract index terms and referenced page numbers
3. Detect see-also cross-references
4. Return IndexEntry objects with hierarchy
```

---

## Phase 4: Configuration System

### 4.1 Element Detection Parameterization (config.py)

```python
@dataclass
class StructureDetectionConfig:
    # Text elements
    detect_text: bool = True
    detect_headings: bool = True
    heading_levels: List[int] = [1, 2, 3, 4, 5, 6]  # H1-H6
    
    # Lists
    detect_lists: bool = True
    list_detection_method: str = "marker"  # "marker" or "spatial"
    
    # Tables
    detect_tables: bool = True
    table_detection_methods: List[str] = ["line_based", "content_based"]
    
    # Visual elements
    detect_figures: bool = True
    detect_captions: bool = True
    figure_types_to_detect: List[str] = ["photo", "chart", "diagram"]
    
    # Mathematics
    detect_formulas: bool = True
    detect_equations: bool = True
    
    # Page structure
    detect_headers: bool = True
    detect_footers: bool = True
    detect_page_numbers: bool = True
    
    # Annotations
    detect_annotations: bool = True
    annotation_types: List[str] = ["highlight", "underline", "strikethrough", "comment"]
    detect_handwriting: bool = False  # Requires specialized model
    
    # Special elements
    detect_watermarks: bool = True
    detect_barcodes: bool = True
    
    # Organization
    detect_block_quotes: bool = True
    detect_code_blocks: bool = True
    
    # References
    detect_citations: bool = True
    detect_footnotes: bool = True
    detect_references: bool = True
    detect_toc: bool = True
    detect_index: bool = True
    
    # Confidence filtering
    min_confidence: float = 0.3
    confidence_threshold_by_type: Dict[ElementType, float] = field(default_factory=dict)
```

---

## Phase 5: Output Formats

### 5.1 DataFrame Structure (Flattened Export)

```
| source_file | document_id | element_id | element_type | content | confidence | bbox | page | level | parent_id | child_ids | processing_method |
|-------------|------------|-----------|-------------|---------|------------|------|------|-------|-----------|-----------|-------------------|
| img1.jpg    | doc_1      | elem_1    | heading     | "Title" | 0.98       | [...] | 1    | 1     | null      | [2,3]     | tesseract_psm11   |
| img1.jpg    | doc_1      | elem_2    | text        | "Para.." | 0.95       | [...] | 1    | 0     | 1         | []        | tesseract_psm6    |
| img1.jpg    | doc_1      | elem_3    | list        | {json}  | 0.92       | [...] | 1    | 0     | 1         | [3a,3b]   | marker_detector   |
| img1.jpg    | doc_1      | elem_4    | table       | {csv}   | 0.88       | [...] | 2    | 0     | null      | []        | line_detector     |
| img1.jpg    | doc_1      | elem_5    | figure      | null    | 0.85       | [...] | 2    | 0     | null      | []        | contour_detector  |
| img1.jpg    | doc_1      | elem_6    | caption     | "Fig.." | 0.91       | [...] | 2    | 0     | null      | []        | caption_detector  |
| img1.jpg    | doc_1      | elem_7    | formula     | "E=mc²" | 0.87       | [...] | 3    | 0     | null      | []        | formula_detector  |
```

**Additional columns for specialized types:**
- **Tables**: table_rows, table_cols, has_headers, has_merged_cells
- **Lists**: list_type, item_count, max_nesting_level
- **Formulas**: latex_representation, mathml_representation
- **Annotations**: annotation_type, annotation_color
- **References**: reference_type, cited_element_id
- **Equations**: equation_number, equation_label

### 5.2 JSON Hierarchy (Structure-Preserving Export)

```json
{
  "batch_id": "batch_001",
  "created_at": "2026-05-10T12:00:00Z",
  "documents": [
    {
      "document_id": "doc_1",
      "source": "image.jpg",
      "metadata": {
        "processed_at": "2026-05-10T10:30:00Z",
        "dimensions": [1920, 1080],
        "confidence_avg": 0.91,
        "status": "completed"
      },
      "elements": [
        {
          "id": "elem_1",
          "type": "heading",
          "content": "Document Title",
          "level": 1,
          "confidence": 0.98,
          "bbox": [100, 50, 500, 100],
          "children": ["elem_2", "elem_3"]
        },
        {
          "id": "elem_2",
          "type": "text",
          "content": "Introduction paragraph...",
          "level": 0,
          "confidence": 0.95,
          "bbox": [100, 110, 800, 200],
          "parent": "elem_1"
        },
        {
          "id": "elem_3",
          "type": "list",
          "level": 0,
          "confidence": 0.92,
          "bbox": [100, 210, 600, 350],
          "parent": "elem_1",
          "children": ["elem_3a", "elem_3b", "elem_3c"],
          "content": {
            "list_type": "bullet",
            "items": [
              {
                "id": "elem_3a",
                "text": "Item 1",
                "level": 0
              },
              {
                "id": "elem_3b",
                "text": "Item 2",
                "level": 0
              }
            ]
          }
        },
        {
          "id": "elem_4",
          "type": "table",
          "confidence": 0.88,
          "bbox": [100, 360, 800, 500],
          "content": {
            "headers": ["Column A", "Column B", "Column C"],
            "rows": [
              ["Data1", "Data2", "Data3"],
              ["Data4", "Data5", "Data6"]
            ],
            "num_rows": 3,
            "num_cols": 3
          }
        },
        {
          "id": "elem_5",
          "type": "figure",
          "confidence": 0.85,
          "bbox": [100, 510, 800, 750],
          "content": {
            "figure_type": "chart",
            "extracted_text": "Chart showing trends...",
            "caption_id": "elem_6"
          }
        },
        {
          "id": "elem_6",
          "type": "caption",
          "content": "Figure 1: Key performance metrics over time",
          "confidence": 0.91,
          "bbox": [100, 755, 800, 780]
        }
      ]
    }
  ],
  "statistics": {
    "total_documents": 5,
    "successful": 5,
    "failed": 0,
    "total_elements": 247,
    "elements_by_type": {
      "text": 120,
      "heading": 15,
      "list": 20,
      "table": 8,
      "figure": 12,
      "caption": 12,
      "formula": 5,
      "annotation": 30,
      "reference": 25
    },
    "average_confidence": 0.91,
    "total_processing_time": 45.3,
    "languages_detected": ["en", "fr"]
  }
}
```

---

## Phase 6: Batch Processor with Element Types

### 6.1 Complete Processing Pipeline

```python
class BatchProcessor:
    def __init__(self, config: OCRConfig):
        - Initialize all detectors based on config
        - One detector per element type
        - Preprocessing pipeline
    
    def process_image(image_path) -> DocumentResult:
        1. Load and preprocess image
        2. Run Tesseract OCR (multiple PSM modes)
        3. For each enabled element type:
           - Run appropriate detector
           - Extract elements
           - Assign element IDs and relationships
           - Compute confidence
        4. Build element hierarchy (parent/child links)
        5. Aggregate to DocumentResult
        6. Return with metadata
    
    def process_batch(file_list) -> BatchResult:
        1. For each file, call process_image
        2. Aggregate DocumentResult objects
        3. Compute BatchStatistics
        4. Return BatchResult with all elements and stats
```

---

## Phase 7: Implementation Roadmap

### Sprint 1: Foundation (Weeks 1-2)
- [x] Set up project structure
- [x] Install dependencies
- [x] Create comprehensive data models with all element types
- [x] Implement basic OCR engine wrapper

### Sprint 2: Text Detection (Weeks 3-4)
- [x] Implement heading detection
- [x] Implement paragraph/text extraction
- [x] Implement block quote detection
- [x] Test on sample images

### Sprint 3: List & Table Detection (Weeks 5-7)
- [ ] Implement marker-based list detection
- [ ] Implement line-based table detection
- [ ] Implement content-based table detection
- [ ] Handle merged cells and irregular structures

### Sprint 4: Advanced Structural Elements (Weeks 8-10)
- [ ] Implement figure detection and caption linking
- [ ] Implement formula/equation detection
- [ ] Implement header/footer detection
- [ ] Implement annotation detection (highlight, underline)

### Sprint 5: Special Elements (Weeks 11-12)
- [ ] Implement watermark detection
- [ ] Implement barcode/QR detection
- [ ] Implement code block detection
- [ ] Implement reference/citation detection

### Sprint 6: Complex Features (Weeks 13-14)
- [ ] Implement TOC extraction
- [ ] Implement index extraction
- [ ] Implement formula LaTeX conversion
- [ ] Build element hierarchy and relationships

### Sprint 7: Batch Processing & Export (Weeks 15-16)
- [ ] Build batch processor
- [ ] Implement CSV export
- [ ] Implement JSON hierarchical export
- [ ] Implement Excel multi-sheet export

### Sprint 8: Testing & Optimization (Weeks 17-18)
- [ ] Write comprehensive tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] Example notebooks

### Sprint 9: Advanced Features (Weeks 19-20)
- [ ] ML-based structure detection
- [ ] Multi-language support
- [ ] GPU acceleration
- [ ] Database backend option

---

## Success Metrics

1. **Element Detection Coverage**: Successfully detect 10+ element types
2. **Accuracy**: OCR confidence scores > 90% on test set
3. **Structure Detection**: > 85% accuracy on tables/lists/complex structures
4. **Performance**: < 5 seconds per image (preprocessing + all detectors)
5. **Flexibility**: Support parameterized detection per element type
6. **Usability**: Single function call to process batches with all element types
7. **Output Quality**: Data ready for downstream processing (export to CSV, JSON, Excel)
8. **Reproducibility**: All parameters logged for reproducible processing

