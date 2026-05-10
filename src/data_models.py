"""
Data models for OCR text extraction system.

This module defines the complete data hierarchy for representing OCR results,
from raw Tesseract output to structured documents with detected tables, lists,
figures, formulas, and other document elements.

Design Philosophy:
- Layer 1: Foundations (enums, geometry)
- Layer 2: Complex domain structures (tables, lists, formulas, etc.)
- Layer 3: Core OCR models (raw results, detected elements)
- Layer 4: Document-level aggregation
- Layer 5: Batch collection

TODO: All models need full __post_init__ validation and serialization methods.
TODO: Performance considerations for large batches documented at end of file.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
import json
import csv
from pathlib import Path


# ============================================================================
# LAYER 1: FOUNDATIONS - Enums and Geometry
# ============================================================================

class ElementType(Enum):
    """
    Enumeration of all detectable structural elements in documents.
    
    These elements span from simple text paragraphs to complex mathematical
    expressions, with each type implying specific detection and extraction logic.
    
    EXPANSION NOTE: This list now includes "future considerations" as first-class
    elements, allowing systematic detection and extraction of each type.
    """
    
    # Text-based elements
    TEXT = "text"  # General body text paragraph
    HEADING = "heading"  # Titles, section headers (with level hierarchy)
    
    # List structures
    LIST = "list"  # Bullet points, ordered lists (nested structure)
    
    # Tabular data
    TABLE = "table"  # Structured table data (cells, rows, columns)
    
    # Visual/Layout elements
    FIGURE = "figure"  # Images, diagrams, charts (non-text regions)
    CAPTION = "caption"  # Text describing figures or tables
    
    # Mathematical/Technical elements
    FORMULA = "formula"  # Mathematical expressions (LaTeX, MathML, or rendered)
    EQUATION = "equation"  # Numbered equations with identifiers
    
    # Metadata/Document structure
    HEADER = "header"  # Page header (footer content, page numbers)
    FOOTER = "footer"  # Page footer (footer content, page numbers)
    PAGE_NUMBER = "page_number"  # Explicit page numbering
    
    # Annotations & Special content
    ANNOTATION = "annotation"  # Highlighted text, notes, comments
    WATERMARK = "watermark"  # Detected watermarks or background text
    BARCODE = "barcode"  # QR codes, barcodes, or other machine-readable markers
    
    # Organizational elements
    BLOCK_QUOTE = "block_quote"  # Quoted text (indented, styled differently)
    CODE_BLOCK = "code_block"  # Monospaced code or pre-formatted text
    
    # Reference elements
    REFERENCE = "reference"  # Citations, footnotes, endnotes, bibliography entries
    TABLE_OF_CONTENTS = "toc"  # Table of contents entries
    INDEX = "index"  # Index entries or cross-references


class ConfidenceLevel(Enum):
    """Classification of confidence scores for easier filtering."""
    VERY_LOW = (0.0, 0.3)      # 0-30%: Unreliable
    LOW = (0.3, 0.6)           # 30-60%: Questionable
    MEDIUM = (0.6, 0.8)        # 60-80%: Acceptable
    HIGH = (0.8, 0.95)         # 80-95%: Reliable
    VERY_HIGH = (0.95, 1.0)    # 95-100%: Very reliable
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Classify a confidence score (0-1 normalized)."""
        # TODO: Implement classification logic
        pass


class PSMMode(Enum):
    """
    Tesseract Page Segmentation Modes.
    Each mode optimizes for different document layouts.
    """
    ORIENTATION_DETECTION = 0
    OSD_ONLY = 1
    AUTO_OSD = 2
    FULLY_AUTOMATIC = 3  # Default
    SINGLE_COLUMN = 4
    SINGLE_BLOCK = 5
    SINGLE_LINE = 6
    SINGLE_WORD = 7
    CIRCLE_WORDS = 8
    COUNT_DIGITS = 9
    SPARSE_TEXT = 11
    RAW_LINE = 13


class OEMMode(Enum):
    """
    Tesseract OCR Engine Modes.
    Legacy vs. Neural Network tradeoffs.
    """
    LEGACY = 0           # Legacy engine only
    LSTM_NEURAL = 1      # Neural network only
    BOTH = 2              # Both engines (slower but more accurate)
    DEFAULT = 3           # Default


class ProcessingStatus(Enum):
    """Status of processing for a document or batch."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some elements extracted, some failed


# ============================================================================
# LAYER 1 CONTINUED: Geometry Models
# ============================================================================

@dataclass
class Coordinates:
    """
    2D point in image space (pixel coordinates).
    
    TODO: Consider adding coordinate system normalization
          (absolute pixels vs. normalized 0-1)
    """
    x: float
    y: float
    
    def __post_init__(self):
        """Validate coordinates are non-negative."""
        # TODO: Implement validation
        pass


@dataclass
class BoundingBox:
    """
    Rectangular region in image space.
    
    Attributes:
        x_min, y_min: Top-left corner
        x_max, y_max: Bottom-right corner
        confidence: Optional confidence in the box location
    
    TODO: Add methods for:
        - intersection(other_box) -> Optional[BoundingBox]
        - union(other_box) -> BoundingBox
        - area() -> float
        - overlap_percentage(other_box) -> float
        - contains_point(x, y) -> bool
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        # TODO: Ensure x_min < x_max, y_min < y_max
        # TODO: Ensure coordinates are non-negative
        # TODO: Validate confidence is 0-1 if provided
        pass


# ============================================================================
# LAYER 2: Complex Domain Structures
# ============================================================================

@dataclass
class TableCell:
    """
    Single cell in a table.
    
    Attributes:
        content: Text or other content in the cell
        row_index: Row position (0-based)
        col_index: Column position (0-based)
        bbox: Bounding box of the cell
        confidence: Extraction confidence for this cell
        colspan: Cells spanned horizontally
        rowspan: Cells spanned vertically
        is_header: Whether cell is part of header row
        background_color: Optional detected background color (hex)
        text_formatting: Optional detected formatting (bold, italic, etc.)
    
    TODO: Handle merged cells properly
    TODO: Detect and extract background colors (might improve structure)
    TODO: Preserve text styling (bold, italic) if detectable
    TODO: Add cell type classification (header, data, total, etc.)
    """
    content: str
    row_index: int
    col_index: int
    bbox: BoundingBox
    confidence: float
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False
    background_color: Optional[str] = None
    text_formatting: Optional[Dict[str, bool]] = None  # {'bold': True, 'italic': False}
    
    def __post_init__(self):
        # TODO: Validate row/col indices are non-negative
        # TODO: Validate colspan/rowspan >= 1
        # TODO: Validate confidence is 0-1
        pass


@dataclass
class TableStructure:
    """
    Complete table representation with structured access to cells.
    
    Attributes:
        cells: List of all cells in table
        num_rows: Total rows (computed from cells)
        num_cols: Total columns (computed from cells)
        headers: Optional list of header cell contents
        bbox: Bounding box of entire table
        confidence: Average confidence across all cells
        caption: Optional table caption/title
        table_type: Classification (data table, layout table, etc.)
        has_irregular_structure: Flag for merged cells or ragged rows
    
    TODO: Add methods for:
        - get_cell(row, col) -> Optional[TableCell]
        - get_row(row) -> List[TableCell]
        - get_column(col) -> List[TableCell]
        - to_2d_array() -> List[List[str]]
        - to_markdown() -> str
        - to_csv() -> str
        - to_pandas_dataframe() -> DataFrame
    TODO: Handle ragged rows vs. fixed column counts
    TODO: Detect and preserve table structure with merged cells
    TODO: Extract table captions automatically
    """
    cells: List[TableCell]
    bbox: BoundingBox
    confidence: float
    num_rows: int = field(default=0)
    num_cols: int = field(default=0)
    headers: Optional[List[str]] = None
    caption: Optional[str] = None
    table_type: Optional[str] = None
    has_irregular_structure: bool = False
    
    def __post_init__(self):
        # TODO: Compute num_rows and num_cols from cells
        # TODO: Validate all cells reference valid indices
        # TODO: Detect headers automatically if not provided
        pass


@dataclass
class ListItem:
    """
    Single item in a list (bullet point, numbered item, etc.).
    
    Attributes:
        content: Text content of the item
        level: Nesting depth (0=top level, 1=indented once, etc.)
        list_type: Type of marker (bullet, number, letter, etc.)
        number: If numbered/lettered, the item's ordinal
        bbox: Bounding box of item
        confidence: Extraction confidence
        parent_item_id: Reference to parent if nested
        child_item_ids: References to child items
    
    TODO: Detect list markers automatically
    TODO: Preserve multi-line list items
    TODO: Handle mixed list types (bullets + numbers in same list)
    """
    content: str
    level: int
    bbox: BoundingBox
    confidence: float
    list_type: str = "bullet"  # "bullet", "number", "letter", "dash", etc.
    number: Optional[int] = None
    parent_item_id: Optional[str] = None
    child_item_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # TODO: Validate level >= 0
        # TODO: Validate confidence is 0-1
        # TODO: Ensure consistency between list_type and number
        pass


@dataclass
class ListStructure:
    """
    Complete list representation preserving hierarchy.
    
    Attributes:
        items: All list items (flat list)
        root_item_ids: IDs of top-level items (for tree reconstruction)
        bbox: Bounding box of entire list
        confidence: Average confidence across items
        list_type: Type of list (bullet, numbered, mixed)
    
    TODO: Add methods for:
        - build_tree() -> TreeNode
        - to_markdown() -> str
        - get_items_at_level(level) -> List[ListItem]
        - flatten() -> List[str]
    TODO: Preserve indentation hierarchies correctly
    """
    items: List[ListItem]
    root_item_ids: List[str]
    bbox: BoundingBox
    confidence: float
    list_type: str = "mixed"
    
    def __post_init__(self):
        # TODO: Validate all items have parent references that exist
        # TODO: Validate tree structure is acyclic
        pass


@dataclass
class FormulaExpression:
    """
    Mathematical formula or expression.
    
    Attributes:
        raw_text: Raw OCR text from image
        latex: LaTeX representation (if detected/converted)
        mathml: MathML representation (if detected/converted)
        rendered_bbox: Location in image where formula appears
        confidence: Extraction confidence
        is_displaystyle: Whether formula is display-style (centered) vs inline
        variables: Detected variable names mentioned in formula
    
    TODO: Add formula parsing and conversion methods
    TODO: Integrate with math OCR library (if using specialized tool)
    TODO: Extract variable definitions and meanings from surrounding text
    TODO: Validate mathematical syntax
    """
    raw_text: str
    bbox: BoundingBox
    confidence: float
    latex: Optional[str] = None
    mathml: Optional[str] = None
    is_displaystyle: bool = False
    variables: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # TODO: Validate raw_text is not empty
        # TODO: If LaTeX provided, validate it's valid
        # TODO: Parse variables from LaTeX/MathML if provided
        pass


@dataclass
class EquationReference:
    """
    Numbered equation with reference label.
    
    Attributes:
        formula: The FormulaExpression for this equation
        equation_number: How it's numbered (e.g., "2.5", "Eq. 3")
        reference_label: Label used for in-text references (e.g., "bernoulli")
        surrounding_text: Context text before/after equation
    
    TODO: Link equations to their in-text citations
    TODO: Build equation database for cross-referencing
    """
    formula: FormulaExpression
    equation_number: str
    reference_label: Optional[str] = None
    surrounding_text: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate equation_number format
        pass


@dataclass
class Annotation:
    """
    Annotated/highlighted text or special markup.
    
    Attributes:
        content: Text being annotated
        bbox: Location in image
        annotation_type: Type of annotation (highlight, underline, strikethrough, etc.)
        color: Detected color of annotation
        confidence: Extraction confidence
        note: Optional OCR'd note or handwritten text associated with annotation
    
    TODO: Detect handwritten annotations
    TODO: Classify annotation intent (emphasis, correction, note, etc.)
    TODO: Link annotations to related document elements
    """
    content: str
    bbox: BoundingBox
    annotation_type: str  # "highlight", "underline", "strikethrough", "comment", etc.
    confidence: float
    color: Optional[str] = None
    note: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate annotation_type is recognized
        pass


@dataclass
class Barcode:
    """
    Detected barcode, QR code, or machine-readable marker.
    
    Attributes:
        barcode_type: Type of code (QR, Code128, UPC, etc.)
        decoded_value: Decoded content/value
        bbox: Location in image
        confidence: Detection confidence
        raw_image: Optional stored barcode image for re-decoding
        metadata: Any additional metadata from decoding
    
    TODO: Integrate barcode/QR decoding library
    TODO: Validate decoded values against expected formats
    TODO: Track which documents have barcodes for batch tracking
    """
    barcode_type: str
    decoded_value: str
    bbox: BoundingBox
    confidence: float
    raw_image: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # TODO: Validate barcode_type is recognized
        # TODO: Validate decoded_value matches expected format for type
        pass


@dataclass
class Reference:
    """
    Citation, footnote, endnote, or bibliography entry.
    
    Attributes:
        content: Full text of reference
        ref_type: Type of reference (citation, footnote, endnote, bibliography, etc.)
        reference_id: Marker or number in document (e.g., "[1]", "[Smith2020]")
        location: Where reference appears (in-text, footnote, end-of-doc)
        bbox: Location in document
        target_ref: What this references (can be another element's ID)
    
    TODO: Parse citation formats (APA, MLA, Chicago, etc.)
    TODO: Link in-text citations to bibliography entries
    TODO: Extract metadata from citations (author, year, title, etc.)
    TODO: Detect and preserve reference numbering schemes
    """
    content: str
    ref_type: str  # "citation", "footnote", "endnote", "bibliography"
    reference_id: str
    bbox: BoundingBox
    location: str = "in-text"
    target_ref: Optional[str] = None  # ID of element being referenced
    
    def __post_init__(self):
        # TODO: Validate ref_type is recognized
        # TODO: Parse citation if it looks like known format
        pass


@dataclass
class CodeBlock:
    """
    Pre-formatted code or monospaced text block.
    
    Attributes:
        content: Code content
        language: Detected programming language (if determinable)
        bbox: Location in image
        confidence: Extraction confidence
        line_numbers: Whether line numbers are present
        start_line_number: First line number (if numbered)
        syntax_elements: Optional detected syntax elements (keywords, etc.)
    
    TODO: Detect programming language automatically
    TODO: Preserve syntax highlighting information
    TODO: Validate code block structure
    """
    content: str
    bbox: BoundingBox
    confidence: float
    language: Optional[str] = None
    line_numbers: bool = False
    start_line_number: Optional[int] = None
    syntax_elements: Optional[Dict[str, List[str]]] = None
    
    def __post_init__(self):
        # TODO: Validate content is not empty
        pass


@dataclass
class PageHeader:
    """
    Content appearing at top of page (running header).
    
    Attributes:
        content: Header text
        bbox: Location in image
        page_number: Which page this appears on
        includes_page_number: Whether header itself contains page number
        repeated: Whether this header appears on multiple pages
    
    TODO: Detect repeated headers across pages
    TODO: Extract page numbers from headers
    TODO: Classify header type (title, section, document name, etc.)
    """
    content: str
    bbox: BoundingBox
    page_number: int
    confidence: float
    includes_page_number: bool = False
    repeated: bool = False
    
    def __post_init__(self):
        # TODO: Validate page_number >= 1
        pass


@dataclass
class PageFooter:
    """
    Content appearing at bottom of page (running footer).
    
    Attributes:
        content: Footer text
        bbox: Location in image
        page_number: Which page this appears on
        includes_page_number: Whether footer itself contains page number
        repeated: Whether this footer appears on multiple pages
    
    TODO: Detect repeated footers across pages
    TODO: Extract page numbers from footers
    TODO: Classify footer type (author, date, document info, etc.)
    """
    content: str
    bbox: BoundingBox
    page_number: int
    confidence: float
    includes_page_number: bool = False
    repeated: bool = False
    
    def __post_init__(self):
        # TODO: Validate page_number >= 1
        pass


@dataclass
class Watermark:
    """
    Detected watermark or background text overlay.
    
    Attributes:
        content: Watermark text (if OCR'd)
        bbox: Location in image
        confidence: Detection confidence
        is_background: Whether text is clearly in background
        opacity_estimate: Estimated opacity (0-1)
        tilt_angle: Estimated rotation angle if tilted
    
    TODO: Detect watermark vs. regular text
    TODO: Estimate opacity and visual prominence
    TODO: Handle rotated/diagonal watermarks
    """
    content: Optional[str]
    bbox: BoundingBox
    confidence: float
    is_background: bool = True
    opacity_estimate: Optional[float] = None
    tilt_angle: Optional[float] = None
    
    def __post_init__(self):
        # TODO: Validate opacity_estimate is 0-1 if provided
        # TODO: Validate tilt_angle is reasonable (-180 to 180)
        pass


@dataclass
class BlockQuote:
    """
    Quoted or indented text block.
    
    Attributes:
        content: The quoted text
        bbox: Location in image
        confidence: Extraction confidence
        indentation_level: How deeply indented (0=none, 1=once indented, etc.)
        source: If attribution is detected, source of quote
        is_indented: Whether indentation is visible
        border_side: If bordered, which side (left, top, right, bottom)
    
    TODO: Detect indentation automatically
    TODO: Extract quote attribution/source
    TODO: Preserve formatting of block quotes
    """
    content: str
    bbox: BoundingBox
    confidence: float
    indentation_level: int = 0
    source: Optional[str] = None
    is_indented: bool = True
    border_side: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate indentation_level >= 0
        pass


@dataclass
class Caption:
    """
    Text caption for a figure, table, or other element.
    
    Attributes:
        content: Caption text
        caption_type: Type (figure caption, table caption, etc.)
        referenced_element_id: ID of element being captioned
        bbox: Location in image
        confidence: Extraction confidence
        caption_number: Numbering (e.g., "Figure 2.3", "Table 1")
    
    TODO: Auto-detect caption numbering
    TODO: Link captions to their referenced elements
    TODO: Extract meaningful metadata from captions
    """
    content: str
    caption_type: str  # "figure", "table", "image", etc.
    bbox: BoundingBox
    confidence: float
    referenced_element_id: Optional[str] = None
    caption_number: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate caption_type is recognized
        pass


@dataclass
class FigureRegion:
    """
    Visual figure/diagram/chart region (image content).
    
    Attributes:
        bbox: Location in document
        figure_type: Type of figure (chart, diagram, photo, graph, etc.)
        confidence: Detection confidence
        extracted_text: Any text detected within figure
        caption_id: Reference to associated caption (if exists)
        contains_tables: Whether figure contains embedded table
        description: Optional manually-added description
    
    TODO: Classify figure types automatically (chart, photo, etc.)
    TODO: Extract text from within figures
    TODO: Detect and link captions
    TODO: For charts, attempt to extract data
    """
    bbox: BoundingBox
    confidence: float
    figure_type: str = "unknown"  # "chart", "photo", "diagram", "graph", etc.
    extracted_text: Optional[str] = None
    caption_id: Optional[str] = None
    contains_tables: bool = False
    description: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate figure_type is recognized
        pass


@dataclass
class TableOfContents:
    """
    Table of contents entry or reference.
    
    Attributes:
        title: Section title
        page_number: Referenced page number
        level: Heading level (1=major, 2=subsection, etc.)
        bbox: Location in TOC
        confidence: Extraction confidence
        target_heading_id: ID of actual heading in document (if linked)
    
    TODO: Link TOC entries to actual headings in document
    TODO: Validate page numbers match
    TODO: Detect section hierarchy from TOC structure
    """
    title: str
    page_number: int
    level: int
    bbox: BoundingBox
    confidence: float
    target_heading_id: Optional[str] = None
    
    def __post_init__(self):
        # TODO: Validate level >= 1
        # TODO: Validate page_number >= 1
        pass


@dataclass
class IndexEntry:
    """
    Index entry or cross-reference.
    
    Attributes:
        term: The indexed term
        page_numbers: Pages where term appears
        level: Entry level (main term, subterm, etc.)
        bbox: Location in index
        confidence: Extraction confidence
        see_also: Related terms (cross-references)
    
    TODO: Parse index structure and cross-references
    TODO: Link to actual page occurrences
    """
    term: str
    page_numbers: List[int]
    level: int
    bbox: BoundingBox
    confidence: float
    see_also: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # TODO: Validate level >= 1
        # TODO: Validate page_numbers are all >= 1
        pass


# ============================================================================
# LAYER 3: Core OCR Models
# ============================================================================

@dataclass
class OCRTextResult:
    """
    Raw output from Tesseract OCR for a single text region.
    
    This represents the lowest-level OCR detection before any
    structural interpretation or element classification.
    
    Attributes:
        text: Extracted text
        confidence: Tesseract confidence (0-100, TODO: normalize to 0-1?)
        bbox: Bounding box of text region
        x_baseline: Y-coordinate of baseline for this text
        is_numeric: Detected whether text is numeric
        language: Detected language code
        font_name: Detected font name (if supported by OCR)
        font_size: Detected font size (if available)
        page_number: Which page this appears on
    
    TODO: Normalize confidence to 0-1 range
    TODO: Validate all coordinates are within image bounds
    TODO: Detect text directionality (LTR, RTL, etc.)
    """
    text: str
    confidence: float
    bbox: BoundingBox
    language: str = "eng"
    page_number: int = 1
    x_baseline: Optional[float] = None
    is_numeric: bool = False
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    
    def __post_init__(self):
        # TODO: Validate confidence is 0-100 (or normalize to 0-1)
        # TODO: Validate text is not empty
        # TODO: Validate page_number >= 1
        pass


@dataclass
class StructuralElement:
    """
    A detected structural element in the document (heading, table, list, etc.).
    
    This is the main output type that combines raw OCR detection with
    structural interpretation. Each element can be of multiple types.
    
    Attributes:
        element_id: Unique identifier for this element
        element_type: What type of element (from ElementType enum)
        content: Actual content (text, or dict for tables, etc.)
                 TODO: Use Union type for proper typing vs. current Any
        bbox: Bounding box in image
        confidence: Confidence in detection (0-1)
        page_number: Which page element appears on
        nesting_level: Hierarchy depth (0=top level, 1=indented, etc.)
        parent_id: ID of parent element (if nested)
        child_ids: IDs of child elements (if has children)
        metadata: Additional type-specific metadata
        processing_method: Which algorithm detected this (e.g., "tesseract_psm6", "table_detector_hough")
        detected_at_timestamp: When this element was detected
        source_ocr_results: References to raw OCRTextResult objects that compose this
        
    Design Note:
        - This model uses Any for content to avoid Union type explosion
        - TODO: Consider Pydantic discriminated unions for better type safety
        - TODO: Add specialized content typing once type hierarchy stabilizes
    
    TODO: Validate element_type matches content structure
    TODO: Implement content serialization methods:
        - to_dict()
        - to_json()
        - to_markdown()
        - to_csv() (for tables)
    TODO: Add query methods:
        - in_region(bbox) -> bool
        - overlaps_with(element) -> bool
        - contains(x, y) -> bool
    TODO: Consider content property type refinement (Union vs specific types per ElementType)
    """
    element_id: str
    element_type: ElementType
    content: Any  # TODO: Refine to Union[str, TableStructure, ListStructure, etc.]
    bbox: BoundingBox
    confidence: float
    page_number: int = 1
    nesting_level: int = 0
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_method: Optional[str] = None
    detected_at_timestamp: datetime = field(default_factory=datetime.now)
    source_ocr_results: List[str] = field(default_factory=list)  # OCRTextResult IDs
    
    def __post_init__(self):
        # TODO: Validate element_type enum value
        # TODO: Validate confidence is 0-1
        # TODO: Validate page_number >= 1
        # TODO: Validate nesting_level >= 0
        # TODO: Validate parent_id exists if specified
        # TODO: Validate child_ids all exist and reference this as parent
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary representation."""
        # TODO: Implement serialization
        pass
    
    def to_json(self) -> str:
        """Convert element to JSON string."""
        # TODO: Implement serialization
        pass
    
    def in_region(self, bbox: BoundingBox) -> bool:
        """Check if element is completely contained in region."""
        # TODO: Implement spatial query
        pass
    
    def overlaps_with(self, other: "StructuralElement") -> bool:
        """Check if element overlaps with another."""
        # TODO: Implement spatial query
        pass


# ============================================================================
# LAYER 4: Document-Level Models
# ============================================================================

@dataclass
class DocumentMetadata:
    """
    Metadata about processing a single document/image.
    
    Tracks all parameters and results for reproducibility and debugging.
    
    Attributes:
        source_file: Path to source image
        document_id: Unique identifier for this document
        processing_timestamp: When document was processed
        processing_duration: How long processing took (seconds)
        image_dimensions: (width, height) of image
        detected_language: Primary language detected
        total_elements_extracted: Count of structural elements found
        average_confidence: Mean confidence across all elements
        processing_status: Whether processing succeeded/failed/partial
        preprocessing_config: Parameters used for preprocessing
        ocr_config: Parameters used for OCR
        structure_detection_config: Parameters used for detection
        pages_processed: Total number of pages (for multi-page documents)
        errors_encountered: List of any errors during processing
        warnings: List of warnings (low confidence, etc.)
        quality_score: Overall quality assessment (0-1)
        metadata_custom: User-defined additional metadata
    
    TODO: Add method to save/load config for reproducibility
    TODO: Add method to generate processing report
    TODO: Implement quality scoring logic
    """
    source_file: str
    document_id: str
    processing_timestamp: datetime
    processing_duration: float
    image_dimensions: Tuple[int, int]
    detected_language: str
    total_elements_extracted: int
    average_confidence: float
    processing_status: ProcessingStatus
    pages_processed: int = 1
    quality_score: float = 0.0
    preprocessing_config: Optional[Dict[str, Any]] = None
    ocr_config: Optional[Dict[str, Any]] = None
    structure_detection_config: Optional[Dict[str, Any]] = None
    errors_encountered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata_custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # TODO: Validate source_file exists
        # TODO: Validate processing_duration > 0
        # TODO: Validate image_dimensions are positive
        # TODO: Validate confidence is 0-1
        # TODO: Validate quality_score is 0-1
        pass


@dataclass
class DocumentResult:
    """
    Complete OCR extraction result for a single document/image.
    
    Combines all detected structural elements with document-level metadata.
    
    Attributes:
        metadata: DocumentMetadata about processing
        elements: List of all StructuralElements detected
        raw_ocr_results: All raw OCRTextResult before structural interpretation
        element_index: Dict mapping element_id -> StructuralElement for quick lookup
    
    TODO: Add methods for:
        - get_elements_by_type(element_type) -> List[StructuralElement]
        - get_elements_in_region(bbox) -> List[StructuralElement]
        - get_elements_on_page(page_num) -> List[StructuralElement]
        - to_dataframe() -> pandas.DataFrame
        - to_json_hierarchical() -> str (preserving nesting)
        - export_to_file(format, path) -> None
    TODO: Implement element indexing for fast queries
    TODO: Validate element relationships (parent/child consistency)
    """
    metadata: DocumentMetadata
    elements: List[StructuralElement] = field(default_factory=list)
    raw_ocr_results: List[OCRTextResult] = field(default_factory=list)
    element_index: Dict[str, StructuralElement] = field(default_factory=dict)
    
    def __post_init__(self):
        # TODO: Build element_index from elements list
        # TODO: Validate element relationships
        # TODO: Validate all elements reference valid source_ocr_results
        pass
    
    def get_elements_by_type(self, element_type: ElementType) -> List[StructuralElement]:
        """Filter elements by type."""
        # TODO: Implement
        pass
    
    def get_elements_in_region(self, bbox: BoundingBox) -> List[StructuralElement]:
        """Get all elements intersecting a region."""
        # TODO: Implement spatial query
        pass
    
    def to_dataframe(self):
        """Export to pandas DataFrame for analysis."""
        # TODO: Implement conversion with columns:
        #       element_id, type, content, confidence, bbox, page, nesting_level, etc.
        pass
    
    def to_json(self, hierarchical: bool = False) -> str:
        """Export to JSON, optionally preserving hierarchy."""
        # TODO: Implement serialization
        pass


# ============================================================================
# LAYER 5: Batch Collection Models
# ============================================================================

@dataclass
class BatchStatistics:
    """
    Aggregate statistics for a batch of documents.
    
    Attributes:
        total_documents: Number of documents processed
        successful_documents: Number that completed successfully
        failed_documents: Number that failed
        partial_documents: Number with partial success
        total_elements: Total structural elements extracted
        elements_by_type: Count per element type
        average_confidence: Mean confidence across all elements
        confidence_distribution: Histogram of confidence scores
        total_processing_time: Total time for entire batch
        average_time_per_document: Mean processing time per doc
        languages_detected: Set of languages found
        errors_summary: Common errors encountered
    
    TODO: Add methods for:
        - print_summary() -> str
        - to_report() -> str (formatted report)
        - quality_assessment() -> float (batch quality score)
    """
    total_documents: int
    successful_documents: int
    failed_documents: int
    partial_documents: int
    total_elements: int
    average_confidence: float
    total_processing_time: float
    elements_by_type: Dict[ElementType, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    average_time_per_document: float = 0.0
    languages_detected: set = field(default_factory=set)
    errors_summary: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # TODO: Validate counts are consistent (success + failed + partial = total)
        # TODO: Compute average_time_per_document
        # TODO: Validate confidence is 0-1
        pass
    
    def print_summary(self) -> str:
        """Generate human-readable summary."""
        # TODO: Implement formatting
        pass


@dataclass
class BatchResult:
    """
    Complete result for a batch of documents.
    
    This is the top-level output object representing entire batch processing.
    
    Attributes:
        batch_id: Unique identifier for this batch
        created_at: When batch was processed
        documents: List of all DocumentResult objects
        statistics: BatchStatistics aggregating results
        document_index: Dict mapping document_id -> DocumentResult for lookup
        batch_config: Parameters used for this batch
        
    TODO: Add methods for:
        - to_csv(path) -> None (flat export with one row per element)
        - to_excel(path) -> None (multi-sheet: elements, documents, stats)
        - to_json(path, hierarchical=True) -> None (preserving structure)
        - to_dataframe() -> pandas.DataFrame
        - filter_by_type(element_type) -> BatchResult (new batch with filtered elements)
        - filter_by_confidence(min_conf) -> BatchResult
        - export_summary_report(path) -> None (human-readable)
    TODO: Consider streaming/chunked export for very large batches
    TODO: Add merge functionality for combining multiple batches
    """
    batch_id: str
    created_at: datetime
    documents: List[DocumentResult] = field(default_factory=list)
    statistics: Optional[BatchStatistics] = None
    document_index: Dict[str, DocumentResult] = field(default_factory=dict)
    batch_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # TODO: Build document_index from documents list
        # TODO: Compute statistics if not provided
        pass
    
    def to_csv(self, path: Path) -> None:
        """Export all elements as CSV (flat structure)."""
        # TODO: Implement flattened export with columns:
        #       batch_id, document_id, element_id, type, content, confidence, bbox, etc.
        pass
    
    def to_excel(self, path: Path) -> None:
        """Export to Excel with multiple sheets."""
        # TODO: Implement multi-sheet export:
        #       - "Elements": flattened elements
        #       - "Documents": document-level metadata
        #       - "Statistics": batch statistics
        #       - "Details": element-specific details (tables on separate sheet, etc.)
        pass
    
    def to_json(self, path: Path, hierarchical: bool = True) -> None:
        """Export to JSON file."""
        # TODO: Implement serialization to file
        pass
    
    def to_dataframe(self):
        """Export to pandas DataFrame."""
        # TODO: Implement conversion
        pass
    
    def filter_by_type(self, element_type: ElementType) -> "BatchResult":
        """Create new batch with only elements of specified type."""
        # TODO: Implement filtering
        pass
    
    def filter_by_confidence(self, min_confidence: float) -> "BatchResult":
        """Create new batch with only high-confidence elements."""
        # TODO: Implement filtering
        pass


# ============================================================================
# PERFORMANCE & SCALING CONSIDERATIONS
# ============================================================================

"""
MEMORY IMPLICATIONS FOR LARGE BATCHES:

1. Raw Image Storage:
   - Large images can consume significant memory if stored in memory
   - Consider streaming image processing instead of loading all at once
   - Possible optimization: Store only metadata + file paths, not image data
   
2. Element Storage:
   - Each StructuralElement stores full BoundingBox and content
   - Batch with 1000 documents × 100 elements = 100k element objects
   - TODO: Consider lazy-loading content until needed
   - TODO: Implement element compression (e.g., store coordinates as ints instead of floats)

3. OCR Results Duplication:
   - Currently storing both raw OCRTextResult and interpreted StructuralElements
   - If storing raw results is important for reproducibility, consider:
     - Storing only references (element_id -> OCRTextResult ids)
     - Compressing raw results after element extraction
     - Archiving raw results to disk if not needed immediately

4. Serialization:
   - Exporting massive batches to single JSON file can fail
   - TODO: Implement streaming export (line-delimited JSON)
   - TODO: Implement chunked CSV export
   - TODO: Consider Parquet format for columnar storage efficiency

5. Indexing:
   - Multiple indices (by type, by page, by confidence) can duplicate data in memory
   - TODO: Implement lazy index building
   - TODO: Consider building indices only for frequently-queried dimensions

SCALING STRATEGIES:

1. Batch Chunking:
   - Process documents in chunks (e.g., 100 at a time)
   - Export intermediate results to avoid memory bloat
   - TODO: Implement checkpointing for recovery from failures

2. Database Backend:
   - For persistent storage of very large batches
   - Store elements in SQLite, PostgreSQL, or similar
   - Load subsets on demand
   - TODO: Design database schema

3. Distributed Processing:
   - Use Celery + Redis for distributed batch processing
   - Process documents in parallel on multiple workers
   - TODO: Design serialization for distributed queue

4. Streaming Output:
   - Don't hold entire batch in memory
   - Stream results directly to CSV/JSON file
   - TODO: Implement iterator-based processing

TODO: Profile memory usage on realistic batch sizes (100-1000 documents)
TODO: Implement efficient export for batches with 10k+ documents
TODO: Add memory usage monitoring/reporting to processing pipeline
"""

