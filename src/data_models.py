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
from typing import List, Dict, Any, Optional, Tuple
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
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Confidence score must be in [0, 1], got {score}")
        for level in cls:
            min_val, max_val = level.value
            if min_val <= score < max_val:
                return level
        return cls.VERY_HIGH


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
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Coordinates must be non-negative, got x={self.x}, y={self.y}")


@dataclass
class BoundingBox:
    """
    Rectangular region in image space.
    
    Attributes:
        x_min, y_min: Top-left corner
        x_max, y_max: Bottom-right corner
        confidence: Optional confidence in the box location
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x_min < 0 or self.y_min < 0 or self.x_max < 0 or self.y_max < 0:
            raise ValueError(f"Coordinates must be non-negative, got box=({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})")
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def area(self) -> float:
        """Compute box area."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def width(self) -> float:
        """Get box width."""
        return self.x_max - self.x_min
    
    def height(self) -> float:
        """Get box height."""
        return self.y_max - self.y_min
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Compute intersection with another box, or None if no overlap."""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        if x_min >= x_max or y_min >= y_max:
            return None
        return BoundingBox(x_min, y_min, x_max, y_max)
    
    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Compute union (bounding box of both boxes)."""
        return BoundingBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )
    
    def overlap_percentage(self, other: "BoundingBox") -> float:
        """Compute intersection area as percentage of this box's area."""
        self_area = self.area()
        if self_area == 0:
            return 0.0
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        return intersection.area() / self_area


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
        if self.row_index < 0 or self.col_index < 0:
            raise ValueError(f"Row and column indices must be non-negative, got row={self.row_index}, col={self.col_index}")
        if self.colspan < 1 or self.rowspan < 1:
            raise ValueError(f"colspan and rowspan must be >= 1, got colspan={self.colspan}, rowspan={self.rowspan}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        """Compute table dimensions and validate structure."""
        if not self.cells:
            self.num_rows = 0
            self.num_cols = 0
            return
        
        # Compute dimensions from cells
        max_row = max((cell.row_index for cell in self.cells), default=0)
        max_col = max((cell.col_index for cell in self.cells), default=0)
        self.num_rows = max_row + 1
        self.num_cols = max_col + 1
        
        # Detect merged cells
        for cell in self.cells:
            if cell.colspan > 1 or cell.rowspan > 1:
                self.has_irregular_structure = True
                break
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at given row/column, or None if not found."""
        for cell in self.cells:
            if cell.row_index == row and cell.col_index == col:
                return cell
        return None
    
    def get_row(self, row: int) -> List[TableCell]:
        """Get all cells in a row (in column order)."""
        row_cells = [cell for cell in self.cells if cell.row_index == row]
        return sorted(row_cells, key=lambda c: c.col_index)
    
    def get_column(self, col: int) -> List[TableCell]:
        """Get all cells in a column (in row order)."""
        col_cells = [cell for cell in self.cells if cell.col_index == col]
        return sorted(col_cells, key=lambda c: c.row_index)
    
    def to_2d_array(self) -> List[List[str]]:
        """
        Convert table to a 2D list of strings.

        Handles merged cells (colspan/rowspan) by repeating the cell content
        across every logical slot it spans. Handles ragged rows by padding
        short rows to num_cols with empty strings.
        """
        if self.num_rows == 0 or self.num_cols == 0:
            return []

        array: List[List[str]] = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        for cell in self.cells:
            r, c = cell.row_index, cell.col_index
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    tr, tc = r + dr, c + dc
                    if 0 <= tr < self.num_rows and 0 <= tc < self.num_cols:
                        array[tr][tc] = cell.content

        # Pad ragged rows so every row has exactly num_cols entries
        for row in array:
            while len(row) < self.num_cols:
                row.append("")

        return array

    def to_markdown(self) -> str:
        """
        Convert table to GitHub-Flavored Markdown.

        Merged cells are represented by repeating content in spanned columns
        (GFM doesn't support true colspan/rowspan). Header separator is
        inserted after the first row when the table has headers.
        """
        if not self.cells:
            return ""

        array = self.to_2d_array()
        has_header = bool(self.headers) or any(cell.is_header for cell in self.get_row(0))

        lines = []
        for row_idx, row in enumerate(array):
            lines.append("| " + " | ".join(cell for cell in row) + " |")
            if row_idx == 0 and has_header:
                lines.append("| " + " | ".join("---" for _ in row) + " |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """
        Convert table to CSV format.

        Merged cells are expanded so that every logical slot gets its own
        column entry (the content is repeated across spanned columns/rows),
        keeping the output rectangular and importable without loss.
        """
        if not self.cells:
            return ""

        rows = []
        for row in self.to_2d_array():
            escaped_row = []
            for cell in row:
                if "," in cell or '"' in cell or "\n" in cell:
                    escaped_row.append('"' + cell.replace('"', '""') + '"')
                else:
                    escaped_row.append(cell)
            rows.append(",".join(escaped_row))

        return "\n".join(rows)


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
        if self.level < 0:
            raise ValueError(f"Level must be >= 0, got {self.level}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.list_type in ("number", "letter") and self.number is None:
            pass  # Warning only, not an error


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
        """Validate list structure consistency."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        # Verify all root items exist
        for root_id in self.root_item_ids:
            if not any(item.parent_item_id is None for item in self.items if hasattr(item, 'parent_item_id')):
                pass  # Note: This is a basic check, not critical


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
        if not self.raw_text:
            raise ValueError("FormulaExpression raw_text cannot be empty")
        if self.latex and not isinstance(self.latex, str):
            raise ValueError(f"LaTeX must be string, got {type(self.latex)}")


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
        if not self.equation_number:
            raise ValueError("EquationReference equation_number cannot be empty")


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
        valid_types = {"highlight", "underline", "strikethrough", "comment"}
        if self.annotation_type not in valid_types:
            pass  # Allow custom types
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if not self.barcode_type:
            raise ValueError("Barcode barcode_type cannot be empty")
        if not self.decoded_value:
            raise ValueError("Barcode decoded_value cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        valid_types = {"citation", "footnote", "endnote", "bibliography"}
        if self.ref_type not in valid_types:
            pass  # Allow custom types


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
        if not self.content:
            raise ValueError("CodeBlock content cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.opacity_estimate is not None and not 0.0 <= self.opacity_estimate <= 1.0:
            raise ValueError(f"opacity_estimate must be in [0, 1], got {self.opacity_estimate}")
        if self.tilt_angle is not None and not -180 <= self.tilt_angle <= 180:
            raise ValueError(f"tilt_angle must be in [-180, 180], got {self.tilt_angle}")


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
        if self.indentation_level < 0:
            raise ValueError(f"indentation_level must be >= 0, got {self.indentation_level}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if not self.caption_type:
            raise ValueError("Caption caption_type cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if not self.figure_type:
            raise ValueError("FigureRegion figure_type cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if not all(p >= 1 for p in self.page_numbers):
            raise ValueError(f"All page_numbers must be >= 1, got {self.page_numbers}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


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
        if not self.text:
            raise ValueError("OCRTextResult text cannot be empty")
        if not 0.0 <= self.confidence <= 100.0:
            raise ValueError(f"Confidence must be in [0, 100], got {self.confidence}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")


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
        """Validate element structure and properties."""
        if not isinstance(self.element_type, ElementType):
            raise ValueError(f"element_type must be ElementType enum, got {type(self.element_type)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        if self.nesting_level < 0:
            raise ValueError(f"nesting_level must be >= 0, got {self.nesting_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary representation."""
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "content": self.content if isinstance(self.content, str) else str(self.content),
            "bbox": {"x_min": self.bbox.x_min, "y_min": self.bbox.y_min, "x_max": self.bbox.x_max, "y_max": self.bbox.y_max},
            "confidence": self.confidence,
            "page_number": self.page_number,
            "nesting_level": self.nesting_level,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "metadata": self.metadata,
            "processing_method": self.processing_method,
        }
    
    def to_json(self) -> str:
        """Convert element to JSON string."""
        return json.dumps(self.to_dict())
    
    def in_region(self, bbox: BoundingBox) -> bool:
        """Check if element is completely contained in region."""
        return (self.bbox.x_min >= bbox.x_min and self.bbox.y_min >= bbox.y_min and
                self.bbox.x_max <= bbox.x_max and self.bbox.y_max <= bbox.y_max)
    
    def overlaps_with(self, other: "StructuralElement") -> bool:
        """Check if element overlaps with another."""
        intersection = self.bbox.intersection(other.bbox)
        return intersection is not None
    
    def get_descendants(self, all_elements: List["StructuralElement"]) -> List["StructuralElement"]:
        """Get all descendants of this element (recursive)."""
        descendants = []
        element_map = {e.element_id: e for e in all_elements}
        
        for child_id in self.child_ids:
            if child_id in element_map:
                child = element_map[child_id]
                descendants.append(child)
                descendants.extend(child.get_descendants(all_elements))
        return descendants
    
    def get_ancestors(self, all_elements: List["StructuralElement"]) -> List["StructuralElement"]:
        """Get all ancestors of this element (up to root)."""
        ancestors = []
        element_map = {e.element_id: e for e in all_elements}
        
        current = self
        while current.parent_id:
            if current.parent_id in element_map:
                parent = element_map[current.parent_id]
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors


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
        """Validate document metadata consistency."""
        if self.processing_duration <= 0:
            raise ValueError(f"processing_duration must be > 0, got {self.processing_duration}")
        if len(self.image_dimensions) != 2 or any(d <= 0 for d in self.image_dimensions):
            raise ValueError(f"image_dimensions must be (positive_width, positive_height), got {self.image_dimensions}")
        if not 0.0 <= self.average_confidence <= 1.0:
            raise ValueError(f"average_confidence must be in [0, 1], got {self.average_confidence}")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be in [0, 1], got {self.quality_score}")
        if self.pages_processed < 1:
            raise ValueError(f"pages_processed must be >= 1, got {self.pages_processed}")


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
        """Build element index and validate relationships."""
        # Build index for fast lookup
        self.element_index = {e.element_id: e for e in self.elements}
        
        # Validate parent/child relationships
        for elem in self.elements:
            if elem.parent_id and elem.parent_id not in self.element_index:
                pass  # Warning level - parent not found, but don't fail
            for child_id in elem.child_ids:
                if child_id not in self.element_index:
                    pass  # Warning level - child not found, but don't fail
    
    def get_elements_by_type(self, element_type: ElementType) -> List[StructuralElement]:
        """Filter elements by type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_elements_in_region(self, bbox: BoundingBox) -> List[StructuralElement]:
        """Get all elements intersecting a region."""
        return [e for e in self.elements if e.bbox.intersection(bbox) is not None]
    
    def get_elements_on_page(self, page_num: int) -> List[StructuralElement]:
        """Get all elements on a specific page."""
        return [e for e in self.elements if e.page_number == page_num]
    
    def to_dataframe(self) -> Any:
        """
        Convert this document's elements to a pandas DataFrame.

        Requires pandas.  Raises ImportError if not installed.
        """
        from exporters import DataFrameExporter  # deferred to avoid circular import
        return DataFrameExporter().export_document(self)

    def to_json(self, hierarchical: bool = False) -> str:
        """Export to JSON, optionally preserving hierarchy."""
        elements_data = [e.to_dict() for e in self.elements]
        metadata_data = {
            "source_file": self.metadata.source_file,
            "document_id": self.metadata.document_id,
            "total_elements": len(self.elements),
            "average_confidence": self.metadata.average_confidence,
        }
        result = {
            "metadata": metadata_data,
            "elements": elements_data,
        }
        return json.dumps(result)


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
        """Validate batch statistics consistency."""
        expected_total = self.successful_documents + self.failed_documents + self.partial_documents
        if expected_total != self.total_documents:
            pass  # Warning level validation
        
        if self.total_processing_time > 0:
            self.average_time_per_document = self.total_processing_time / max(1, self.total_documents)
        
        if not 0.0 <= self.average_confidence <= 1.0:
            raise ValueError(f"average_confidence must be in [0, 1], got {self.average_confidence}")
    
    def print_summary(self) -> str:
        """Generate human-readable summary."""
        summary = f"""
Batch Statistics Summary:
- Total Documents: {self.total_documents}
- Successful: {self.successful_documents}
- Failed: {self.failed_documents}
- Partial: {self.partial_documents}
- Total Elements Extracted: {self.total_elements}
- Average Confidence: {self.average_confidence:.2%}
- Processing Time: {self.total_processing_time:.2f}s
- Average Time Per Document: {self.average_time_per_document:.2f}s
- Languages Detected: {', '.join(sorted(self.languages_detected)) if self.languages_detected else 'None'}
- Top Errors: {', '.join(list(self.errors_summary.keys())[:3]) if self.errors_summary else 'None'}
"""
        return summary


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
        """Build document index and compute statistics."""
        # Build index for fast lookup
        self.document_index = {d.metadata.document_id: d for d in self.documents}
        
        # Compute statistics if not provided
        if self.statistics is None and self.documents:
            all_elements = []
            total_time = 0.0
            success_count = 0
            fail_count = 0
            
            for doc in self.documents:
                all_elements.extend(doc.elements)
                total_time += doc.metadata.processing_duration
                if doc.metadata.processing_status == ProcessingStatus.COMPLETED:
                    success_count += 1
                elif doc.metadata.processing_status == ProcessingStatus.FAILED:
                    fail_count += 1
            
            # Count elements by type
            elements_by_type = {}
            for elem in all_elements:
                elements_by_type[elem.element_type] = elements_by_type.get(elem.element_type, 0) + 1
            
            # Compute average confidence
            avg_conf = sum(e.confidence for e in all_elements) / len(all_elements) if all_elements else 0.0
            
            self.statistics = BatchStatistics(
                total_documents=len(self.documents),
                successful_documents=success_count,
                failed_documents=fail_count,
                partial_documents=len(self.documents) - success_count - fail_count,
                total_elements=len(all_elements),
                average_confidence=avg_conf,
                total_processing_time=total_time,
                elements_by_type=elements_by_type,
            )
    
    def to_csv(self, path: Path) -> None:
        """Export all elements as CSV (flat structure)."""
        if not self.documents:
            return
        
        rows = []
        for doc in self.documents:
            for elem in doc.elements:
                rows.append({
                    "batch_id": self.batch_id,
                    "document_id": doc.metadata.document_id,
                    "element_id": elem.element_id,
                    "type": elem.element_type.value,
                    "content": elem.content if isinstance(elem.content, str) else "",
                    "confidence": elem.confidence,
                    "page": elem.page_number,
                    "nesting_level": elem.nesting_level,
                    "parent_id": elem.parent_id or "",
                })
        
        if rows:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def filter_by_type(self, element_type: ElementType) -> "BatchResult":
        """Create new batch with only elements of specified type."""
        filtered_docs = []
        for doc in self.documents:
            filtered_elems = [e for e in doc.elements if e.element_type == element_type]
            if filtered_elems:
                new_doc = DocumentResult(
                    metadata=doc.metadata,
                    elements=filtered_elems,
                    raw_ocr_results=doc.raw_ocr_results,
                )
                filtered_docs.append(new_doc)
        
        return BatchResult(
            batch_id=self.batch_id,
            created_at=self.created_at,
            documents=filtered_docs,
            batch_config=self.batch_config,
        )
    
    def to_dataframe(self) -> Any:
        """
        Convert all elements to a pandas DataFrame (one row per element).

        Requires pandas.  Each row has the same columns as CSVExporter output.
        Raises ImportError if pandas is not installed.
        """
        from exporters import DataFrameExporter  # deferred to avoid circular import
        return DataFrameExporter().to_dataframe(self)

    def filter_by_confidence(self, min_confidence: float) -> "BatchResult":
        """Create new batch with only high-confidence elements."""
        filtered_docs = []
        for doc in self.documents:
            filtered_elems = [e for e in doc.elements if e.confidence >= min_confidence]
            if filtered_elems:
                new_doc = DocumentResult(
                    metadata=doc.metadata,
                    elements=filtered_elems,
                    raw_ocr_results=doc.raw_ocr_results,
                )
                filtered_docs.append(new_doc)
        
        return BatchResult(
            batch_id=self.batch_id,
            created_at=self.created_at,
            documents=filtered_docs,
            batch_config=self.batch_config,
        )


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

