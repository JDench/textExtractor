# Architectural Decisions for OCR Text Extraction System

## Overview
This document crystallizes key architectural decisions that shape the entire system design, ensuring consistency, scalability, and maintainability.

---

## 1. COORDINATE SYSTEM: Relative Percentages with Absolute Roots

### Decision
- **Root level** (Document/Page): Stores absolute pixel dimensions (e.g., 1920×1080 px)
- **All child elements**: Use percentage-based coordinates (0.0 to 1.0) relative to their parent bounding box
- **Reading direction**: Left-to-Right (LTR) - x=0.0 is left edge, x=1.0 is right edge
- **Cached access**: `absolute_bbox_px` computed and cached on demand for pixel-level queries

### Rationale
- **Scale invariance**: Document resizing preserves all element relationships
- **Hierarchical composition**: Each element's position expressed relative to parent container
- **Memory efficiency**: Percentages require less precision than absolute coordinates
- **Consistency**: All coordinate math uses same 0.0-1.0 scale
- **LTR support**: Natural reading order (left to right) matches English and most document layouts

### Coordinate Semantics

**Percentage-Based Coordinates:**
```
x: 0.0 = left edge of parent bbox
x: 1.0 = right edge of parent bbox
y: 0.0 = top edge of parent bbox  
y: 1.0 = bottom edge of parent bbox

Example:
  Parent bbox = (0, 0) to (1000, 500) pixels
  Child element with x=0.1, y=0.2, width=0.3, height=0.4:
    Absolute pixels = (100, 100) to (400, 300)
```

### Implementation in Data Model

```python
@dataclass
class RelativeBoundingBox:
    """
    Bounding box using percentage coordinates (0.0-1.0) relative to parent.
    Coordinates are normalized to parent container for scale invariance.
    """
    x_min: float  # 0.0 = left edge
    y_min: float  # 0.0 = top edge
    x_max: float  # 1.0 = right edge
    y_max: float  # 1.0 = bottom edge
    
    def __post_init__(self):
        assert 0.0 <= self.x_min <= self.x_max <= 1.0, "x coords must be 0-1 and ordered"
        assert 0.0 <= self.y_min <= self.y_max <= 1.0, "y coords must be 0-1 and ordered"
    
    def to_absolute(self, parent_bbox_px: "AbsoluteBoundingBox") -> "AbsoluteBoundingBox":
        """Convert relative coordinates to absolute pixels using parent dimensions."""
        parent_width = parent_bbox_px.width_px
        parent_height = parent_bbox_px.height_px
        
        abs_x_min = parent_bbox_px.x_min_px + (self.x_min * parent_width)
        abs_y_min = parent_bbox_px.y_min_px + (self.y_min * parent_height)
        abs_x_max = parent_bbox_px.x_min_px + (self.x_max * parent_width)
        abs_y_max = parent_bbox_px.y_min_px + (self.y_max * parent_height)
        
        return AbsoluteBoundingBox(
            x_min_px=abs_x_min,
            y_min_px=abs_y_min,
            x_max_px=abs_x_max,
            y_max_px=abs_y_max
        )


@dataclass
class AbsoluteBoundingBox:
    """
    Bounding box using absolute pixel coordinates.
    Only used at root level (page/document).
    """
    x_min_px: float
    y_min_px: float
    x_max_px: float
    y_max_px: float
    
    @property
    def width_px(self) -> float:
        return self.x_max_px - self.x_min_px
    
    @property
    def height_px(self) -> float:
        return self.y_max_px - self.y_min_px
    
    def to_relative(self) -> RelativeBoundingBox:
        """Convert to 0-1 normalized coordinates."""
        return RelativeBoundingBox(
            x_min=0.0,
            y_min=0.0,
            x_max=1.0,
            y_max=1.0
        )
```

### Root Element Tracking

```python
@dataclass
class DocumentMetadata:
    """Tracks absolute pixel dimensions at root level."""
    source_file: str
    document_id: str
    image_dimensions_px: Tuple[int, int]  # (width, height) in pixels
    
    @property
    def root_bbox(self) -> AbsoluteBoundingBox:
        """Root bounding box for entire document."""
        width, height = self.image_dimensions_px
        return AbsoluteBoundingBox(
            x_min_px=0.0,
            y_min_px=0.0,
            x_max_px=float(width),
            y_max_px=float(height)
        )
```

### Cached Pixel Queries

```python
@dataclass
class StructuralElement:
    """Unified element with efficient coordinate access."""
    
    bbox_relative: RelativeBoundingBox  # Stored as percentages
    
    # Cached values for fast pixel-level queries
    _cached_bbox_px: Optional[AbsoluteBoundingBox] = field(default=None, init=False, repr=False)
    _cached_parent_bbox_px: Optional[AbsoluteBoundingBox] = field(default=None, init=False, repr=False)
    
    def get_absolute_bbox(self, document: "DocumentResult") -> AbsoluteBoundingBox:
        """Get absolute pixel coordinates, computing from parent hierarchy if needed."""
        # Root element: direct conversion
        if self.parent_id is None:
            if self._cached_bbox_px is None:
                self._cached_bbox_px = self.bbox_relative.to_absolute(document.metadata.root_bbox)
            return self._cached_bbox_px
        
        # Child element: convert relative to parent's absolute coords
        parent = document.element_index[self.parent_id]
        parent_abs_bbox = parent.get_absolute_bbox(document)
        
        if self._cached_bbox_px is None:
            self._cached_bbox_px = self.bbox_relative.to_absolute(parent_abs_bbox)
        return self._cached_bbox_px
    
    def invalidate_cache(self):
        """Invalidate cached coordinates (call if element is modified)."""
        self._cached_bbox_px = None
        self._cached_parent_bbox_px = None
```

---

## 2. UNIFIED ELEMENT MODEL: Single StructuralElement Wrapper

### Decision
All 20+ element types (text, table, list, formula, annotation, barcode, etc.) are wrapped in the same `StructuralElement` container. Content is polymorphic using type-specific content classes and Union typing.

### Rationale
- **Consistency**: Single interface for all element operations
- **Flexibility**: Easy to add new element types without changing core
- **Simplicity**: Uniform serialization, querying, and navigation
- **Type safety**: Union types provide static type checking per element type

### Content Polymorphism

```python
# Type-specific content classes
@dataclass
class TextContent:
    """Content for TEXT, HEADING, BLOCK_QUOTE elements."""
    text: str
    language: str = "en"
    is_numeric: bool = False
    detected_language: Optional[str] = None


@dataclass
class ListContent:
    """Content for LIST elements."""
    items: List["ListItem"]
    list_type: str  # "bullet", "number", "letter", "mixed"
    max_nesting_level: int


@dataclass
class TableContent:
    """Content for TABLE elements."""
    cells: List["TableCell"]
    num_rows: int
    num_cols: int
    headers: Optional[List[str]] = None
    caption: Optional[str] = None
    has_merged_cells: bool = False


@dataclass
class FormulaContent:
    """Content for FORMULA, EQUATION elements."""
    raw_text: str
    latex: Optional[str] = None
    mathml: Optional[str] = None
    variables: List[str] = field(default_factory=list)


@dataclass
class AnnotationContent:
    """Content for ANNOTATION elements."""
    text: str
    annotation_type: str  # "highlight", "underline", "strikethrough", "comment"
    color: Optional[str] = None
    note: Optional[str] = None


@dataclass
class BarcodeContent:
    """Content for BARCODE elements."""
    barcode_type: str  # "QR", "Code128", "UPC", etc.
    decoded_value: str
    raw_image: Optional[bytes] = None


@dataclass
class ReferenceContent:
    """Content for REFERENCE, CITATION, FOOTNOTE elements."""
    text: str
    ref_type: str  # "citation", "footnote", "endnote", "bibliography"
    reference_id: str
    target_ref: Optional[str] = None


@dataclass
class CodeContent:
    """Content for CODE_BLOCK elements."""
    code: str
    language: Optional[str] = None
    line_numbers: bool = False
    start_line_number: Optional[int] = None


@dataclass
class FigureContent:
    """Content for FIGURE elements."""
    figure_type: str  # "photo", "chart", "diagram", "graph", "unknown"
    extracted_text: Optional[str] = None
    caption_id: Optional[str] = None


@dataclass
class CaptionContent:
    """Content for CAPTION elements."""
    text: str
    caption_type: str  # "figure", "table", "image"
    caption_number: Optional[str] = None
    referenced_element_id: Optional[str] = None


# Union type for all content variants
ElementContent = Union[
    TextContent, ListContent, TableContent, FormulaContent, AnnotationContent,
    BarcodeContent, ReferenceContent, CodeContent, FigureContent, CaptionContent,
    # Additional types as needed
]


@dataclass
class StructuralElement:
    """
    Unified wrapper for all structural element types.
    Polymorphic content via Union typing.
    """
    element_id: str
    element_type: ElementType  # TEXT, HEADING, TABLE, LIST, FORMULA, etc.
    content: ElementContent  # Type-specific content
    bbox_relative: RelativeBoundingBox
    confidence: float
    
    page_number: int = 1
    nesting_level: int = 0
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_trace: Optional["ProcessingTrace"] = None
    source_ocr_result_ids: List[str] = field(default_factory=list)
    
    # Cached values
    _cached_bbox_px: Optional[AbsoluteBoundingBox] = field(default=None, init=False, repr=False)
```

### Benefits of Unified Model

**Unified Querying:**
```python
# Single list for all elements
all_elements: List[StructuralElement]

# Filter by type
tables = [e for e in all_elements if e.element_type == ElementType.TABLE]
headings = [e for e in all_elements if e.element_type == ElementType.HEADING]

# Query across types
high_confidence = [e for e in all_elements if e.confidence > 0.9]
on_page_2 = [e for e in all_elements if e.page_number == 2]
```

**Consistent Serialization:**
```python
# Same serialization for all types
for element in all_elements:
    element.to_dict()
    element.to_json()
    # ... same interface regardless of element_type
```

**Uniform Navigation:**
```python
# Parent-child navigation works for all types
def get_children(element: StructuralElement, doc: DocumentResult) -> List[StructuralElement]:
    return [doc.element_index[cid] for cid in element.child_ids]
```

---

## 3. PARAMETERIZED DETECTION: Pluggable Strategy Pattern

### Decision
Detection is fully parameterized using a strategy pattern. Each element type has:
- Enable/disable flag
- Strategy selection (for types with multiple detection approaches)
- Per-type confidence thresholds
- Type-specific parameters

### Rationale
- **Flexibility**: Experiment with detection strategies without code changes
- **Reproducibility**: Configuration captures exact detection logic
- **Performance**: Enable only needed detectors
- **Debugging**: Each detector has isolated configuration

### Configuration Hierarchy

```python
@dataclass
class DetectorConfig(ABC):
    """Base class for all detector configurations."""
    enabled: bool = True
    min_confidence: float = 0.3
    
    def validate(self):
        """Subclasses implement type-specific validation."""
        pass


@dataclass
class TextDetectorConfig(DetectorConfig):
    """Configuration for text/heading/paragraph detection."""
    detect_headings: bool = True
    heading_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    detect_block_quotes: bool = True
    min_heading_size_ratio: float = 1.2  # Min size vs body text


@dataclass
class ListDetectorConfig(DetectorConfig):
    """Configuration for list detection."""
    strategy: str = "marker"  # "marker", "spatial", "multi"
    detect_nested_lists: bool = True
    min_list_items: int = 2


@dataclass
class TableDetectorConfig(DetectorConfig):
    """Configuration for table detection."""
    strategies: List[str] = field(default_factory=lambda: ["line_based", "content_based"])
    detect_merged_cells: bool = True
    min_table_rows: int = 2
    min_table_cols: int = 2
    line_detection_threshold: float = 0.5
    content_spacing_tolerance: float = 0.1


@dataclass
class FormulaDetectorConfig(DetectorConfig):
    """Configuration for formula/equation detection."""
    detect_formulas: bool = True
    detect_equation_numbers: bool = True
    attempt_latex_conversion: bool = True


@dataclass
class AnnotationDetectorConfig(DetectorConfig):
    """Configuration for annotation detection."""
    annotation_types: List[str] = field(
        default_factory=lambda: ["highlight", "underline", "strikethrough"]
    )
    detect_handwriting: bool = False  # Requires specialized model
    highlight_color_threshold: float = 0.7


@dataclass
class WatermarkDetectorConfig(DetectorConfig):
    """Configuration for watermark detection."""
    min_opacity_estimate: float = 0.05  # 5% minimum visible
    detect_rotated: bool = True


@dataclass
class BarcodeDetectorConfig(DetectorConfig):
    """Configuration for barcode/QR detection."""
    barcode_types: List[str] = field(
        default_factory=lambda: ["QR", "Code128", "UPC", "EAN"]
    )


@dataclass
class StructureDetectionConfig:
    """Master configuration for all structure detection."""
    
    # Per-type detector configs
    text_config: TextDetectorConfig = field(default_factory=TextDetectorConfig)
    list_config: ListDetectorConfig = field(default_factory=ListDetectorConfig)
    table_config: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    formula_config: FormulaDetectorConfig = field(default_factory=FormulaDetectorConfig)
    annotation_config: AnnotationDetectorConfig = field(default_factory=AnnotationDetectorConfig)
    watermark_config: WatermarkDetectorConfig = field(default_factory=WatermarkDetectorConfig)
    barcode_config: BarcodeDetectorConfig = field(default_factory=BarcodeDetectorConfig)
    
    # Global settings
    global_min_confidence: float = 0.3
    
    def validate_all(self):
        """Validate all detector configs."""
        for config in [self.text_config, self.list_config, self.table_config, 
                       self.formula_config, self.annotation_config, 
                       self.watermark_config, self.barcode_config]:
            config.validate()
```

### Detector Interface

```python
@dataclass
class ElementDetectionContext:
    """Context passed to detectors with all needed information."""
    ocr_results: List["OCRTextResult"]
    image: np.ndarray
    image_dims_px: Tuple[int, int]
    document_metadata: "DocumentMetadata"
    already_detected_elements: List["StructuralElement"]  # Elements found by prior detectors
    config: StructureDetectionConfig


class ElementDetector(ABC):
    """Base class for all element detectors."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
    
    @abstractmethod
    def detect(self, context: ElementDetectionContext) -> List[StructuralElement]:
        """
        Detect elements of this type.
        
        Returns:
            List of detected StructuralElement objects with processing traces.
        """
        pass


class TextDetector(ElementDetector):
    def detect(self, context: ElementDetectionContext) -> List[StructuralElement]:
        if not self.config.enabled:
            return []
        # Implementation detects TEXT, HEADING, BLOCK_QUOTE elements
        pass


class ListDetector(ElementDetector):
    def detect(self, context: ElementDetectionContext) -> List[StructuralElement]:
        if not self.config.enabled:
            return []
        # Implementation detects LIST elements using configured strategy
        pass


class TableDetector(ElementDetector):
    def detect(self, context: ElementDetectionContext) -> List[StructuralElement]:
        if not self.config.enabled:
            return []
        # Implementation detects TABLE elements using configured strategies
        pass
```

### Configuration in Practice

```yaml
# config.yaml
structure_detection:
  text:
    enabled: true
    detect_headings: true
    heading_levels: [1, 2, 3, 4, 5, 6]
    min_confidence: 0.5
  
  list:
    enabled: true
    strategy: "multi"  # Try marker then spatial
    min_confidence: 0.4
  
  table:
    enabled: true
    strategies: ["line_based", "content_based"]
    detect_merged_cells: true
    min_confidence: 0.6
  
  formula:
    enabled: true
    attempt_latex_conversion: true
    min_confidence: 0.5
  
  annotation:
    enabled: true
    annotation_types: ["highlight", "underline"]
    detect_handwriting: false
    min_confidence: 0.4
  
  watermark:
    enabled: false  # Disabled for this batch
  
  barcode:
    enabled: true
    min_confidence: 0.8
```

---

## 4. HIERARCHICAL RELATIONS: Parent-Child Tracking

### Decision
All hierarchical relationships are tracked via:
- `parent_id`: Reference to immediate parent (null for root elements)
- `child_ids`: List of immediate children (empty for leaf elements)
- `nesting_level`: Depth in hierarchy (0=root, 1=direct children, etc.)
- Optional transitive indexes for fast ancestor/descendant queries

### Rationale
- **Flexibility**: Arbitrary nesting depth for complex documents
- **Query efficiency**: Parent/child relationships enable fast tree traversal
- **Composability**: Elements can contain other elements at any level
- **Reproducibility**: Hierarchy captured explicitly in data

### Hierarchy Examples

**Text with Nested Lists:**
```
Heading "Introduction" (level=0, parent=null)
├─ Paragraph (level=1, parent=Heading)
├─ List (level=1, parent=Heading)
│  ├─ ListItem "First point" (level=2, parent=List)
│  ├─ ListItem "Second point" (level=2, parent=List)
│  │  └─ NestedList (level=3, parent=ListItem)
│  │     ├─ NestedItem "2.1" (level=4, parent=NestedList)
│  │     └─ NestedItem "2.2" (level=4, parent=NestedList)
│  └─ ListItem "Third point" (level=2, parent=List)
└─ Paragraph (level=1, parent=Heading)
```

**Complex Multi-Section:**
```
Heading "Section 1" (level=0, parent=null)
├─ Paragraph (level=1)
├─ Table (level=1)
│  └─ Caption (level=2, parent=Table)
├─ List (level=1)
└─ Heading "Subsection 1.1" (level=1, parent=Heading)
   ├─ Paragraph (level=2)
   └─ Figure (level=2)
      └─ Caption (level=3, parent=Figure)
```

### HierarchyHelper Utilities

```python
class HierarchyHelper:
    """Navigation utilities for element hierarchies."""
    
    def __init__(self, document: "DocumentResult"):
        self.document = document
        self.index = document.element_index
    
    def get_parent(self, element: StructuralElement) -> Optional[StructuralElement]:
        """Get immediate parent element."""
        if element.parent_id is None:
            return None
        return self.index.get(element.parent_id)
    
    def get_children(self, element: StructuralElement) -> List[StructuralElement]:
        """Get all immediate children."""
        return [self.index[cid] for cid in element.child_ids if cid in self.index]
    
    def get_ancestors(self, element: StructuralElement) -> List[StructuralElement]:
        """Get all ancestors from immediate parent to root."""
        ancestors = []
        current = self.get_parent(element)
        while current is not None:
            ancestors.append(current)
            current = self.get_parent(current)
        return ancestors
    
    def get_descendants(self, element: StructuralElement) -> List[StructuralElement]:
        """Get all descendants (recursive)."""
        descendants = []
        for child in self.get_children(element):
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        return descendants
    
    def get_siblings(self, element: StructuralElement) -> List[StructuralElement]:
        """Get elements at same nesting level with same parent."""
        parent = self.get_parent(element)
        if parent is None:
            return []
        return [c for c in self.get_children(parent) if c.element_id != element.element_id]
    
    def build_tree(self, root_id: Optional[str] = None) -> "HierarchyTree":
        """Build complete hierarchical tree structure."""
        # Implementation builds tree starting from root or specified element
        pass
    
    def get_containing_heading(self, element: StructuralElement) -> Optional[StructuralElement]:
        """Find the heading that contains this element (walks up ancestors)."""
        for ancestor in self.get_ancestors(element):
            if ancestor.element_type == ElementType.HEADING:
                return ancestor
        return None
    
    def get_section_elements(self, heading: StructuralElement) -> List[StructuralElement]:
        """Get all elements in a section (heading + its descendants)."""
        return [heading] + self.get_descendants(heading)


@dataclass
class HierarchyTree:
    """Complete hierarchical tree for visualization/export."""
    root_elements: List["TreeNode"]  # Top-level elements
    
    def to_markdown(self, indent: str = "  ") -> str:
        """Export tree as markdown outline."""
        pass
    
    def to_json(self) -> str:
        """Export tree as nested JSON."""
        pass


@dataclass
class TreeNode:
    """Single node in hierarchy tree."""
    element: StructuralElement
    children: List["TreeNode"] = field(default_factory=list)
    
    def to_markdown(self, level: int = 0) -> str:
        """Convert subtree to markdown."""
        pass
```

### Usage Examples

```python
# Navigate hierarchy
doc = DocumentResult(...)
helper = HierarchyHelper(doc)

# Get structure around an element
element = doc.element_index["table_1"]
parent = helper.get_parent(element)
siblings = helper.get_siblings(element)
section_heading = helper.get_containing_heading(element)

# Explore entire section
section_elements = helper.get_section_elements(section_heading)

# Export structure
tree = helper.build_tree()
markdown_outline = tree.to_markdown()
```

---

## 5. PROCESSING TRACEBACK: Full Audit Trail

### Decision
Every element has a complete `ProcessingTrace` that records:
- Which detector found it (detector name, version)
- Which detector config was used
- Step-by-step processing pipeline
- Confidence breakdown per component
- Timing information
- Links to raw OCR results

### Rationale
- **Reproducibility**: Re-run exact same processing with same config
- **Debugging**: Understand why/how elements were detected
- **Transparency**: Full audit trail for validation
- **Quality assessment**: Confidence breakdown shows component strengths/weaknesses
- **Error recovery**: Identify which step failed and why

### Trace Data Structures

```python
@dataclass
class ProcessingStep:
    """Single step in detection pipeline."""
    step_name: str  # e.g., "line_detection", "cell_extraction", "merge"
    detector_method: str  # e.g., "_detect_line_based", "hough_transform"
    status: str  # "success", "failed", "skipped"
    duration_ms: float
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        assert self.status in ["success", "failed", "skipped"], "Invalid status"
        assert self.confidence_score is None or 0.0 <= self.confidence_score <= 1.0


@dataclass
class ProcessingTrace:
    """Complete trace of how an element was detected."""
    element_id: str
    detector_name: str  # e.g., "TableDetector", "ListDetector"
    detector_version: str  # e.g., "1.0.0"
    created_at: datetime
    detector_config: Dict[str, Any]  # Serialized DetectorConfig used
    
    # Pipeline steps
    steps: List[ProcessingStep] = field(default_factory=list)
    
    # Final results
    final_status: str = "unknown"  # "success", "failed", "partial"
    final_confidence: float = 0.0
    processing_duration_ms: float = 0.0
    
    # Provenance
    source_ocr_result_ids: List[str] = field(default_factory=list)  # Raw OCR results used
    merged_from_element_ids: List[str] = field(default_factory=list)  # If merged from multiple
    
    def add_step(self, step: ProcessingStep):
        """Record a processing step."""
        self.steps.append(step)
        self.processing_duration_ms += step.duration_ms
    
    def set_result(self, status: str, confidence: float):
        """Set final result of detection."""
        self.final_status = status
        self.final_confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Export trace as dictionary."""
        return {
            "element_id": self.element_id,
            "detector": self.detector_name,
            "version": self.detector_version,
            "created_at": self.created_at.isoformat(),
            "config": self.detector_config,
            "steps": [asdict(step) for step in self.steps],
            "final_status": self.final_status,
            "final_confidence": self.final_confidence,
            "processing_duration_ms": self.processing_duration_ms,
            "source_ocr_ids": self.source_ocr_result_ids
        }
    
    def to_readable_string(self) -> str:
        """Human-readable trace output."""
        lines = [
            f"Element: {self.element_id}",
            f"Detected by: {self.detector_name} v{self.detector_version}",
            f"Status: {self.final_status} (confidence: {self.final_confidence:.2%})",
            f"Duration: {self.processing_duration_ms:.1f}ms",
            f"Steps:"
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step.step_name}")
            lines.append(f"     Method: {step.detector_method}")
            lines.append(f"     Status: {step.status}")
            lines.append(f"     Duration: {step.duration_ms:.1f}ms")
            if step.confidence_score is not None:
                lines.append(f"     Confidence: {step.confidence_score:.2%}")
            if step.error_message:
                lines.append(f"     Error: {step.error_message}")
        return "\n".join(lines)
```

### Recording Traces in Detectors

```python
class TableDetector(ElementDetector):
    def detect(self, context: ElementDetectionContext) -> List[StructuralElement]:
        elements = []
        
        for strategy in self.config.strategies:
            start_time = time.time()
            
            # Create trace
            trace = ProcessingTrace(
                element_id="",  # Will be set after element created
                detector_name=self.__class__.__name__,
                detector_version="1.0.0",
                created_at=datetime.now(),
                detector_config=asdict(self.config)
            )
            
            # Step 1: Line detection
            step1_start = time.time()
            try:
                line_tables = self._detect_line_based(context.image)
                trace.add_step(ProcessingStep(
                    step_name="line_detection",
                    detector_method="_detect_line_based",
                    status="success",
                    duration_ms=(time.time() - step1_start) * 1000,
                    confidence_score=0.85
                ))
            except Exception as e:
                trace.add_step(ProcessingStep(
                    step_name="line_detection",
                    detector_method="_detect_line_based",
                    status="failed",
                    duration_ms=(time.time() - step1_start) * 1000,
                    error_message=str(e)
                ))
                continue
            
            # Step 2: Cell extraction
            step2_start = time.time()
            try:
                cells = self._extract_cells(line_tables, context.ocr_results)
                trace.add_step(ProcessingStep(
                    step_name="cell_extraction",
                    detector_method="_extract_cells",
                    status="success",
                    duration_ms=(time.time() - step2_start) * 1000,
                    confidence_score=0.88
                ))
            except Exception as e:
                trace.add_step(ProcessingStep(
                    step_name="cell_extraction",
                    detector_method="_extract_cells",
                    status="failed",
                    duration_ms=(time.time() - step2_start) * 1000,
                    error_message=str(e)
                ))
                continue
            
            # Step 3: Content merge
            step3_start = time.time()
            try:
                merged_tables = self._merge_detections(line_tables, context.ocr_results)
                trace.add_step(ProcessingStep(
                    step_name="content_merge",
                    detector_method="_merge_detections",
                    status="success",
                    duration_ms=(time.time() - step3_start) * 1000,
                    confidence_score=0.90
                ))
            except Exception as e:
                trace.add_step(ProcessingStep(
                    step_name="content_merge",
                    detector_method="_merge_detections",
                    status="failed",
                    duration_ms=(time.time() - step3_start) * 1000,
                    error_message=str(e)
                ))
                continue
            
            # Create element with trace
            for table_data in merged_tables:
                element = StructuralElement(
                    element_id=f"table_{uuid.uuid4().hex[:8]}",
                    element_type=ElementType.TABLE,
                    content=TableContent(...),
                    bbox_relative=table_data.bbox,
                    confidence=0.88,
                    processing_trace=trace
                )
                trace.element_id = element.element_id
                trace.set_result("success", 0.88)
                elements.append(element)
        
        return elements
```

### Usage Examples

```python
# Inspect why an element was detected
doc = DocumentResult(...)
table_element = doc.element_index["table_5"]

if table_element.processing_trace:
    print(table_element.processing_trace.to_readable_string())
    # Output:
    # Element: table_5
    # Detected by: TableDetector v1.0.0
    # Status: success (confidence: 88.00%)
    # Duration: 245.3ms
    # Steps:
    #   1. line_detection
    #      Method: _detect_line_based
    #      Status: success
    #      Duration: 120.5ms
    #      Confidence: 85.00%
    #   2. cell_extraction
    #      Method: _extract_cells
    #      Status: success
    #      Duration: 89.2ms
    #      Confidence: 88.00%
    #   3. content_merge
    #      Method: _merge_detections
    #      Status: success
    #      Duration: 35.6ms
    #      Confidence: 90.00%

# Access trace data programmatically
trace_dict = table_element.processing_trace.to_dict()
config_used = trace_dict["config"]
ocr_sources = trace_dict["source_ocr_ids"]
```

### Re-running Processing

```python
# Re-run detection with exact same parameters as original
original_config = StructureDetectionConfig(**element.processing_trace.detector_config)
detector = TableDetector(original_config.table_config)
context = ElementDetectionContext(...)
new_elements = detector.detect(context)

# Results should be identical if input image is same
```

---

## Summary Table

| Decision | Implementation | Benefit |
|----------|-----------------|---------|
| **Coordinates** | Relative % with absolute roots | Scale invariant, hierarchical composition |
| **Element Model** | Single StructuralElement wrapper | Consistent interface, easy to extend |
| **Detection** | Parameterized strategy pattern | Flexible, reproducible, configurable |
| **Hierarchy** | Parent-child IDs + helpers | Arbitrary nesting, fast queries |
| **Traceback** | ProcessingTrace + steps | Reproducible, debuggable, transparent |

