"""
Detectors Package - Structure detection modules.

All detectors follow the Config + Detector + Trace pattern:
1. Config dataclass    — parameterizes the detector
2. Detector class      — detect() method returns (List[StructuralElement], Trace)
3. Trace dataclass     — records timing, counts, and config for reproducibility

Sprint 1-2
----------
- text_detector         : TEXT, HEADING, BLOCK_QUOTE

Sprint 3
--------
- list_detector         : LIST (marker-based, nested hierarchy)
- table_detector        : TABLE (line-based, Hough transform)
- content_table_detector: TABLE (content/spatial-based, borderless fallback)

Sprint 4
--------
- header_footer_detector: HEADER, FOOTER, PAGE_NUMBER + date annotation
- formula_detector      : FORMULA, EQUATION (+ optional pix2tex/sympy layers)
- figure_detector       : FIGURE, CAPTION (contour-based + caption-pattern linking)
- annotation_detector   : ANNOTATION (highlight/underline/strikethrough)

Sprint 5
--------
- watermark_detector    : WATERMARK (vocabulary + span + pixel-opacity signals)
- barcode_detector      : BARCODE (pyzbar primary, cv2.QRCodeDetector fallback)
- code_block_detector   : CODE_BLOCK (visual gray-box + structural alignment)
- reference_detector    : REFERENCE citation/footnote/bibliography + cross-linking

Sprint 6
--------
- toc_detector          : TABLE_OF_CONTENTS (heading scan + entry pattern matching)
- index_detector        : INDEX (index heading scan + entry/sub-entry parsing)
See also src/hierarchy_builder.py — HierarchyBuilder post-processor
"""

from .text_detector import TextDetector, TextDetectorConfig, TextDetectionTrace
from .list_detector import ListDetector, ListDetectorConfig, ListDetectionTrace
from .table_detector import TableDetector, TableDetectorConfig, TableDetectionTrace
from .content_table_detector import (
    ContentTableDetector,
    ContentTableDetectorConfig,
    ContentTableDetectionTrace,
)
from .header_footer_detector import (
    HeaderFooterDetector,
    HeaderFooterDetectorConfig,
    HeaderFooterDetectionTrace,
    DateInfo,
)
from .formula_detector import (
    FormulaDetector,
    FormulaDetectorConfig,
    FormulaDetectionTrace,
)
from .figure_detector import (
    FigureDetector,
    FigureDetectorConfig,
    FigureDetectionTrace,
)
from .annotation_detector import (
    AnnotationDetector,
    AnnotationDetectorConfig,
    AnnotationDetectionTrace,
)
from .watermark_detector import (
    WatermarkDetector,
    WatermarkDetectorConfig,
    WatermarkDetectionTrace,
)
from .barcode_detector import (
    BarcodeDetector,
    BarcodeDetectorConfig,
    BarcodeDetectionTrace,
)
from .code_block_detector import (
    CodeBlockDetector,
    CodeBlockDetectorConfig,
    CodeBlockDetectionTrace,
)
from .reference_detector import (
    ReferenceDetector,
    ReferenceDetectorConfig,
    ReferenceDetectionTrace,
)
from .toc_detector import (
    TOCDetector,
    TOCDetectorConfig,
    TOCDetectionTrace,
)
from .index_detector import (
    IndexDetector,
    IndexDetectorConfig,
    IndexDetectionTrace,
)

__all__ = [
    # Sprint 2
    "TextDetector",
    "TextDetectorConfig",
    "TextDetectionTrace",
    # Sprint 3
    "ListDetector",
    "ListDetectorConfig",
    "ListDetectionTrace",
    "TableDetector",
    "TableDetectorConfig",
    "TableDetectionTrace",
    "ContentTableDetector",
    "ContentTableDetectorConfig",
    "ContentTableDetectionTrace",
    # Sprint 4
    "HeaderFooterDetector",
    "HeaderFooterDetectorConfig",
    "HeaderFooterDetectionTrace",
    "DateInfo",
    "FormulaDetector",
    "FormulaDetectorConfig",
    "FormulaDetectionTrace",
    # Sprint 4 (continued)
    "FigureDetector",
    "FigureDetectorConfig",
    "FigureDetectionTrace",
    "AnnotationDetector",
    "AnnotationDetectorConfig",
    "AnnotationDetectionTrace",
    # Sprint 5
    "WatermarkDetector",
    "WatermarkDetectorConfig",
    "WatermarkDetectionTrace",
    "BarcodeDetector",
    "BarcodeDetectorConfig",
    "BarcodeDetectionTrace",
    "CodeBlockDetector",
    "CodeBlockDetectorConfig",
    "CodeBlockDetectionTrace",
    "ReferenceDetector",
    "ReferenceDetectorConfig",
    "ReferenceDetectionTrace",
    # Sprint 6
    "TOCDetector",
    "TOCDetectorConfig",
    "TOCDetectionTrace",
    "IndexDetector",
    "IndexDetectorConfig",
    "IndexDetectionTrace",
]
