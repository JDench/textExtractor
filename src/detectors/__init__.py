"""
Detectors Package - Structure detection modules.

This package contains specialized detectors for different element types:
- text_detector: Headings, paragraphs, block quotes
- list_detector: Lists and nested structures
- table_detector: Tables with cells and merged cells (line-based, Hough transform)
- content_table_detector: Tables without visible grid lines (spatial analysis fallback)
- (coming) figure_detector: Figures and captions
- (coming) formula_detector: Formulas and equations
- And more...

All detectors follow the same pattern:
1. Config class for parameterization
2. Detector class with detect/extract method
3. ProcessingTrace for reproducibility
"""

from .text_detector import TextDetector, TextDetectorConfig, TextDetectionTrace
from .list_detector import ListDetector, ListDetectorConfig, ListDetectionTrace
from .table_detector import TableDetector, TableDetectorConfig, TableDetectionTrace
from .content_table_detector import (
    ContentTableDetector,
    ContentTableDetectorConfig,
    ContentTableDetectionTrace,
)

__all__ = [
    "TextDetector",
    "TextDetectorConfig",
    "TextDetectionTrace",
    "ListDetector",
    "ListDetectorConfig",
    "ListDetectionTrace",
    "TableDetector",
    "TableDetectorConfig",
    "TableDetectionTrace",
    "ContentTableDetector",
    "ContentTableDetectorConfig",
    "ContentTableDetectionTrace",
]
