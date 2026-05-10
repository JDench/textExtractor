"""
Detectors Package - Structure detection modules.

This package contains specialized detectors for different element types:
- text_detector: Headings, paragraphs, block quotes
- (coming) list_detector: Lists and nested structures
- (coming) table_detector: Tables with cells and merged cells
- (coming) figure_detector: Figures and captions
- (coming) formula_detector: Formulas and equations
- And more...

All detectors follow the same pattern:
1. Config class for parameterization
2. Detector class with detect/extract method
3. ProcessingTrace for reproducibility
"""
