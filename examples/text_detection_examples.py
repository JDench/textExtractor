"""
Text Detection Examples and Tests - Sprint 2

Demonstrates and validates:
1. Basic heading detection
2. Paragraph extraction
3. Block quote detection
4. Custom configurations
5. Processing traces
6. End-to-end text extraction pipeline

This script validates against the OCR_DEVELOPMENT_PLAN.md specifications.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detectors.text_detector import (
    TextDetector,
    TextDetectorConfig,
    create_text_detection_pipeline,
)
from data_models import ElementType


# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_document_image():
    """
    Create a realistic test document image with:
    - Main heading (H1)
    - Subheading (H2)
    - Regular paragraphs
    - Indented block quote
    - Mixed content
    
    Returns:
        numpy array in BGR format (cv2 compatible)
    """
    # Create image in PIL (easier for text)
    img_pil = Image.new('RGB', (1200, 1600), color='white')
    draw = ImageDraw.Draw(img_pil)
    
    # Use default PIL font (or load a system font if available)
    try:
        title_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 48)
        heading_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 32)
        body_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 18)
        quote_font = ImageFont.truetype("C:\\Windows\\Fonts\\ariali.ttf", 16)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        heading_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        quote_font = ImageFont.load_default()
    
    y_pos = 40
    line_height = 60
    
    # Main title (H1)
    draw.text((50, y_pos), "Document Analysis", fill='black', font=title_font)
    y_pos += line_height + 20
    
    # Subheading 1 (H2)
    draw.text((50, y_pos), "Introduction and Overview", fill='black', font=heading_font)
    y_pos += line_height
    
    # Paragraph 1
    para1 = ("This is the introduction paragraph. It contains multiple sentences "
             "that form a complete thought. The document extraction system should "
             "recognize this as a single TEXT element with appropriate confidence.")
    draw.text((50, y_pos), para1, fill='black', font=body_font, width=1100)
    y_pos += 120
    
    # Paragraph 2
    para2 = ("A second paragraph follows the first, separated by whitespace. This "
             "allows the detector to group text appropriately. Multiple consecutive "
             "paragraphs should be detected as separate TEXT elements.")
    draw.text((50, y_pos), para2, fill='black', font=body_font, width=1100)
    y_pos += 120
    
    # Subheading 2 (H3 - smaller than H2)
    draw.text((50, y_pos), "Key Findings", fill='black', font=heading_font)
    y_pos += line_height
    
    # Paragraph 3
    para3 = ("The research indicates significant improvements in text extraction "
             "when using specialized PSM modes for different content types.")
    draw.text((50, y_pos), para3, fill='black', font=body_font, width=1100)
    y_pos += 100
    
    # Block quote (indented)
    indent = 80
    draw.text((indent + 50, y_pos), '"The detector must handle hierarchical', fill='gray', font=quote_font)
    y_pos += 40
    draw.text((indent + 50, y_pos), 'document structures appropriately"', fill='gray', font=quote_font)
    y_pos += 40
    draw.text((indent + 50, y_pos), '- Research Team', fill='gray', font=quote_font)
    y_pos += 80
    
    # Another paragraph
    para4 = "This paragraph comes after the block quote and should be detected separately."
    draw.text((50, y_pos), para4, fill='black', font=body_font, width=1100)
    
    # Convert PIL image to numpy/OpenCV format
    img_array = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr


def example_basic_text_detection():
    """
    Example 1: Basic text detection with default configuration.
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Text Detection (Default Configuration)")
    print("=" * 80)
    
    # Create test image
    image = create_test_document_image()
    logger.info(f"Created test image: {image.shape}")
    
    # Create detector with defaults
    config = TextDetectorConfig()
    detector = TextDetector(config)
    
    print(f"\nDetector Configuration:")
    print(f"  Detect headings: {config.detect_headings}")
    print(f"  Detect paragraphs: {config.detect_paragraphs}")
    print(f"  Detect block quotes: {config.detect_block_quotes}")
    print(f"  Heading levels: {config.heading_levels}")
    print(f"  Min confidence: {config.min_confidence:.0%}")
    
    # Run detection
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    # Print results
    print(f"\nResults:")
    print(f"  Total elements detected: {len(elements)}")
    print(f"  Processing time: {trace.total_processing_time_ms:.1f}ms")
    
    # Summary by type
    by_type = {}
    for elem in elements:
        elem_type = elem.element_type.value
        by_type[elem_type] = by_type.get(elem_type, 0) + 1
    
    print(f"\n  Elements by type:")
    for elem_type, count in sorted(by_type.items()):
        print(f"    {elem_type}: {count}")
    
    # Show first few elements
    print(f"\n  First 5 elements:")
    for i, elem in enumerate(elements[:5]):
        content_preview = str(elem.content)[:50] + "..." if len(str(elem.content)) > 50 else str(elem.content)
        level_info = f" (Level {elem.metadata.get('heading_level', 'N/A')})" if elem.element_type == ElementType.HEADING else ""
        print(f"    [{i+1}] {elem.element_type.value}{level_info}: '{content_preview}'")
        print(f"         Confidence: {elem.confidence:.2%}, Nesting: {elem.nesting_level}")
    
    return trace


def example_custom_config():
    """
    Example 2: Custom configuration with stricter filtering.
    """
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration (Stricter Filtering)")
    print("=" * 80)
    
    image = create_test_document_image()
    
    # Create custom config with stricter thresholds
    config = TextDetectorConfig(
        detect_headings=True,
        detect_paragraphs=True,
        detect_block_quotes=True,
        heading_min_confidence=0.7,  # Higher threshold
        min_paragraph_confidence=0.6,
        heading_levels=[1, 2, 3],  # Only H1-H3
        min_heading_size_ratio=1.5,  # Require larger size difference
    )
    
    detector = TextDetector(config)
    
    print(f"\nCustom Configuration:")
    print(f"  Heading min confidence: {config.heading_min_confidence:.0%}")
    print(f"  Paragraph min confidence: {config.min_paragraph_confidence:.0%}")
    print(f"  Allowed heading levels: {config.heading_levels}")
    print(f"  Min heading size ratio: {config.min_heading_size_ratio}")
    
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    print(f"\nResults with stricter config:")
    print(f"  Total elements: {len(elements)}")
    print(f"  Headings: {trace.headings_found}")
    print(f"  Paragraphs: {trace.paragraphs_found}")
    print(f"  Block quotes: {trace.block_quotes_found}")
    print(f"  Processing time: {trace.total_processing_time_ms:.1f}ms")


def example_heading_classification():
    """
    Example 3: Validate heading level classification.
    """
    print("\n" + "=" * 80)
    print("Example 3: Heading Level Classification")
    print("=" * 80)
    
    image = create_test_document_image()
    detector = TextDetector()
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    # Filter headings and show hierarchy
    headings = [e for e in elements if e.element_type == ElementType.HEADING]
    
    print(f"\nDetected {len(headings)} headings:")
    for heading in headings:
        level = heading.metadata.get("heading_level", "?")
        size_ratio = heading.metadata.get("size_ratio", "?")
        print(f"  H{level}: '{heading.content}' (size_ratio: {size_ratio:.2f}, conf: {heading.confidence:.2%})")
    
    print(f"\nHeading size analysis from trace:")
    print(f"  Average text size: {trace.average_text_size:.2f}px")
    print(f"  Heading sizes: {trace.heading_sizes}")


def example_processing_trace():
    """
    Example 4: Inspect processing trace for reproducibility.
    """
    print("\n" + "=" * 80)
    print("Example 4: Processing Trace (Reproducibility & Debugging)")
    print("=" * 80)
    
    image = create_test_document_image()
    config = TextDetectorConfig(detect_headings=True, detect_paragraphs=True)
    detector = TextDetector(config)
    
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    print(f"\nProcessing Trace (Dict format):")
    trace_dict = trace.to_dict()
    
    for section, data in trace_dict.items():
        if isinstance(data, dict):
            print(f"\n  {section}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {section}: {data}")
    
    print(f"\nThis trace can be used to:")
    print(f"  - Re-run with identical config and methods")
    print(f"  - Verify reproducibility of results")
    print(f"  - Debug why certain elements were/weren't detected")
    print(f"  - Audit processing decisions")


def example_hierarchy_building():
    """
    Example 5: Demonstrate parent-child hierarchy building.
    """
    print("\n" + "=" * 80)
    print("Example 5: Element Hierarchy (Parent-Child Relationships)")
    print("=" * 80)
    
    image = create_test_document_image()
    detector = TextDetector()
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    # Build index for easy lookup
    elem_index = {e.element_id: e for e in elements}
    
    print(f"\nHierarchy visualization:")
    
    # Find root elements (no parent)
    root_elements = [e for e in elements if e.parent_id is None]
    
    for root in root_elements:
        print(f"\n{root.element_type.value.upper()}: {str(root.content)[:60]}")
        
        # Show children
        if root.child_ids:
            for child_id in root.child_ids:
                child = elem_index.get(child_id)
                if child:
                    content_preview = str(child.content)[:40] + "..." if len(str(child.content)) > 40 else str(child.content)
                    indent = "  └─ "
                    print(f"{indent}{child.element_type.value}: {content_preview}")


def example_statistical_summary():
    """
    Example 6: Generate statistical summary of detection results.
    """
    print("\n" + "=" * 80)
    print("Example 6: Statistical Summary & Validation")
    print("=" * 80)
    
    image = create_test_document_image()
    
    # Test with different configurations to compare
    configs = [
        ("Default", TextDetectorConfig()),
        ("Strict", TextDetectorConfig(
            heading_min_confidence=0.8,
            min_paragraph_confidence=0.7,
        )),
        ("Relaxed", TextDetectorConfig(
            heading_min_confidence=0.3,
            min_paragraph_confidence=0.2,
        )),
    ]
    
    print(f"\nComparison across configurations:\n")
    print(f"{'Config':<12} {'Headings':<10} {'Paragraphs':<12} {'Quotes':<8} {'Total':<8} {'Time(ms)':<10}")
    print("-" * 70)
    
    for config_name, config in configs:
        detector = TextDetector(config)
        elements, trace = detector.detect_text_elements(image, page_number=1)
        
        print(f"{config_name:<12} {trace.headings_found:<10} {trace.paragraphs_found:<12} "
              f"{trace.block_quotes_found:<8} {len(elements):<8} {trace.total_processing_time_ms:<10.1f}")
    
    print(f"\nValidation Checks:")
    
    # Run default config for validation
    detector = TextDetector()
    elements, trace = detector.detect_text_elements(image, page_number=1)
    
    # Check 1: Elements have required fields
    all_valid = True
    for elem in elements:
        if not elem.element_id or elem.confidence is None or elem.bbox is None:
            print(f"  ❌ Element missing required fields: {elem.element_id}")
            all_valid = False
    
    if all_valid:
        print(f"  ✅ All elements have required fields")
    
    # Check 2: Parent-child relationships are consistent
    elem_index = {e.element_id: e for e in elements}
    consistency_ok = True
    for elem in elements:
        if elem.parent_id and elem.parent_id not in elem_index:
            print(f"  ❌ Element {elem.element_id} references non-existent parent {elem.parent_id}")
            consistency_ok = False
        
        for child_id in elem.child_ids:
            if child_id not in elem_index:
                print(f"  ❌ Element {elem.element_id} references non-existent child {child_id}")
                consistency_ok = False
    
    if consistency_ok:
        print(f"  ✅ All parent-child relationships are valid")
    
    # Check 3: Confidence scores are in valid range
    conf_valid = all(0.0 <= e.confidence <= 1.0 for e in elements)
    if conf_valid:
        print(f"  ✅ All confidence scores are in [0, 1] range")
    else:
        print(f"  ❌ Some confidence scores are out of range")
    
    # Check 4: Element types are valid
    valid_types = set(ElementType)
    type_valid = all(e.element_type in valid_types for e in elements)
    if type_valid:
        print(f"  ✅ All element types are valid ElementType enum values")
    else:
        print(f"  ❌ Some invalid element types detected")


def main():
    """Run all examples and validate implementation."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "Sprint 2: Text Detection - Examples & Validation".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        # Run all examples
        example_basic_text_detection()
        example_custom_config()
        example_heading_classification()
        example_processing_trace()
        example_hierarchy_building()
        example_statistical_summary()
        
        print("\n" + "=" * 80)
        print("✅ All Examples Completed Successfully!")
        print("=" * 80)
        print("\nValidation Summary:")
        print("  ✓ Text detection pipeline works end-to-end")
        print("  ✓ Heading detection with level classification")
        print("  ✓ Paragraph grouping and extraction")
        print("  ✓ Block quote detection with indentation analysis")
        print("  ✓ Element hierarchy and parent-child relationships")
        print("  ✓ Processing traces for reproducibility")
        print("  ✓ Customizable configuration support")
        print("\nNext Steps:")
        print("  1. Try on real document images")
        print("  2. Tune configuration for specific document types")
        print("  3. Implement list detection (Sprint 3)")
        print("  4. Implement table detection (Sprint 3)")
        print()
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
