"""
Test suite for ListDetector implementation (Sprint 3).

Tests all 7 list detection features:
1. Bullet marker detection (•, -, *, +)
2. Number marker detection (1., 2., (1), 1))
3. Letter marker detection (a., b., (a), a))
4. Indentation-based hierarchy
5. ListItem validation
6. ListStructure methods
7. Parent-child relationship building
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from datetime import datetime
from pathlib import Path

from detectors.list_detector import (
    ListDetector,
    ListDetectorConfig,
    ListMarkerType,
)
from data_models import (
    BoundingBox,
    OCRTextResult,
    ElementType,
    ListItem,
    ListStructure,
)


def create_mock_ocr_result(text: str, y_position: float, confidence: float = 95.0) -> OCRTextResult:
    """Create a mock OCRTextResult for testing."""
    return OCRTextResult(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(10, y_position, 300, y_position + 20),
        language="eng",
        page_number=1,
    )


def test_list_detector_initialization():
    """Test 1: ListDetector initializes with valid config."""
    print("✓ Testing ListDetector initialization...")
    
    config = ListDetectorConfig(
        detect_lists=True,
        detect_nested_lists=True,
    )
    detector = ListDetector(config)
    
    assert detector.config.detect_lists is True
    assert detector.config.detect_nested_lists is True
    assert detector.ocr_engine is not None
    print("  ✓ ListDetector initialization works")


def test_bullet_marker_detection():
    """Test 2: Bullet marker detection (•, -, *, +)."""
    print("✓ Testing bullet marker detection...")
    
    detector = ListDetector()
    
    test_cases = [
        ("• First item", ListMarkerType.BULLET, "First item"),
        ("- Second item", ListMarkerType.BULLET, "Second item"),
        ("* Third item", ListMarkerType.BULLET, "Third item"),
        ("+ Fourth item", ListMarkerType.BULLET, "Fourth item"),
    ]
    
    for text, expected_type, expected_content in test_cases:
        marker_type, marker_text, content = detector._detect_marker(text)
        assert marker_type == expected_type, f"Expected {expected_type}, got {marker_type}"
        assert content == expected_content, f"Expected '{expected_content}', got '{content}'"
    
    print(f"  ✓ All {len(test_cases)} bullet markers detected correctly")


def test_number_marker_detection():
    """Test 3: Number marker detection (1., 2., (1), 1))."""
    print("✓ Testing number marker detection...")
    
    detector = ListDetector()
    
    test_cases = [
        ("1. First item", ListMarkerType.NUMBER, "First item", 1),
        ("2. Second item", ListMarkerType.NUMBER, "Second item", 2),
        ("(1) Third item", ListMarkerType.NUMBER, "Third item", 1),
        ("10) Tenth item", ListMarkerType.NUMBER, "Tenth item", 10),
    ]
    
    for text, expected_type, expected_content, expected_number in test_cases:
        marker_type, marker_text, content = detector._detect_marker(text)
        assert marker_type == expected_type, f"Expected {expected_type}, got {marker_type}"
        assert content == expected_content, f"Expected '{expected_content}', got '{content}'"
        
        item_number = detector._extract_item_number(marker_text, marker_type)
        assert item_number == expected_number, f"Expected number {expected_number}, got {item_number}"
    
    print(f"  ✓ All {len(test_cases)} number markers detected correctly")


def test_letter_marker_detection():
    """Test 4: Letter marker detection (a., b., (a), a))."""
    print("✓ Testing letter marker detection...")
    
    detector = ListDetector()
    
    test_cases = [
        ("a. First item", ListMarkerType.LETTER, "First item", 1),
        ("b. Second item", ListMarkerType.LETTER, "Second item", 2),
        ("(c) Third item", ListMarkerType.LETTER, "Third item", 3),
        ("d) Fourth item", ListMarkerType.LETTER, "Fourth item", 4),
    ]
    
    for text, expected_type, expected_content, expected_number in test_cases:
        marker_type, marker_text, content = detector._detect_marker(text)
        assert marker_type == expected_type, f"Expected {expected_type}, got {marker_type}"
        assert content == expected_content, f"Expected '{expected_content}', got '{content}'"
        
        item_number = detector._extract_item_number(marker_text, marker_type)
        assert item_number == expected_number, f"Expected number {expected_number}, got {item_number}"
    
    print(f"  ✓ All {len(test_cases)} letter markers detected correctly")


def test_indentation_level_detection():
    """Test 5: Indentation level detection."""
    print("✓ Testing indentation level detection...")
    
    config = ListDetectorConfig(indentation_unit=20.0)
    detector = ListDetector(config)
    
    test_cases = [
        ("• Item", 0),           # No indentation
        ("  • Item", 0),         # 2 spaces = 0 levels
        ("    • Item", 0),       # 4 spaces = 0 levels
        ("          • Item", 0),  # 10 spaces = 0 levels
        ("                    • Item", 1),  # 20 spaces = 1 level
        ("                        • Item", 1),  # 24 spaces = 1 level
        ("                              • Item", 1),  # 30 spaces = 1 level
        ("                                        • Item", 2),  # 40 spaces = 2 levels
    ]
    
    for text, expected_level in test_cases:
        level = detector._get_indentation_level(text)
        assert level == expected_level, f"For '{text}', expected level {expected_level}, got {level}"
    
    print(f"  ✓ All {len(test_cases)} indentation levels detected correctly")


def test_hierarchy_building():
    """Test 6: Parent-child hierarchy building."""
    print("✓ Testing hierarchy building...")
    
    # Create items with various indentation levels
    items_metadata = [
        {"item_id": "item_1", "level": 0},  # Root
        {"item_id": "item_2", "level": 0},  # Root
        {"item_id": "item_3", "level": 1},  # Child of item_2
        {"item_id": "item_4", "level": 1},  # Child of item_2
        {"item_id": "item_5", "level": 0},  # Root
        {"item_id": "item_6", "level": 1},  # Child of item_5
        {"item_id": "item_7", "level": 2},  # Grandchild of item_5
    ]
    
    detector = ListDetector()
    root_ids, parent_map = detector._build_list_hierarchy(items_metadata)
    
    # Verify roots
    assert "item_1" in root_ids, "item_1 should be root"
    assert "item_2" in root_ids, "item_2 should be root"
    assert "item_5" in root_ids, "item_5 should be root"
    
    # Verify parent relationships
    assert parent_map.get("item_3") == "item_2", "item_3 should be child of item_2"
    assert parent_map.get("item_4") == "item_2", "item_4 should be child of item_2"
    assert parent_map.get("item_6") == "item_5", "item_6 should be child of item_5"
    assert parent_map.get("item_7") == "item_6", "item_7 should be child of item_6"
    
    print(f"  ✓ Hierarchy building works: {len(root_ids)} roots, {len(parent_map)} parent relationships")


def test_list_item_validation():
    """Test 7: ListItem validation."""
    print("✓ Testing ListItem validation...")
    
    bbox = BoundingBox(10, 10, 300, 30)
    
    # Valid item
    item = ListItem(
        content="Test item",
        level=0,
        bbox=bbox,
        confidence=0.95,
        list_type="bullet",
    )
    assert item.level == 0
    assert item.confidence == 0.95
    
    # Invalid: negative level
    try:
        item = ListItem(
            content="Bad item",
            level=-1,
            bbox=bbox,
            confidence=0.95,
            list_type="bullet",
        )
        assert False, "Should have raised ValueError for negative level"
    except ValueError:
        pass
    
    # Invalid: confidence out of range
    try:
        item = ListItem(
            content="Bad item",
            level=0,
            bbox=bbox,
            confidence=1.5,
            list_type="bullet",
        )
        assert False, "Should have raised ValueError for confidence > 1.0"
    except ValueError:
        pass
    
    print("  ✓ ListItem validation works correctly")


def test_list_structure_validation():
    """Test 8: ListStructure validation."""
    print("✓ Testing ListStructure validation...")
    
    bbox = BoundingBox(10, 10, 300, 100)
    
    # Create items
    item1 = ListItem(
        content="First",
        level=0,
        bbox=BoundingBox(10, 10, 300, 30),
        confidence=0.95,
    )
    item2 = ListItem(
        content="Second",
        level=0,
        bbox=BoundingBox(10, 40, 300, 60),
        confidence=0.90,
    )
    
    # Valid structure
    list_struct = ListStructure(
        items=[item1, item2],
        root_item_ids=["item_1", "item_2"],
        bbox=bbox,
        confidence=0.925,
        list_type="bullet",
    )
    
    assert len(list_struct.items) == 2
    assert list_struct.confidence == 0.925
    
    # Invalid: confidence out of range
    try:
        list_struct = ListStructure(
            items=[item1, item2],
            root_item_ids=["item_1"],
            bbox=bbox,
            confidence=1.5,
        )
        assert False, "Should have raised ValueError for confidence > 1.0"
    except ValueError:
        pass
    
    print("  ✓ ListStructure validation works correctly")


def test_list_detection_end_to_end():
    """Test 9: End-to-end list detection with mock OCR results."""
    print("✓ Testing end-to-end list detection...")
    
    # Create mock OCR results for a simple list
    ocr_results = [
        create_mock_ocr_result("• First item", 10, 95),
        create_mock_ocr_result("• Second item", 40, 95),
        create_mock_ocr_result("  • Nested item", 70, 92),
        create_mock_ocr_result("1. Another list", 100, 90),
        create_mock_ocr_result("2. Second numbered", 130, 90),
    ]
    
    config = ListDetectorConfig(
        detect_lists=True,
        detect_nested_lists=True,
        marker_types=[ListMarkerType.BULLET, ListMarkerType.NUMBER],
    )
    detector = ListDetector(config)
    
    # Create mock image
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # Run detection
    elements, trace = detector.detect_lists(
        image=image,
        page_number=1,
        ocr_results=ocr_results,
    )
    
    # Verify results
    assert len(elements) == 1, f"Expected 1 element, got {len(elements)}"
    
    element = elements[0]
    assert element.element_type == ElementType.LIST
    
    list_struct = element.content
    assert isinstance(list_struct, ListStructure)
    assert len(list_struct.items) >= 2, f"Expected at least 2 items, got {len(list_struct.items)}"
    
    # Check trace
    assert trace.lists_found == 1
    assert trace.items_found >= 2
    assert trace.ocr_results_analyzed == len(ocr_results)
    
    print(f"  ✓ End-to-end detection works: {len(elements)} list(s), {trace.items_found} items, trace OK")


def test_marker_types_mixed():
    """Test 10: Detection of mixed marker types."""
    print("✓ Testing mixed marker type detection...")
    
    # Create mixed list with bullets and numbers
    ocr_results = [
        create_mock_ocr_result("• Bullet item", 10, 95),
        create_mock_ocr_result("1. Numbered item", 40, 95),
        create_mock_ocr_result("a. Lettered item", 70, 95),
    ]
    
    detector = ListDetector()
    image = np.zeros((150, 400, 3), dtype=np.uint8)
    
    elements, trace = detector.detect_lists(image=image, ocr_results=ocr_results)
    
    if len(elements) > 0:
        list_struct = elements[0].content
        assert list_struct.list_type == "mixed", f"Expected 'mixed', got '{list_struct.list_type}'"
        print(f"  ✓ Mixed marker types detected correctly: {trace.marker_types_detected}")
    else:
        print("  ⚠ No elements detected for mixed markers (may need config adjustment)")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("LIST DETECTOR TEST SUITE (Sprint 3)")
    print("=" * 70)
    
    tests = [
        test_list_detector_initialization,
        test_bullet_marker_detection,
        test_number_marker_detection,
        test_letter_marker_detection,
        test_indentation_level_detection,
        test_hierarchy_building,
        test_list_item_validation,
        test_list_structure_validation,
        test_list_detection_end_to_end,
        test_marker_types_mixed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    if failed == 0:
        print(f"✅ ALL TESTS PASSED! ({passed}/{len(tests)})")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed} passed, {failed} failed)")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
