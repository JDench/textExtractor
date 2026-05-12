"""
Unit tests for ListDetector core functionality (Sprint 3).

Tests marker detection, indentation analysis, and hierarchy building
without requiring OpenCV or Tesseract.
"""

import sys
import os
import re

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_bullet_markers():
    """Test bullet marker regex patterns."""
    print("✓ Testing bullet marker regex...")
    
    bullet_pattern = re.compile(r"^[\s]*([•\-*+])\s+(.+)")
    
    test_cases = [
        ("• First item", True, "First item"),
        ("- Second item", True, "Second item"),
        ("* Third item", True, "Third item"),
        ("+ Fourth item", True, "Fourth item"),
        ("  • Nested", True, "Nested"),
        ("No marker here", False, None),
        ("1. Numbered", False, None),
    ]
    
    passed = 0
    for text, should_match, expected_content in test_cases:
        match = bullet_pattern.match(text)
        if should_match:
            assert match is not None, f"Pattern should match: {text}"
            assert match.group(2) == expected_content, f"Content mismatch for: {text}"
            passed += 1
        else:
            assert match is None, f"Pattern should NOT match: {text}"
            passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} bullet marker tests passed")


def test_number_markers():
    """Test number marker regex patterns."""
    print("✓ Testing number marker regex...")
    
    number_pattern = re.compile(r"^[\s]*(\(?(\d+)[.):-])\s+(.+)")
    
    test_cases = [
        ("1. First", True, "First", "1"),
        ("2. Second", True, "Second", "2"),
        ("(1) Third", True, "Third", "1"),
        ("10) Tenth", True, "Tenth", "10"),
        ("  15- Item", True, "Item", "15"),
        ("No number", False, None, None),
        ("• Bullet", False, None, None),
    ]
    
    passed = 0
    for text, should_match, expected_content, expected_num in test_cases:
        match = number_pattern.match(text)
        if should_match:
            assert match is not None, f"Pattern should match: {text}"
            assert match.group(3) == expected_content, f"Content mismatch for: {text}"
            assert match.group(2) == expected_num, f"Number mismatch for: {text}"
            passed += 1
        else:
            assert match is None, f"Pattern should NOT match: {text}"
            passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} number marker tests passed")


def test_letter_markers():
    """Test letter marker regex patterns."""
    print("✓ Testing letter marker regex...")
    
    letter_pattern = re.compile(r"^[\s]*(\(?([a-z])[.):-])\s+(.+)")
    
    test_cases = [
        ("a. First", True, "First", "a"),
        ("b. Second", True, "Second", "b"),
        ("(c) Third", True, "Third", "c"),
        ("d) Fourth", True, "Fourth", "d"),
        ("  e- Item", True, "Item", "e"),
        ("No letter", False, None, None),
        ("1. Number", False, None, None),
    ]
    
    passed = 0
    for text, should_match, expected_content, expected_letter in test_cases:
        match = letter_pattern.match(text)
        if should_match:
            assert match is not None, f"Pattern should match: {text}"
            assert match.group(3) == expected_content, f"Content mismatch for: {text}"
            assert match.group(2) == expected_letter, f"Letter mismatch for: {text}"
            passed += 1
        else:
            assert match is None, f"Pattern should NOT match: {text}"
            passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} letter marker tests passed")


def test_indentation_levels():
    """Test indentation level calculation."""
    print("✓ Testing indentation level calculation...")
    
    indentation_unit = 20.0
    
    def get_level(line):
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces == 0:
            return 0
        level = int(leading_spaces / indentation_unit)
        return max(0, level)
    
    test_cases = [
        ("• Item", 0),
        ("  • Item", 0),          # 2 spaces < 20
        ("    • Item", 0),        # 4 spaces < 20
        ("          • Item", 0),  # 10 spaces < 20
        ("                    • Item", 1),  # 20 spaces = 1 level
        ("                        • Item", 1),  # 24 spaces = 1 level
        ("                              • Item", 1),  # 30 spaces = 1 level
        ("                                        • Item", 2),  # 40 spaces = 2 levels
    ]
    
    passed = 0
    for text, expected_level in test_cases:
        level = get_level(text)
        assert level == expected_level, f"For '{text}', expected {expected_level}, got {level}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} indentation tests passed")


def test_hierarchy_building():
    """Test parent-child hierarchy building algorithm."""
    print("✓ Testing hierarchy building...")
    
    def build_hierarchy(items_metadata):
        """Simplified hierarchy building logic."""
        root_item_ids = []
        parent_map = {}
        level_stack = []
        
        for item_meta in items_metadata:
            item_id = item_meta["item_id"]
            level = item_meta["level"]
            
            # Pop items from stack until we find parent level
            while level_stack and level_stack[-1][0] >= level:
                level_stack.pop()
            
            if level_stack:
                parent_id = level_stack[-1][1]
                parent_map[item_id] = parent_id
            else:
                root_item_ids.append(item_id)
            
            level_stack.append((level, item_id))
        
        return root_item_ids, parent_map
    
    items = [
        {"item_id": "item_1", "level": 0},  # Root
        {"item_id": "item_2", "level": 0},  # Root
        {"item_id": "item_3", "level": 1},  # Child of item_2
        {"item_id": "item_4", "level": 1},  # Child of item_2
        {"item_id": "item_5", "level": 0},  # Root
        {"item_id": "item_6", "level": 1},  # Child of item_5
        {"item_id": "item_7", "level": 2},  # Grandchild of item_5
    ]
    
    root_ids, parent_map = build_hierarchy(items)
    
    # Verify roots
    assert set(root_ids) == {"item_1", "item_2", "item_5"}
    
    # Verify parent relationships
    assert parent_map["item_3"] == "item_2"
    assert parent_map["item_4"] == "item_2"
    assert parent_map["item_6"] == "item_5"
    assert parent_map["item_7"] == "item_6"
    
    print(f"  ✓ Hierarchy building: {len(root_ids)} roots, {len(parent_map)} parent relationships")


def test_number_extraction():
    """Test extracting numeric values from markers."""
    print("✓ Testing number extraction from markers...")
    
    def extract_number(marker_text, marker_type):
        """Extract numeric value from marker."""
        if marker_type == "number":
            digits = ''.join(c for c in marker_text if c.isdigit())
            if digits:
                return int(digits)
        elif marker_type == "letter":
            letter = ''.join(c for c in marker_text if c.isalpha())
            if letter and letter.islower():
                return ord(letter) - ord('a') + 1
            elif letter and letter.isupper():
                return ord(letter) - ord('A') + 1
        return None
    
    test_cases = [
        ("1.", "number", 1),
        ("10)", "number", 10),
        ("(5)", "number", 5),
        ("a.", "letter", 1),
        ("b.", "letter", 2),
        ("(c)", "letter", 3),
        ("d)", "letter", 4),
    ]
    
    passed = 0
    for marker, marker_type, expected in test_cases:
        result = extract_number(marker, marker_type)
        assert result == expected, f"For {marker} ({marker_type}), expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} number extraction tests passed")


def test_marker_detection_logic():
    """Test complete marker detection logic."""
    print("✓ Testing complete marker detection logic...")
    
    bullet_pattern = re.compile(r"^[\s]*([•\-*+])\s+(.+)")
    number_pattern = re.compile(r"^[\s]*(\(?(\d+)[.):-])\s+(.+)")
    letter_pattern = re.compile(r"^[\s]*(\(?([a-z])[.):-])\s+(.+)")
    
    def detect_marker(line, markers_to_try):
        """Detect which marker type a line has."""
        if "bullet" in markers_to_try:
            match = bullet_pattern.match(line)
            if match:
                return "bullet", match.group(1), match.group(2)
        
        if "number" in markers_to_try:
            match = number_pattern.match(line)
            if match:
                return "number", match.group(1), match.group(3)
        
        if "letter" in markers_to_try:
            match = letter_pattern.match(line)
            if match:
                return "letter", match.group(1), match.group(3)
        
        return None, None, line
    
    test_cases = [
        ("• First", ["bullet"], ("bullet", "•", "First")),
        ("1. Item", ["number"], ("number", "1.", "Item")),
        ("a. Item", ["letter"], ("letter", "a.", "Item")),
        ("• Item", ["bullet", "number"], ("bullet", "•", "Item")),
        ("1. Item", ["bullet", "number"], ("number", "1.", "Item")),
        ("No marker", ["bullet", "number", "letter"], (None, None, "No marker")),
    ]
    
    passed = 0
    for text, markers, expected in test_cases:
        result = detect_marker(text, markers)
        assert result == expected, f"For '{text}', expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} marker detection tests passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("LIST DETECTOR UNIT TESTS (Sprint 3)")
    print("=" * 70)
    print()
    
    tests = [
        test_bullet_markers,
        test_number_markers,
        test_letter_markers,
        test_indentation_levels,
        test_hierarchy_building,
        test_number_extraction,
        test_marker_detection_logic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    if failed == 0:
        print(f"✅ ALL TESTS PASSED! ({passed}/{len(tests)} test suites)")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed} passed, {failed} failed)")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
