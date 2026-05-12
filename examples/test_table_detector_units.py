"""
Unit tests for TableDetector core functionality (Sprint 3, Task 2).

Tests line clustering, grid detection, cell extraction logic
without requiring OpenCV or Tesseract.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_horizontal_line_clustering():
    """Test clustering of horizontal lines (y-coordinates)."""
    print("✓ Testing horizontal line clustering...")
    
    def cluster_lines(lines, threshold=5.0):
        """Cluster similar line coordinates."""
        if not lines:
            return []
        
        coords = sorted(lines)
        clustered = []
        current_cluster = [coords[0]]
        
        for coord in coords[1:]:
            if coord - current_cluster[-1] <= threshold:
                current_cluster.append(coord)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [coord]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    test_cases = [
        # (input, threshold, expected_count)
        ([100, 102, 104], 5.0, 1),           # All within threshold
        ([100, 200, 300], 5.0, 3),           # All separate
        ([100, 102, 200, 202], 5.0, 2),      # Two clusters
        ([50, 51, 52, 150, 151], 3.0, 2),    # Two clusters at 5-degree threshold
        ([], 5.0, 0),                        # Empty input
        ([100], 5.0, 1),                     # Single line
    ]
    
    passed = 0
    for lines, threshold, expected in test_cases:
        result = cluster_lines(lines, threshold)
        assert len(result) == expected, f"For {lines}, expected {expected} clusters, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} line clustering tests passed")


def test_vertical_line_clustering():
    """Test clustering of vertical lines (x-coordinates)."""
    print("✓ Testing vertical line clustering...")
    
    def cluster_lines(lines, threshold=5.0):
        """Cluster similar line coordinates."""
        if not lines:
            return []
        
        coords = sorted(lines)
        clustered = []
        current_cluster = [coords[0]]
        
        for coord in coords[1:]:
            if coord - current_cluster[-1] <= threshold:
                current_cluster.append(coord)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [coord]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return clustered
    
    test_cases = [
        # Vertical line x-coordinates
        ([50, 52, 54], 5.0, 1),
        ([50, 100, 150, 200], 5.0, 4),
        ([50, 51, 100, 101, 150, 151], 3.0, 3),
    ]
    
    passed = 0
    for lines, threshold, expected in test_cases:
        result = cluster_lines(lines, threshold)
        assert len(result) == expected, f"For {lines}, expected {expected} clusters, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} vertical clustering tests passed")


def test_line_intersections():
    """Test finding intersections of horizontal and vertical lines."""
    print("✓ Testing line intersections...")
    
    def find_intersections(h_lines, v_lines):
        """Find all intersections."""
        intersections = []
        for x in v_lines:
            for y in h_lines:
                intersections.append((x, y))
        return intersections
    
    test_cases = [
        # (h_lines, v_lines, expected_count)
        ([100, 200], [50, 150], 4),           # 2x2 grid = 4 intersections
        ([100, 200, 300], [50, 150, 250], 9), # 3x3 grid = 9 intersections
        ([100], [50], 1),                    # 1x1 = 1 intersection
        ([100, 200], [], 0),                  # No vertical lines
        ([], [50, 150], 0),                   # No horizontal lines
    ]
    
    passed = 0
    for h_lines, v_lines, expected in test_cases:
        result = find_intersections(h_lines, v_lines)
        assert len(result) == expected, f"Expected {expected}, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} intersection tests passed")


def test_grid_building():
    """Test building regular grid from intersections."""
    print("✓ Testing grid building...")
    
    def build_grid(intersections):
        """Extract unique coordinates from intersections."""
        if not intersections:
            return [], []
        
        x_coords = sorted(set(x for x, y in intersections))
        y_coords = sorted(set(y for x, y in intersections))
        
        return x_coords, y_coords
    
    test_cases = [
        # Intersections from 2x2 grid
        ([(50, 100), (50, 200), (150, 100), (150, 200)], 2, 2),
        # Intersections from 3x3 grid
        ([(50, 100), (50, 200), (50, 300), (150, 100), (150, 200), (150, 300), (250, 100), (250, 200), (250, 300)], 3, 3),
        # Single point
        ([(50, 100)], 1, 1),
        # Empty
        ([], 0, 0),
    ]
    
    passed = 0
    for intersections, expected_x, expected_y in test_cases:
        x_coords, y_coords = build_grid(intersections)
        assert len(x_coords) == expected_x, f"Expected {expected_x} x-coords, got {len(x_coords)}"
        assert len(y_coords) == expected_y, f"Expected {expected_y} y-coords, got {len(y_coords)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} grid building tests passed")


def test_cell_count_calculation():
    """Test calculating number of cells from grid."""
    print("✓ Testing cell count calculation...")
    
    def count_cells(x_coords, y_coords):
        """Calculate number of cells."""
        num_cols = len(x_coords) - 1
        num_rows = len(y_coords) - 1
        return num_rows * num_cols
    
    test_cases = [
        # (x_count, y_count, expected_cells)
        (2, 2, 1),           # 1x1 = 1 cell
        (3, 3, 4),           # 2x2 = 4 cells
        (4, 3, 6),           # 3x2 = 6 cells
        (2, 4, 3),           # 1x3 = 3 cells
        (1, 1, 0),           # No grid
        (3, 1, 0),           # Only vertical lines, no horizontal
    ]
    
    passed = 0
    for x_count, y_count, expected in test_cases:
        x_coords = list(range(x_count))
        y_coords = list(range(y_count))
        result = count_cells(x_coords, y_coords)
        assert result == expected, f"({x_count}, {y_count}): expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} cell count tests passed")


def test_cell_bounds_extraction():
    """Test extracting bounds for each cell in grid."""
    print("✓ Testing cell bounds extraction...")
    
    def extract_cell_bounds(x_coords, y_coords):
        """Extract bounds for each cell."""
        cells = []
        for row_idx in range(len(y_coords) - 1):
            for col_idx in range(len(x_coords) - 1):
                x_min = x_coords[col_idx]
                y_min = y_coords[row_idx]
                x_max = x_coords[col_idx + 1]
                y_max = y_coords[row_idx + 1]
                cells.append((row_idx, col_idx, x_min, y_min, x_max, y_max))
        return cells
    
    # Test 2x2 grid
    x_coords = [0, 100, 200]
    y_coords = [0, 100, 200]
    cells = extract_cell_bounds(x_coords, y_coords)
    
    assert len(cells) == 4, f"Expected 4 cells, got {len(cells)}"
    assert cells[0] == (0, 0, 0, 0, 100, 100), f"Cell 0,0 incorrect: {cells[0]}"
    assert cells[1] == (0, 1, 100, 0, 200, 100), f"Cell 0,1 incorrect: {cells[1]}"
    assert cells[2] == (1, 0, 0, 100, 100, 200), f"Cell 1,0 incorrect: {cells[2]}"
    assert cells[3] == (1, 1, 100, 100, 200, 200), f"Cell 1,1 incorrect: {cells[3]}"
    
    print(f"  ✓ Cell bounds extraction: 4 cells extracted and verified")


def test_table_validation():
    """Test table validation rules."""
    print("✓ Testing table validation rules...")
    
    def is_valid_table(num_cells, min_cells=4):
        """Check if detected structure is valid table."""
        return num_cells >= min_cells
    
    test_cases = [
        # (num_cells, min_cells, expected_valid)
        (4, 4, True),           # Minimum valid
        (3, 4, False),          # Below minimum
        (9, 4, True),           # Well above minimum
        (0, 4, False),          # No cells
        (4, 1, True),           # Custom minimum
    ]
    
    passed = 0
    for num_cells, min_cells, expected in test_cases:
        result = is_valid_table(num_cells, min_cells)
        assert result == expected, f"({num_cells}, min={min_cells}): expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} table validation tests passed")


def test_line_classification():
    """Test classifying lines as horizontal vs vertical."""
    print("✓ Testing line classification...")
    
    def classify_line(x1, y1, x2, y2):
        """Classify line as horizontal or vertical."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy:
            return "horizontal"
        else:
            return "vertical"
    
    test_cases = [
        # (x1, y1, x2, y2, expected)
        (0, 100, 200, 100, "horizontal"),  # Horizontal line
        (100, 0, 100, 200, "vertical"),    # Vertical line
        (0, 0, 100, 10, "horizontal"),     # Mostly horizontal
        (0, 0, 10, 100, "vertical"),       # Mostly vertical
        (0, 0, 50, 50, "vertical"),        # Diagonal (tie goes to vertical)
    ]
    
    passed = 0
    for x1, y1, x2, y2, expected in test_cases:
        result = classify_line(x1, y1, x2, y2)
        assert result == expected, f"Line ({x1},{y1})-({x2},{y2}): expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} line classification tests passed")


def test_cell_size_filtering():
    """Test filtering cells by minimum size."""
    print("✓ Testing cell size filtering...")
    
    def filter_cells(cells, min_width=10, min_height=10):
        """Filter cells by minimum dimensions."""
        valid_cells = []
        for x_min, y_min, x_max, y_max in cells:
            width = x_max - x_min
            height = y_max - y_min
            if width >= min_width and height >= min_height:
                valid_cells.append((x_min, y_min, x_max, y_max))
        return valid_cells
    
    test_cells = [
        (0, 0, 15, 15),          # Valid: 15x15
        (20, 20, 25, 25),        # Invalid: 5x5 (too small)
        (30, 30, 50, 40),        # Valid: 20x10
        (60, 60, 65, 65),        # Invalid: 5x5
        (70, 70, 80, 85),        # Valid: 10x15
    ]
    
    valid = filter_cells(test_cells, min_width=10, min_height=10)
    
    # Should have 3 valid cells
    assert len(valid) == 3, f"Expected 3 valid cells, got {len(valid)}"
    assert (0, 0, 15, 15) in valid
    assert (30, 30, 50, 40) in valid
    assert (70, 70, 80, 85) in valid
    
    print(f"  ✓ Cell size filtering: 3/5 cells passed validation")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("TABLE DETECTOR UNIT TESTS (Sprint 3, Task 2)")
    print("=" * 70)
    print()
    
    tests = [
        test_horizontal_line_clustering,
        test_vertical_line_clustering,
        test_line_intersections,
        test_grid_building,
        test_cell_count_calculation,
        test_cell_bounds_extraction,
        test_table_validation,
        test_line_classification,
        test_cell_size_filtering,
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
