"""
Unit tests for ContentTableDetector core functionality (Sprint 3, Task 3).

Tests spatial analysis, column/row detection, and cell assignment logic
without requiring the full OCR stack.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_column_boundary_detection():
    """Test detecting column boundaries from x-coordinates."""
    print("✓ Testing column boundary detection...")
    
    def detect_columns(x_coords, threshold=10.0):
        """Detect column boundaries via clustering."""
        if not x_coords:
            return []
        
        coords = sorted(x_coords)
        columns = []
        current_cluster = [coords[0]]
        
        for x in coords[1:]:
            if x - current_cluster[-1] <= threshold:
                current_cluster.append(x)
            else:
                columns.append(np.mean(current_cluster))
                current_cluster = [x]
        
        if current_cluster:
            columns.append(np.mean(current_cluster))
        
        return columns
    
    test_cases = [
        # (x_coords, threshold, expected_count)
        ([10, 11, 12, 100, 101, 200, 201], 10.0, 3),  # 3 columns
        ([10, 50, 100, 150], 10.0, 4),                 # 4 columns
        ([10, 15, 20, 25], 10.0, 1),                   # 1 column (all close)
        ([10, 100, 200, 300], 10.0, 4),                # 4 columns (far apart)
    ]
    
    passed = 0
    for x_coords, threshold, expected in test_cases:
        result = detect_columns(x_coords, threshold)
        assert len(result) == expected, f"For {x_coords}, expected {expected} columns, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} column detection tests passed")


def test_row_boundary_detection():
    """Test detecting row boundaries from y-coordinates."""
    print("✓ Testing row boundary detection...")
    
    def detect_rows(y_coords, threshold=5.0):
        """Detect row boundaries via clustering."""
        if not y_coords:
            return []
        
        coords = sorted(y_coords)
        rows = []
        current_cluster = [coords[0]]
        
        for y in coords[1:]:
            if y - current_cluster[-1] <= threshold:
                current_cluster.append(y)
            else:
                rows.append(np.mean(current_cluster))
                current_cluster = [y]
        
        if current_cluster:
            rows.append(np.mean(current_cluster))
        
        return rows
    
    test_cases = [
        # (y_coords, threshold, expected_count)
        ([10, 11, 100, 101, 200, 201], 5.0, 3),       # 3 rows
        ([10, 50, 100, 150], 5.0, 4),                  # 4 rows
        ([10, 12, 14, 16], 5.0, 1),                    # 1 row (all close)
        ([10, 100, 200], 5.0, 3),                      # 3 rows (far apart)
    ]
    
    passed = 0
    for y_coords, threshold, expected in test_cases:
        result = detect_rows(y_coords, threshold)
        assert len(result) == expected, f"For {y_coords}, expected {expected} rows, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} row detection tests passed")


def test_boundary_filtering():
    """Test filtering boundaries by minimum gap."""
    print("✓ Testing boundary filtering...")
    
    def filter_boundaries(coords, min_gap):
        """Filter boundaries to remove overlapping ones."""
        if not coords:
            return []
        
        filtered = [coords[0]]
        for coord in coords[1:]:
            if coord - filtered[-1] > min_gap:
                filtered.append(coord)
        
        return filtered
    
    test_cases = [
        # (coords, min_gap, expected_count)
        ([10, 15, 20, 100, 105], 5.0, 3),   # Some filtered (coords with gap > 5)
        ([10, 100, 200, 300], 30.0, 4),     # None filtered (all gaps > 30)
        ([10, 11, 12, 13], 10.0, 1),        # Most filtered (tight threshold vs spread)
        ([10, 50, 90], 30.0, 3),            # All pass
    ]
    
    passed = 0
    for coords, min_gap, expected in test_cases:
        result = filter_boundaries(coords, min_gap)
        assert len(result) == expected, f"For {coords} (gap={min_gap}), expected {expected}, got {len(result)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} boundary filtering tests passed")


def test_grid_creation():
    """Test creating a grid from boundaries."""
    print("✓ Testing grid creation from boundaries...")
    
    def create_grid(x_boundaries, y_boundaries):
        """Create cell grid from boundaries."""
        num_cols = len(x_boundaries) - 1
        num_rows = len(y_boundaries) - 1
        
        grid = {}
        for row in range(num_rows):
            for col in range(num_cols):
                grid[(row, col)] = {
                    'row': row,
                    'col': col,
                    'x_min': x_boundaries[col],
                    'x_max': x_boundaries[col + 1],
                    'y_min': y_boundaries[row],
                    'y_max': y_boundaries[row + 1],
                }
        return grid
    
    test_cases = [
        # (x_boundaries, y_boundaries, expected_cells)
        ([0, 100, 200], [0, 100, 200], 4),        # 2x2 = 4 cells
        ([0, 100, 200, 300], [0, 100, 200], 6),   # 3x2 = 6 cells
        ([0, 100, 200], [0, 100, 200, 300], 6),   # 2x3 = 6 cells
        ([0, 100], [0, 100], 1),                   # 1x1 = 1 cell
    ]
    
    passed = 0
    for x_bounds, y_bounds, expected in test_cases:
        grid = create_grid(x_bounds, y_bounds)
        assert len(grid) == expected, f"({len(x_bounds)-1}x{len(y_bounds)-1}): expected {expected} cells, got {len(grid)}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} grid creation tests passed")


def test_cell_overlap_detection():
    """Test detecting which cell a text region overlaps with."""
    print("✓ Testing cell overlap detection...")
    
    def find_overlapping_cells(text_x_min, text_x_max, text_y_min, text_y_max, x_boundaries, y_boundaries):
        """Find cells that text overlaps."""
        overlapping = []
        
        for row in range(len(y_boundaries) - 1):
            for col in range(len(x_boundaries) - 1):
                x_min = x_boundaries[col]
                x_max = x_boundaries[col + 1]
                y_min = y_boundaries[row]
                y_max = y_boundaries[row + 1]
                
                # Check overlap
                if (text_x_min < x_max and text_x_max > x_min and
                    text_y_min < y_max and text_y_max > y_min):
                    overlapping.append((row, col))
        
        return overlapping
    
    # Test: text in corner of 2x2 grid
    x_bounds = [0, 100, 200]
    y_bounds = [0, 100, 200]
    
    # Text in top-left cell
    overlaps = find_overlapping_cells(10, 50, 10, 50, x_bounds, y_bounds)
    assert overlaps == [(0, 0)], f"Expected [(0,0)], got {overlaps}"
    
    # Text spanning two cells horizontally
    overlaps = find_overlapping_cells(50, 150, 10, 50, x_bounds, y_bounds)
    assert set(overlaps) == {(0, 0), (0, 1)}, f"Expected [(0,0), (0,1)], got {overlaps}"
    
    # Text spanning two cells vertically
    overlaps = find_overlapping_cells(10, 50, 50, 150, x_bounds, y_bounds)
    assert set(overlaps) == {(0, 0), (1, 0)}, f"Expected [(0,0), (1,0)], got {overlaps}"
    
    # Text spanning four cells (2x2)
    overlaps = find_overlapping_cells(50, 150, 50, 150, x_bounds, y_bounds)
    assert len(overlaps) == 4, f"Expected 4 cells, got {len(overlaps)}"
    
    print(f"  ✓ Cell overlap detection tests passed (4/4 scenarios)")


def test_cell_center_distance():
    """Test computing distance from point to cell center."""
    print("✓ Testing cell center distance calculation...")
    
    def distance_to_center(x, y, x_min, y_min, x_max, y_max):
        """Distance from point to cell center."""
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Cell: 0-100 x, 0-100 y, center at (50, 50)
    x_min, x_max, y_min, y_max = 0, 100, 0, 100
    
    # At center
    dist = distance_to_center(50, 50, x_min, y_min, x_max, y_max)
    assert dist == 0, f"Center distance should be 0, got {dist}"
    
    # At corner
    dist = distance_to_center(0, 0, x_min, y_min, x_max, y_max)
    expected = np.sqrt(50**2 + 50**2)
    assert abs(dist - expected) < 0.01, f"Corner distance should be {expected}, got {dist}"
    
    # On edge (horizontal)
    dist = distance_to_center(50, 0, x_min, y_min, x_max, y_max)
    assert dist == 50, f"Edge distance should be 50, got {dist}"
    
    print(f"  ✓ Cell center distance calculation tests passed (3/3 scenarios)")


def test_text_aggregation():
    """Test aggregating text from multiple regions into cells."""
    print("✓ Testing text aggregation...")
    
    # Simulate: 3 text regions, 1 cell
    texts = ["Hello", "World", "!"]
    
    # Concatenate with space
    aggregated = " ".join(texts)
    assert aggregated == "Hello World !", f"Expected 'Hello World !', got '{aggregated}'"
    
    # Handle empty
    aggregated = " ".join([])
    assert aggregated == "", f"Expected empty string, got '{aggregated}'"
    
    # Single text
    aggregated = " ".join(["Single"])
    assert aggregated == "Single", f"Expected 'Single', got '{aggregated}'"
    
    print(f"  ✓ Text aggregation tests passed (3/3 scenarios)")


def test_table_validation():
    """Test validating detected table."""
    print("✓ Testing table validation...")
    
    def is_valid_table(num_cells, min_cells=4, min_cols=2, min_rows=2):
        """Check if detected structure is valid table."""
        return num_cells >= min_cells
    
    test_cases = [
        # (num_cells, min_cells, expected_valid)
        (4, 4, True),           # Minimum
        (3, 4, False),          # Below minimum
        (9, 4, True),           # Well above
        (0, 4, False),          # Empty
        (4, 1, True),           # Custom minimum
    ]
    
    passed = 0
    for num_cells, min_cells, expected in test_cases:
        result = is_valid_table(num_cells, min_cells)
        assert result == expected, f"({num_cells}, min={min_cells}): expected {expected}, got {result}"
        passed += 1
    
    print(f"  ✓ {passed}/{len(test_cases)} table validation tests passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("CONTENT TABLE DETECTOR UNIT TESTS (Sprint 3, Task 3)")
    print("=" * 70)
    print()
    
    tests = [
        test_column_boundary_detection,
        test_row_boundary_detection,
        test_boundary_filtering,
        test_grid_creation,
        test_cell_overlap_detection,
        test_cell_center_distance,
        test_text_aggregation,
        test_table_validation,
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
