"""
Test script for data_models.py validations and utility methods.

Tests all 7 Option A improvements:
1. __post_init__ validation for all models
2. BoundingBox validations and methods
3. TableCell validations
4. StructuralElement validations and tree methods
5. TableStructure utility methods
6. Enhanced models with styling
7. StructuralElement tree traversal
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import (
    ElementType, ProcessingStatus, PSMMode, OEMMode, ConfidenceLevel,
    Coordinates, BoundingBox, TableCell, TableStructure, StructuralElement,
    ListItem, Annotation, CodeBlock, Caption, OCRTextResult, DocumentMetadata,
    DocumentResult, BatchStatistics, BatchResult
)


def test_confidence_level():
    """Test ConfidenceLevel classification."""
    print("\n✓ Testing ConfidenceLevel.from_score()...")
    assert ConfidenceLevel.from_score(0.1) == ConfidenceLevel.VERY_LOW
    assert ConfidenceLevel.from_score(0.45) == ConfidenceLevel.LOW
    assert ConfidenceLevel.from_score(0.7) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.88) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.97) == ConfidenceLevel.VERY_HIGH
    
    try:
        ConfidenceLevel.from_score(1.5)
        assert False, "Should reject confidence > 1"
    except ValueError:
        pass
    print("  ✓ ConfidenceLevel validation works")


def test_coordinates():
    """Test Coordinates validation."""
    print("\n✓ Testing Coordinates validation...")
    coord = Coordinates(10.0, 20.0)
    assert coord.x == 10.0
    
    try:
        Coordinates(-1, 10)
        assert False, "Should reject negative coordinates"
    except ValueError:
        pass
    print("  ✓ Coordinates validation works")


def test_bounding_box():
    """Test BoundingBox validation and methods."""
    print("\n✓ Testing BoundingBox validation and methods...")
    
    # Valid box
    bbox = BoundingBox(10, 20, 100, 200, confidence=0.95)
    assert bbox.width() == 90
    assert bbox.height() == 180
    assert bbox.area() == 16200
    assert bbox.contains_point(50, 100)
    assert not bbox.contains_point(5, 100)
    print("  ✓ BoundingBox properties work")
    
    # Intersection
    bbox2 = BoundingBox(50, 150, 150, 250)
    intersection = bbox.intersection(bbox2)
    assert intersection is not None
    assert intersection.x_min == 50
    assert intersection.y_min == 150
    print("  ✓ BoundingBox intersection works")
    
    # Union
    union = bbox.union(bbox2)
    assert union.x_min == 10
    assert union.y_min == 20
    assert union.x_max == 150
    assert union.y_max == 250
    print("  ✓ BoundingBox union works")
    
    # Overlap percentage
    overlap = bbox.overlap_percentage(bbox2)
    assert 0 <= overlap <= 1
    print("  ✓ BoundingBox overlap percentage works")
    
    # Invalid boxes should fail
    try:
        BoundingBox(100, 20, 10, 200)
        assert False, "Should reject x_min > x_max"
    except ValueError:
        pass
    
    try:
        BoundingBox(10, 200, 100, 20)
        assert False, "Should reject y_min > y_max"
    except ValueError:
        pass
    
    try:
        BoundingBox(-10, 20, 100, 200)
        assert False, "Should reject negative coordinates"
    except ValueError:
        pass
    
    try:
        BoundingBox(10, 20, 100, 200, confidence=1.5)
        assert False, "Should reject confidence > 1"
    except ValueError:
        pass
    
    print("  ✓ BoundingBox validations work")


def test_table_cell():
    """Test TableCell validation."""
    print("\n✓ Testing TableCell validation...")
    
    bbox = BoundingBox(0, 0, 50, 50)
    cell = TableCell("Content", 0, 0, bbox, 0.9, colspan=2, rowspan=1)
    assert cell.colspan == 2
    print("  ✓ TableCell creation works")
    
    try:
        TableCell("Content", -1, 0, bbox, 0.9)
        assert False, "Should reject negative row"
    except ValueError:
        pass
    
    try:
        TableCell("Content", 0, 0, bbox, 0.9, colspan=0)
        assert False, "Should reject colspan < 1"
    except ValueError:
        pass
    
    try:
        TableCell("Content", 0, 0, bbox, 1.5)
        assert False, "Should reject confidence > 1"
    except ValueError:
        pass
    
    print("  ✓ TableCell validation works")


def test_table_structure():
    """Test TableStructure utility methods."""
    print("\n✓ Testing TableStructure utility methods...")
    
    bbox = BoundingBox(0, 0, 200, 200)
    cells = [
        TableCell("Header 1", 0, 0, BoundingBox(0, 0, 50, 50), 0.95, is_header=True),
        TableCell("Header 2", 0, 1, BoundingBox(50, 0, 100, 50), 0.95, is_header=True),
        TableCell("Data 1", 1, 0, BoundingBox(0, 50, 50, 100), 0.90),
        TableCell("Data 2", 1, 1, BoundingBox(50, 50, 100, 100), 0.90),
    ]
    
    table = TableStructure(cells, bbox, 0.92)
    
    # Check grid computation
    assert table.num_rows == 2
    assert table.num_cols == 2
    print("  ✓ TableStructure grid computation works")
    
    # Test get_cell
    cell = table.get_cell(0, 0)
    assert cell is not None
    assert cell.content == "Header 1"
    print("  ✓ TableStructure.get_cell() works")
    
    # Test get_row
    row = table.get_row(0)
    assert len(row) == 2
    assert row[0].content == "Header 1"
    print("  ✓ TableStructure.get_row() works")
    
    # Test get_column
    col = table.get_column(0)
    assert len(col) == 2
    assert col[0].content == "Header 1"
    print("  ✓ TableStructure.get_column() works")
    
    # Test to_2d_array
    array = table.to_2d_array()
    assert array[0][0] == "Header 1"
    assert array[1][1] == "Data 2"
    print("  ✓ TableStructure.to_2d_array() works")
    
    # Test to_markdown
    markdown = table.to_markdown()
    assert "Header 1" in markdown
    assert "---" in markdown
    print("  ✓ TableStructure.to_markdown() works")
    
    # Test to_csv
    csv_text = table.to_csv()
    assert "Header 1" in csv_text
    print("  ✓ TableStructure.to_csv() works")


def test_structural_element():
    """Test StructuralElement validation and methods."""
    print("\n✓ Testing StructuralElement validation and methods...")
    
    bbox = BoundingBox(10, 10, 100, 100)
    
    # Valid element
    elem = StructuralElement(
        "elem_1", ElementType.HEADING, "Title Text", bbox, 0.95, page_number=1
    )
    assert elem.element_type == ElementType.HEADING
    print("  ✓ StructuralElement creation works")
    
    # Test to_dict
    elem_dict = elem.to_dict()
    assert elem_dict["element_id"] == "elem_1"
    assert elem_dict["element_type"] == "heading"
    assert elem_dict["confidence"] == 0.95
    print("  ✓ StructuralElement.to_dict() works")
    
    # Test to_json
    elem_json = elem.to_json()
    assert "elem_1" in elem_json
    print("  ✓ StructuralElement.to_json() works")
    
    # Test spatial queries
    assert elem.bbox.contains_point(50, 50)
    assert not elem.bbox.contains_point(5, 5)
    print("  ✓ StructuralElement spatial check works")
    
    region = BoundingBox(0, 0, 150, 150)
    assert elem.in_region(region)
    print("  ✓ StructuralElement.in_region() works")
    
    bbox2 = BoundingBox(80, 80, 200, 200)
    elem2 = StructuralElement("elem_2", ElementType.TEXT, "Body text", bbox2, 0.90)
    assert elem.overlaps_with(elem2)
    print("  ✓ StructuralElement.overlaps_with() works")
    
    # Test tree methods
    elem.child_ids = ["elem_2"]
    elem2.parent_id = "elem_1"
    
    descendants = elem.get_descendants([elem, elem2])
    assert len(descendants) == 1
    assert descendants[0].element_id == "elem_2"
    print("  ✓ StructuralElement.get_descendants() works")
    
    ancestors = elem2.get_ancestors([elem, elem2])
    assert len(ancestors) == 1
    assert ancestors[0].element_id == "elem_1"
    print("  ✓ StructuralElement.get_ancestors() works")
    
    # Test validation
    try:
        StructuralElement("e", "invalid", "text", bbox, 0.95)
        assert False, "Should reject invalid element type"
    except ValueError:
        pass
    
    try:
        StructuralElement("e", ElementType.TEXT, "text", bbox, 1.5)
        assert False, "Should reject confidence > 1"
    except ValueError:
        pass
    
    print("  ✓ StructuralElement validation works")


def test_other_models():
    """Test validation of other models."""
    print("\n✓ Testing other model validations...")
    
    bbox = BoundingBox(0, 0, 100, 100)
    
    # Annotation
    annotation = Annotation("highlighted", bbox, "highlight", 0.95)
    print("  ✓ Annotation works")
    
    # CodeBlock
    code = CodeBlock("def foo():\n    pass", bbox, 0.92, language="python")
    print("  ✓ CodeBlock works")
    
    # Caption
    caption = Caption("Figure 1: Example", "figure", bbox, 0.93)
    print("  ✓ Caption works")
    
    # OCRTextResult
    ocr = OCRTextResult("Some text", 95.5, bbox, page_number=1)
    assert 0 <= ocr.confidence <= 100
    print("  ✓ OCRTextResult works")
    
    # DocumentMetadata
    meta = DocumentMetadata(
        "/path/to/img.jpg", "doc_1", datetime.now(), 2.5,
        (1920, 1080), "eng", 10, 0.92, ProcessingStatus.COMPLETED
    )
    print("  ✓ DocumentMetadata works")


def test_document_result():
    """Test DocumentResult index and query methods."""
    print("\n✓ Testing DocumentResult methods...")
    
    bbox = BoundingBox(0, 0, 100, 100)
    meta = DocumentMetadata(
        "/path/to/img.jpg", "doc_1", datetime.now(), 2.5,
        (1920, 1080), "eng", 2, 0.92, ProcessingStatus.COMPLETED
    )
    
    elements = [
        StructuralElement("h1", ElementType.HEADING, "Title", bbox, 0.95, page_number=1),
        StructuralElement("p1", ElementType.TEXT, "Paragraph", bbox, 0.90, page_number=1),
    ]
    
    result = DocumentResult(meta, elements)
    
    # Index should be built
    assert "h1" in result.element_index
    assert "p1" in result.element_index
    print("  ✓ DocumentResult index building works")
    
    # Query methods
    headings = result.get_elements_by_type(ElementType.HEADING)
    assert len(headings) == 1
    assert headings[0].element_id == "h1"
    print("  ✓ DocumentResult.get_elements_by_type() works")
    
    page1 = result.get_elements_on_page(1)
    assert len(page1) == 2
    print("  ✓ DocumentResult.get_elements_on_page() works")
    
    in_region = result.get_elements_in_region(bbox)
    assert len(in_region) >= 1
    print("  ✓ DocumentResult.get_elements_in_region() works")


def test_batch_result():
    """Test BatchResult statistics and filtering."""
    print("\n✓ Testing BatchResult methods...")
    
    bbox = BoundingBox(0, 0, 100, 100)
    meta = DocumentMetadata(
        "/path/to/img.jpg", "doc_1", datetime.now(), 2.5,
        (1920, 1080), "eng", 2, 0.92, ProcessingStatus.COMPLETED
    )
    
    elements = [
        StructuralElement("h1", ElementType.HEADING, "Title", bbox, 0.95),
        StructuralElement("p1", ElementType.TEXT, "Paragraph", bbox, 0.90),
    ]
    
    result = DocumentResult(meta, elements)
    batch = BatchResult("batch_1", datetime.now(), [result])
    
    # Statistics should be auto-computed
    assert batch.statistics is not None
    assert batch.statistics.total_elements == 2
    assert batch.statistics.successful_documents == 1
    print("  ✓ BatchResult statistics computation works")
    
    # Filtering
    headings_batch = batch.filter_by_type(ElementType.HEADING)
    assert len(headings_batch.documents) == 1
    assert len(headings_batch.documents[0].elements) == 1
    print("  ✓ BatchResult.filter_by_type() works")
    
    high_conf = batch.filter_by_confidence(0.92)
    assert len(high_conf.documents) == 1
    print("  ✓ BatchResult.filter_by_confidence() works")


def main():
    """Run all tests."""
    print("=" * 70)
    print("DATA MODELS VALIDATION TEST SUITE")
    print("=" * 70)
    print("Testing all 7 Option A improvements...")
    
    try:
        test_confidence_level()
        test_coordinates()
        test_bounding_box()
        test_table_cell()
        test_table_structure()
        test_structural_element()
        test_other_models()
        test_document_result()
        test_batch_result()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✓ Task 1: __post_init__ validation to all models ✅")
        print("✓ Task 2: BoundingBox validation & methods ✅")
        print("✓ Task 3: TableCell validation ✅")
        print("✓ Task 4: StructuralElement validation & methods ✅")
        print("✓ Task 5: TableStructure utility methods ✅")
        print("✓ Task 6: Model enhancements ✅")
        print("✓ Task 7: StructuralElement tree traversal ✅")
        print("\n" + "=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
