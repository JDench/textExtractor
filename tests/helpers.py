"""
Shared factory helpers for the test suite.

Plain functions (not fixtures) — import these directly from test files.
Pytest fixtures that wrap these are defined in conftest.py.
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Path setup — makes src importable from any test file ──────────────────────
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_models import (
    BatchResult,
    BoundingBox,
    DocumentMetadata,
    DocumentResult,
    ElementType,
    OCRTextResult,
    ProcessingStatus,
    StructuralElement,
)


def make_bbox(
    x_min: float = 0,
    y_min: float = 0,
    x_max: float = 100,
    y_max: float = 30,
) -> BoundingBox:
    return BoundingBox(x_min, y_min, x_max, y_max)


def make_ocr(
    text: str,
    x_min: float = 0,
    y_min: float = 0,
    x_max: Optional[float] = None,
    y_max: Optional[float] = None,
    confidence: float = 0.85,
    page_number: int = 1,
) -> OCRTextResult:
    if x_max is None:
        x_max = x_min + max(len(text) * 8, 20)
    if y_max is None:
        y_max = y_min + 20
    return OCRTextResult(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
        page_number=page_number,
    )


def make_element(
    element_type: ElementType,
    content,
    x_min: float = 0,
    y_min: float = 0,
    x_max: float = 400,
    y_max: float = 30,
    confidence: float = 0.90,
    page_number: int = 1,
    nesting_level: int = 0,
    parent_id: Optional[str] = None,
    element_id: Optional[str] = None,
    heading_level: Optional[int] = None,
) -> StructuralElement:
    meta = {}
    if heading_level is not None:
        meta["heading_level"] = heading_level
    return StructuralElement(
        element_id=element_id or f"elem_{uuid.uuid4().hex[:6]}",
        element_type=element_type,
        content=content,
        bbox=BoundingBox(x_min, y_min, x_max, y_max),
        confidence=confidence,
        page_number=page_number,
        nesting_level=nesting_level,
        parent_id=parent_id,
        metadata=meta,
    )


def make_heading(
    text: str,
    level: int,
    y_min: float = 0,
    element_id: Optional[str] = None,
) -> StructuralElement:
    return make_element(
        ElementType.HEADING,
        text,
        y_min=y_min,
        y_max=y_min + 20,
        nesting_level=level - 1,
        element_id=element_id,
        heading_level=level,
    )


def make_doc(
    elements: Optional[List[StructuralElement]] = None,
    doc_id: Optional[str] = None,
    document_id: Optional[str] = None,
    source_file: str = "test.png",
) -> DocumentResult:
    if elements is None:
        elements = []
    resolved_id = document_id or doc_id or "doc_1"
    avg_conf = (
        sum(e.confidence for e in elements) / len(elements) if elements else 0.0
    )
    meta = DocumentMetadata(
        source_file=source_file,
        document_id=resolved_id,
        processing_timestamp=datetime.now(),
        processing_duration=0.5,
        image_dimensions=(800, 600),
        detected_language="eng",
        total_elements_extracted=len(elements),
        average_confidence=avg_conf,
        processing_status=ProcessingStatus.COMPLETED,
    )
    return DocumentResult(metadata=meta, elements=elements)


def make_batch(
    elements_per_doc: Optional[List[List[StructuralElement]]] = None,
    batch_id: str = "test_batch",
    documents: Optional[List[DocumentResult]] = None,
) -> BatchResult:
    if documents is not None:
        return BatchResult(batch_id=batch_id, created_at=datetime.now(), documents=documents)
    if elements_per_doc is None:
        elements_per_doc = [[]]
    docs = [
        make_doc(elems, doc_id=f"doc_{i}", source_file=f"test_{i}.png")
        for i, elems in enumerate(elements_per_doc)
    ]
    return BatchResult(batch_id=batch_id, created_at=datetime.now(), documents=docs)
