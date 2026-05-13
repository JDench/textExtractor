"""
Pytest fixtures for the test suite.

Factory helper functions live in helpers.py (importable directly).
This file exposes them as pytest fixtures for tests that prefer injection.
"""

import numpy as np
import pytest

from helpers import (
    make_batch,
    make_doc,
    make_element,
    make_heading,
    make_bbox,
    make_ocr,
)
from data_models import ElementType


@pytest.fixture
def blank_image():
    """Small blank BGR image (200×400) for passing to detectors."""
    return np.zeros((200, 400, 3), dtype=np.uint8)


@pytest.fixture
def gray_image():
    """Medium-gray BGR image — useful for watermark opacity tests."""
    return np.full((200, 400, 3), 160, dtype=np.uint8)


@pytest.fixture
def sample_elements():
    h1 = make_heading("H1 Title", level=1, y_min=0, element_id="h1")
    h2 = make_heading("H2 Subtitle", level=2, y_min=40, element_id="h2")
    p1 = make_element(ElementType.TEXT, "Body paragraph", y_min=70, y_max=90, element_id="p1")
    return [h1, h2, p1]


@pytest.fixture
def sample_batch(sample_elements):
    return make_batch([sample_elements])
