# OCR Text Extraction System - Development Plan

## Project Overview
Build a parameterized Python-based OCR system using Tesseract that extracts text, tables, lists, and structural information from batches of images, outputting structured data with metadata.

---

## Phase 1: Foundation & Setup

### 1.1 Environment & Dependencies
```
Core Dependencies:
- pytesseract: Python wrapper for Tesseract OCR
- Tesseract-OCR: System-level OCR engine
- opencv-python: Image preprocessing
- Pillow: Image handling
- numpy: Numerical operations
- pandas: Structured data output
- pdf2image: If handling PDFs
- layoutlm: (Optional) Advanced document understanding

Development Tools:
- pytest: Unit testing
- black: Code formatting
- mypy: Type checking
```

### 1.2 Project Structure
```
textExtractor/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & parameters
│   ├── preprocessing.py       # Image enhancement pipeline
│   ├── ocr_engine.py         # Core OCR functionality
│   ├── structure_detector.py # Table/list/figure detection
│   ├── data_models.py        # Result data classes
│   └── batch_processor.py    # Batch operations
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ocr_engine.py
│   ├── test_structure_detector.py
│   └── fixtures/             # Test images
├── examples/
│   └── basic_usage.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Phase 2: Data Models & Output Structure

### 2.1 Core Data Classes (data_models.py)

**OCRResult Class:**
- Extracted text content
- Confidence scores per region
- Detected language
- Raw coordinates of text regions

**StructuralElement Class:**
- Type: "text", "table", "list", "figure", "heading"
- Content (text or table data)
- Bounding box coordinates
- Confidence score
- Nesting level (for hierarchical structures)

**DocumentMetadata Class:**
- Source file path
- Processing timestamp
- Image dimensions
- Preprocessing parameters used
- OCR confidence (average)
- Detected languages

**BatchResult Class:**
- List of DocumentMetadata
- List of StructuralElement objects
- Processing statistics (total time, success rate)
- Output as DataFrame with columns:
  - `source_file`
  - `element_type`
  - `content`
  - `confidence_score`
  - `bounding_box`
  - `page_number`
  - `nesting_level`
  - `processing_time`

---

## Phase 3: Image Preprocessing Pipeline

### 3.1 Preprocessing Strategies (preprocessing.py)

**Implement parameterized preprocessing:**

```python
class PreprocessingConfig:
    - grayscale: bool
    - threshold_type: "otsu" | "adaptive" | "fixed"
    - threshold_value: int (0-255)
    - blur_kernel: tuple (width, height)
    - dilation_iterations: int
    - erosion_iterations: int
    - resize_scale: float
    - denoise_strength: int
```

**Key Preprocessing Functions:**
1. **Grayscale Conversion** - Reduces noise
2. **Thresholding** - Otsu, adaptive, or fixed value
3. **Denoising** - cv2.fastNlMeansDenoising()
4. **Dilation/Erosion** - Morphological operations
5. **Resizing** - For consistency
6. **Skew Correction** - Detect and correct image tilt
7. **Contrast Enhancement** - CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Rationale:** Different image types (book photos, posters, murals) need different preprocessing. Parameterization allows testing multiple strategies.

---

## Phase 4: Core OCR Engine

### 4.1 Main OCR Implementation (ocr_engine.py)

```python
class OCREngine:
    def __init__(self, config: OCRConfig):
        - Initialize Tesseract path
        - Set PSM (Page Segmentation Mode) modes
        - Configure language support
        - Set up preprocessing pipeline
    
    def extract_text_with_regions(image) -> List[OCRResult]:
        - Use Tesseract with detailed output
        - Get per-word bounding boxes
        - Extract confidence scores
        - Return structured results
    
    def get_psm_result(image, psm_mode: int) -> OCRResult:
        - Try different PSM modes (0-13)
        - PSM 3: Fully automatic
        - PSM 6: Uniform block of text
        - PSM 11: Sparse text
        - Return best result based on confidence
```

**Tesseract PSM Modes:**
- 0: Orientation and script detection
- 3: Fully automatic (default)
- 6: Uniform block of text
- 11: Sparse text
- 13: Raw line detection (useful for tables)

**OEM (Engine Mode):**
- OEM 0: Legacy engine
- OEM 1: Neural network
- OEM 2: Both
- OEM 3: Default

---

## Phase 5: Structure Detection

### 5.1 Table Detection (structure_detector.py)

**Approach 1: Line-Based Detection**
```python
def detect_table_regions(image) -> List[BoundingBox]:
    1. Convert to grayscale
    2. Detect horizontal and vertical lines
    3. Find intersections (grid patterns)
    4. Group intersections into tables
    5. Return table bounding boxes
```

**Approach 2: Content-Based Detection**
```python
def detect_table_by_content(ocr_results) -> List[StructuralElement]:
    1. Analyze spatial distribution of text
    2. Identify regular column/row spacing
    3. Extract cell content using OCR results
    4. Build table structure
```

**Approach 3: ML-Based (Advanced)**
- Use LayoutLM or DETR (Detection Transformer)
- Train or use pre-trained models for table detection

### 5.2 List Detection

```python
def detect_lists(ocr_results, image) -> List[StructuralElement]:
    1. Identify bullet points or numbering patterns
    2. Group related text items
    3. Detect indentation levels
    4. Return nested list structure
```

### 5.3 Figure/Image Detection

```python
def detect_figures(image) -> List[StructuralElement]:
    1. Detect large regions without text
    2. Identify boundaries of figures
    3. Mark as figure region
    4. Optionally extract figure description from nearby text
```

### 5.4 Heading Detection

```python
def detect_headings(ocr_results) -> List[StructuralElement]:
    1. Identify text size changes
    2. Detect bold/font variations
    3. Find text above body text
    4. Classify as headings with hierarchy level
```

---

## Phase 6: Configuration System

### 6.1 Parameterization (config.py)

```python
@dataclass
class OCRConfig:
    # Preprocessing
    enable_preprocessing: bool = True
    preprocessing_config: PreprocessingConfig
    
    # OCR Engine
    tesseract_path: str
    languages: List[str] = ["eng"]
    oem_mode: int = 3
    psm_modes: List[int] = [3]
    
    # Structure Detection
    detect_tables: bool = True
    detect_lists: bool = True
    detect_figures: bool = True
    detect_headings: bool = True
    
    # Advanced
    min_confidence: float = 0.3
    use_gpu: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
```

**YAML Configuration File Example:**
```yaml
ocr_config:
  preprocessing:
    grayscale: true
    threshold_type: "otsu"
    denoise: true
  engine:
    languages: ["eng", "fra"]
    oem_mode: 3
    psm_modes: [3, 11]
  structure_detection:
    tables: true
    lists: true
    figures: true
    min_confidence: 0.5
  batch:
    parallel: true
    workers: 4
```

---

## Phase 7: Batch Processor

### 7.1 Main Batch Processing (batch_processor.py)

```python
class BatchProcessor:
    def __init__(self, config: OCRConfig):
        - Initialize OCR engine
        - Load configuration
    
    def process_directory(directory_path, output_format="dataframe") -> BatchResult:
        1. Discover all image files
        2. Process images sequentially (or in parallel)
        3. Aggregate results
        4. Add metadata
        5. Return structured BatchResult
    
    def process_files(file_list: List[str]) -> BatchResult:
        1. Validate files
        2. Process each with error handling
        3. Combine results
        4. Add batch statistics
    
    def export_results(result: BatchResult, format: str):
        - CSV export
        - JSON export (hierarchical)
        - Excel (with formatting)
        - Parquet (for large datasets)
        - HTML (for preview)
```

---

## Phase 8: Output Formats

### 8.1 DataFrame Structure

```
| source_file | element_type | content | confidence | bbox | page | level | processing_time |
|-------------|--------------|---------|------------|------|------|-------|-----------------|
| img1.jpg    | text         | "Hello" | 0.95       | [...] | 1    | 0     | 1.23s           |
| img1.jpg    | table        | {json}  | 0.88       | [...] | 1    | 0     | 1.23s           |
| img1.jpg    | list         | [...] | 0.92       | [...] | 1    | 1     | 1.23s           |
```

### 8.2 JSON Hierarchy (for complex structures)

```json
{
  "source": "image.jpg",
  "metadata": {
    "processed_at": "2026-05-10T10:30:00Z",
    "dimensions": [1920, 1080],
    "confidence_avg": 0.91
  },
  "elements": [
    {
      "type": "heading",
      "content": "Document Title",
      "level": 1,
      "confidence": 0.98,
      "bbox": [100, 50, 500, 100]
    },
    {
      "type": "table",
      "content": {
        "headers": ["Column A", "Column B"],
        "rows": [["Data1", "Data2"]]
      },
      "confidence": 0.87,
      "bbox": [100, 150, 800, 300]
    },
    {
      "type": "list",
      "items": [
        {"text": "Item 1", "level": 0},
        {"text": "Sub-item 1.1", "level": 1}
      ],
      "confidence": 0.93,
      "bbox": [100, 320, 600, 450]
    }
  ]
}
```

---

## Phase 9: Implementation Roadmap

### Sprint 1: Foundation (Weeks 1-2)
- [x] Set up project structure
- [x] Install dependencies
- [ ] Create data models
- [ ] Implement basic OCR engine wrapper

### Sprint 2: Preprocessing (Weeks 3-4)
- [ ] Implement preprocessing pipeline
- [ ] Create parameterized config system
- [ ] Test on sample images
- [ ] Optimize parameters

### Sprint 3: Structure Detection (Weeks 5-7)
- [ ] Implement table detection
- [ ] Implement list detection
- [ ] Implement figure detection
- [ ] Add heading detection

### Sprint 4: Batch Processing (Weeks 8-9)
- [ ] Build batch processor
- [ ] Implement parallel processing
- [ ] Create export functions
- [ ] Add error handling

### Sprint 5: Testing & Optimization (Weeks 10-11)
- [ ] Write comprehensive tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] Example notebooks

### Sprint 6: Advanced Features (Week 12+)
- [ ] ML-based structure detection
- [ ] Multi-language support
- [ ] GPU acceleration
- [ ] API server (optional)

---

## Phase 10: Example Usage

```python
from src.batch_processor import BatchProcessor
from src.config import OCRConfig, PreprocessingConfig

# Configure preprocessing
prep_config = PreprocessingConfig(
    grayscale=True,
    threshold_type="otsu",
    denoise_strength=10,
    resize_scale=1.5
)

# Configure OCR
ocr_config = OCRConfig(
    enable_preprocessing=True,
    preprocessing_config=prep_config,
    languages=["eng"],
    detect_tables=True,
    detect_lists=True,
    detect_figures=True,
    min_confidence=0.5
)

# Process batch
processor = BatchProcessor(ocr_config)
results = processor.process_directory("./images/")

# Export
results.to_csv("extracted_data.csv")
results.to_json("extracted_data.json", hierarchical=True)
results.to_excel("extracted_data.xlsx", with_formatting=True)

# Access results
df = results.as_dataframe()
print(df.groupby("element_type").size())
```

---

## Key Considerations

### Performance Optimization
- Parallel processing for batch operations
- GPU acceleration for image preprocessing (CUDA with OpenCV)
- Caching of preprocessed images
- Incremental processing with checkpoints

### Accuracy Tuning
- Test multiple PSM modes and take best result
- Multiple preprocessing parameter combinations
- Confidence-based filtering
- Manual validation pipeline for critical data

### Robustness
- Error handling for corrupted images
- Fallback strategies for failed extraction
- Logging and monitoring
- Recovery mechanisms for batch failures

### Scalability
- Stream processing for very large batches
- Database storage option (SQLite, PostgreSQL)
- Distributed processing (Celery + Redis)
- Incremental database updates

### Documentation
- Type hints throughout
- Docstrings for all functions
- Example notebooks for each feature
- Troubleshooting guide

---

## Technology Alternatives to Consider

| Component | Primary | Alternative | Use Case |
|-----------|---------|-------------|----------|
| OCR | Tesseract | PaddleOCR, EasyOCR | If Tesseract isn't accurate enough |
| Table Detection | CV-based | LayoutLM, DETR | Complex tables, modern documents |
| Preprocessing | OpenCV | PIL, scikit-image | Specific image processing needs |
| Output | Pandas | DuckDB, Polars | Very large datasets |

---

## Success Metrics

1. **Accuracy**: OCR confidence scores > 90% on test set
2. **Structure Detection**: > 85% accuracy on tables/lists
3. **Performance**: < 5 seconds per image (preprocessing + OCR)
4. **Flexibility**: Support 5+ parameterized configurations
5. **Usability**: Single function call to process batches
6. **Output Quality**: Structured data ready for downstream processing

