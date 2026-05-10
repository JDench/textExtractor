# OCR Engine Wrapper Implementation

## Overview

The **OCR Engine Wrapper** (`src/ocr_engine.py`) is a parameterized Tesseract OCR integration that serves as the foundation for the text extraction system. It implements Sprint 1 deliverables and establishes the architectural patterns used throughout the project.

## What It Does

The OCR engine:
1. **Wraps Tesseract OCR** with a clean Python interface
2. **Supports multiple PSM modes** (Page Segmentation Modes) for different document layouts
3. **Normalizes confidence scores** from Tesseract's 0-100 to project's 0-1 scale
4. **Preprocesses images** (binary conversion, upscaling, etc.) for better OCR accuracy
5. **Records processing traces** for reproducibility and debugging
6. **Returns structured data** as `OCRTextResult` objects with bounding boxes

## Key Features

### ✅ Parameterized Configuration
The `OCREngineConfig` dataclass allows full customization:
```python
config = OCREngineConfig(
    psm_modes=[PSMMode.FULLY_AUTOMATIC, PSMMode.SPARSE_TEXT],
    oem_mode=OEMMode.DEFAULT,
    languages="eng",
    min_confidence=0.3,
    enable_preprocessing=True,
    use_binary=True,
    target_dpi=300,
)
engine = OCREngine(config)
```

### ✅ Multiple PSM Modes
Tries different segmentation strategies to find best results:
- `PSMMode.FULLY_AUTOMATIC` - Default, works for most documents
- `PSMMode.SINGLE_COLUMN` - Single column text (books, articles)
- `PSMMode.SPARSE_TEXT` - Scattered text (forms, documents with whitespace)
- And 11+ others for specialized layouts

### ✅ Preprocessing Pipeline
Optional image preprocessing improves OCR accuracy:
- Grayscale conversion
- DPI-aware upscaling (100 DPI → 300 DPI)
- Binary thresholding (converts to pure black/white)
- Extensible for adding denoise, rotation detection, etc.

### ✅ Confidence Filtering
Low-confidence results can be filtered at extraction time:
```python
config = OCREngineConfig(min_confidence=0.5)  # 50% minimum
results, trace = engine.extract_text(image)
# All results will have confidence >= 0.5
```

### ✅ Processing Traces
Every extraction records what happened for reproducibility:
```python
results, trace = engine.extract_text(image)
trace_dict = trace.to_dict()
# Includes: config used, PSM mode, preprocessing steps, timing, stats

# Later: can re-run with exact same config
engine2 = OCREngine(OCREngineConfig(**trace_dict["config"]))
```

## Usage Examples

### Basic OCR
```python
from ocr_engine import OCREngine, load_image

image = load_image("document.png")
engine = OCREngine()  # Uses default config
results, trace = engine.extract_text(image, page_number=1)

for result in results:
    print(f"{result.text}: {result.confidence:.2%}")
    print(f"  Location: ({result.bbox.x_min}, {result.bbox.y_min})")
```

### Custom Configuration
```python
config = OCREngineConfig(
    psm_modes=[PSMMode.SINGLE_COLUMN],
    enable_preprocessing=True,
    use_binary=True,
    min_confidence=0.6,
)
engine = OCREngine(config)
results, trace = engine.extract_text(image)
print(f"Processed in {trace.processing_duration_ms:.0f}ms")
print(f"Found {len(results)} text regions")
```

### Fallback Strategy (Multiple PSM Modes)
```python
# Engine will try each mode until one produces results
config = OCREngineConfig(
    psm_modes=[
        PSMMode.FULLY_AUTOMATIC,
        PSMMode.SPARSE_TEXT,
        PSMMode.SINGLE_COLUMN,
    ]
)
engine = OCREngine(config)
results, trace = engine.extract_text(image)
print(f"Successfully used PSM mode: {trace.psm_mode_used.name}")
```

### Extract Simple Text String
```python
text = engine.extract_text_simple(image)
print(text)  # All text as single string
```

## Data Output Format

### OCRTextResult Objects
Each extracted text region returns an `OCRTextResult`:
```python
OCRTextResult(
    text="Hello World",              # Extracted text
    confidence=0.95,                 # Normalized 0-1
    bbox=BoundingBox(                # Location in image
        x_min=100, y_min=50,
        x_max=300, y_max=80,
        confidence=0.95
    ),
    language="eng",                  # Detected language
    page_number=1,                   # Page number
    x_baseline=79,                   # Text baseline (if available)
    is_numeric=False,                # Numeric content indicator
)
```

### OCRProcessingTrace Objects
Each extraction also returns a `OCRProcessingTrace`:
```python
OCRProcessingTrace(
    config=config,                   # Config used
    psm_mode_used=PSMMode.FULLY_AUTOMATIC,
    processing_duration_ms=245.5,    # Timing
    image_dimensions_before=(1920, 1080),
    image_dimensions_after=(1920, 1080),
    preprocessing_applied=["grayscale_conversion"],
    total_results=42,                # Statistics
    average_confidence=0.92,
)
```

## Running Examples

The `examples/ocr_engine_example.py` contains 5 runnable examples:

```bash
cd examples
python ocr_engine_example.py
```

Examples demonstrate:
1. **Basic OCR** with default configuration
2. **Preprocessing** (binary conversion, upscaling)
3. **Multiple PSM modes** (fallback strategy)
4. **Confidence filtering** with different thresholds
5. **Processing traces** for reproducibility

## Architecture & Design Patterns

### ✅ Aligns with Architectural Decisions
1. **Strategy Pattern** (AD #3): Full parameterization via `OCREngineConfig`
2. **Processing Traces** (AD #5): `OCRProcessingTrace` records all metadata
3. **Confidence Normalization**: Converts Tesseract's 0-100 to 0-1
4. **Hierarchical Relations**: Results linked by page and location

### ✅ Data Flow
```
Image + OCREngineConfig
    ↓
    [Preprocessing (optional)]
    ↓
    [Tesseract OCR - try each PSM mode]
    ↓
List[OCRTextResult] + OCRProcessingTrace
```

## Requirements

**Python Dependencies:**
```bash
pip install pytesseract opencv-python numpy
```

**System Requirements:**
- Tesseract OCR (must be installed separately)
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Performance Characteristics

- **Single image**: 100-500ms (depends on image size and content)
- **Preprocessing overhead**: ~10-20% of total time
- **Memory**: Efficient - stores only detected results, not full image
- **Scalability**: Designed for batch processing (hundreds of images)

## Next Steps

This OCR engine is the foundation for:
- **Sprint 2** (Text Detection): Use `OCRTextResult` objects to detect headings and paragraphs
- **Sprint 3-5** (Complex Structures): Implement table, list, formula detection
- **Sprint 7** (Batch Processing): Orchestrate processing of multiple documents

## Configuration Reference

### OCREngineConfig Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `psm_modes` | List[PSMMode] | [FULLY_AUTOMATIC] | Page segmentation modes to try |
| `oem_mode` | OEMMode | DEFAULT | OCR engine: legacy vs. neural network |
| `languages` | str | "eng" | Tesseract language codes (e.g., "eng+fra") |
| `min_confidence` | float | 0.3 | Filter results below this (0-1) |
| `enable_preprocessing` | bool | True | Apply image preprocessing |
| `target_dpi` | Optional[int] | None | Upscale to DPI (e.g., 300) |
| `use_binary` | bool | False | Binary threshold conversion |
| `include_font_properties` | bool | False | Extract font name/size |
| `detect_language` | bool | True | Attempt language detection |
| `tesseract_path` | Optional[str] | None | Explicit Tesseract path |
| `extra_config` | str | "" | Additional Tesseract config |

## Troubleshooting

**"Tesseract not found"**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Set `pytesseract.pytesseract_cmd` if not in PATH

**Low confidence scores**
- Try different PSM modes
- Enable preprocessing (`use_binary=True`)
- Check image quality (resolution, clarity)
- Try `target_dpi=300` for upscaling

**Wrong text detected**
- Try different PSM mode (e.g., `SPARSE_TEXT` for forms)
- Check image is upright (no rotation)
- Ensure document is readable by human eye

## Implementation Notes

- **Language Support**: Change `languages` parameter for multi-language OCR (e.g., "eng+fra")
- **Confidence Thresholds**: Consider per-element type thresholds in later phases
- **Extensibility**: New preprocessing steps can be added to `_preprocess_image()`
- **Reproducibility**: Always save the `OCRProcessingTrace` for audit trail

## Testing Strategy

1. Test on various document types (printed text, handwriting, tables)
2. Compare results with/without preprocessing
3. Try different PSM modes to find optimal for your documents
4. Monitor confidence scores - they indicate OCR reliability
5. Save processing traces to verify reproducibility

---

**Status**: ✅ Sprint 1 Complete  
**Next**: Sprint 2 - Text Detection (Heading & Paragraph extraction)
