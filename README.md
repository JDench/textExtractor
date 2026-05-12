# textExtractor

OCR-based text and document structure extraction using Tesseract.

## Environment Setup

This project uses a conda virtual environment (`.venv`).

### Install dependencies

```bash
# Activate the conda environment first
conda activate .venv

# Install Python dependencies
pip install -r requirements.txt
```

### System dependency: Tesseract

`pytesseract` requires Tesseract OCR to be installed on the system separately from the Python packages.

- **Windows**: Download from <https://github.com/UB-Mannheim/tesseract/wiki> and add to PATH
- **macOS**: `brew install tesseract`
- **Linux**: `apt install tesseract-ocr`

## Dependencies (`requirements.txt`)

|Package|Purpose|
|---|---|
|`numpy`|Array and image data manipulation|
|`opencv-python`|Image preprocessing and computer vision|
|`pytesseract`|Python wrapper for Tesseract OCR|
|`Pillow`|Image loading and rendering utilities|

**Keep `requirements.txt` updated** whenever a new package is added to the project — this is part of the development workflow (see `OCR_DEVELOPMENT_PLAN.md`).

## Project Structure

```
textExtractor/
├── src/
│   ├── data_models.py       # Core data classes and enums
│   ├── ocr_engine.py        # Tesseract OCR wrapper
│   └── detectors/
│       ├── text_detector.py # Heading and paragraph detection
│       └── list_detector.py # Bullet/numbered list detection
├── examples/                # Usage examples and validation scripts
├── requirements.txt         # Python package dependencies
└── OCR_DEVELOPMENT_PLAN.md  # Full development roadmap
```

## Development Status

- Sprint 1 (Foundation): Complete
- Sprint 2 (Text Detection): Complete
- Sprint 3 (List & Table Detection): In progress — see `TODO_REVIEW.md`
