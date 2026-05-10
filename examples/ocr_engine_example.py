"""
Example and test script for basic OCR engine wrapper.

Demonstrates:
1. Initializing OCR engine with different configurations
2. Running OCR on test images
3. Handling results and processing traces
4. Confidence filtering and preprocessing options
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from ocr_engine import OCREngine, OCREngineConfig
from data_models import PSMMode, OEMMode


def example_basic_ocr():
    """
    Basic OCR example: Use default configuration on a test image.
    """
    print("=" * 70)
    print("Example 1: Basic OCR with Default Configuration")
    print("=" * 70)
    
    # Create a simple test image with text
    # In real usage, you would load an actual image file
    image = create_test_image()
    
    # Initialize engine with defaults
    config = OCREngineConfig()
    engine = OCREngine(config)
    
    print(f"\nEngine config:")
    print(f"  PSM modes: {[m.name for m in config.psm_modes]}")
    print(f"  OEM mode: {config.oem_mode.name}")
    print(f"  Languages: {config.languages}")
    print(f"  Min confidence: {config.min_confidence:.0%}")
    
    # Extract text
    results, trace = engine.extract_text(image, page_number=1)
    
    print(f"\nResults:")
    print(f"  Text regions found: {len(results)}")
    print(f"  PSM mode used: {trace.psm_mode_used.name}")
    print(f"  Processing time: {trace.processing_duration_ms:.1f}ms")
    print(f"  Average confidence: {trace.average_confidence:.2%}")
    
    if results:
        print(f"\n  First 5 text regions:")
        for i, result in enumerate(results[:5]):
            print(f"    [{i+1}] '{result.text}' @ ({result.bbox.x_min:.0f},"
                  f"{result.bbox.y_min:.0f}) - {result.confidence:.2%}")
    
    return trace


def example_preprocessing():
    """
    Example with preprocessing enabled: binary conversion, upscaling.
    """
    print("\n" + "=" * 70)
    print("Example 2: OCR with Preprocessing")
    print("=" * 70)
    
    image = create_test_image()
    
    # Configure with preprocessing
    config = OCREngineConfig(
        enable_preprocessing=True,
        use_binary=True,  # Apply binary threshold
        target_dpi=300,   # Upscale to 300 DPI
        min_confidence=0.5,  # Higher confidence threshold
    )
    
    engine = OCREngine(config)
    
    print(f"\nEngine config:")
    print(f"  Preprocessing: Enabled")
    print(f"  Binary mode: {config.use_binary}")
    print(f"  Target DPI: {config.target_dpi}")
    print(f"  Min confidence: {config.min_confidence:.0%}")
    
    results, trace = engine.extract_text(image, page_number=1)
    
    print(f"\nResults:")
    print(f"  Text regions found: {len(results)}")
    print(f"  Preprocessing applied: {trace.preprocessing_applied}")
    print(f"  Image size before: {trace.image_dimensions_before}")
    print(f"  Image size after: {trace.image_dimensions_after}")
    print(f"  Processing time: {trace.processing_duration_ms:.1f}ms")
    print(f"  Average confidence: {trace.average_confidence:.2%}")


def example_multiple_psm():
    """
    Example with multiple PSM modes: tries each until successful.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multiple PSM Modes")
    print("=" * 70)
    
    image = create_test_image()
    
    # Try multiple PSM modes for robustness
    config = OCREngineConfig(
        psm_modes=[
            PSMMode.FULLY_AUTOMATIC,
            PSMMode.SINGLE_COLUMN,
            PSMMode.SPARSE_TEXT,
        ],
        oem_mode=OEMMode.DEFAULT,
    )
    
    engine = OCREngine(config)
    
    print(f"\nEngine config:")
    print(f"  PSM modes to try: {[m.name for m in config.psm_modes]}")
    print(f"  (Will use first mode that produces results)")
    
    results, trace = engine.extract_text(image, page_number=1)
    
    print(f"\nResults:")
    print(f"  PSM mode that worked: {trace.psm_mode_used.name}")
    print(f"  Text regions found: {len(results)}")
    print(f"  Average confidence: {trace.average_confidence:.2%}")


def example_confidence_filtering():
    """
    Example showing confidence score normalization and filtering.
    """
    print("\n" + "=" * 70)
    print("Example 4: Confidence Filtering")
    print("=" * 70)
    
    image = create_test_image()
    
    # Try different confidence thresholds
    for min_conf in [0.0, 0.3, 0.6, 0.9]:
        config = OCREngineConfig(min_confidence=min_conf)
        engine = OCREngine(config)
        
        results, trace = engine.extract_text(image, page_number=1)
        
        print(f"\nThreshold {min_conf:.0%}:")
        print(f"  Results: {len(results)}")
        print(f"  Avg confidence: {trace.average_confidence:.2%}")


def example_processing_trace():
    """
    Example showing the processing trace for reproducibility.
    """
    print("\n" + "=" * 70)
    print("Example 5: Processing Trace (Reproducibility)")
    print("=" * 70)
    
    image = create_test_image()
    
    config = OCREngineConfig(
        enable_preprocessing=True,
        use_binary=True,
    )
    
    engine = OCREngine(config)
    results, trace = engine.extract_text(image, page_number=1)
    
    print(f"\nProcessing trace (for reproducibility):")
    trace_dict = trace.to_dict()
    
    for key, value in trace_dict.items():
        if key == "config":
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nThis trace can be saved and used to:")
    print(f"  - Re-run OCR with identical parameters")
    print(f"  - Verify reproducibility of results")
    print(f"  - Debug OCR quality issues")
    print(f"  - Audit processing decisions")


def create_test_image():
    """
    Create a simple test image with text for demonstration.
    
    Note: In real usage, you would load an actual document image.
    """
    # Create blank image
    height, width = 400, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some text using cv2.putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "OCR Engine Test", (50, 100), font, 1.5, (0, 0, 0), 2)
    cv2.putText(image, "This is a test image", (50, 150), font, 1, (0, 0, 0), 1)
    cv2.putText(image, "For OCR extraction", (50, 200), font, 1, (0, 0, 0), 1)
    cv2.putText(image, "Confidence: 95%", (50, 250), font, 0.8, (0, 0, 0), 1)
    
    # Add some simple geometric shapes
    cv2.rectangle(image, (50, 300), (150, 350), (0, 0, 0), 2)
    cv2.circle(image, (400, 350), 30, (0, 0, 0), 2)
    
    return image


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "OCR Engine Wrapper - Examples & Tests".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        # Run examples
        example_basic_ocr()
        example_preprocessing()
        example_multiple_psm()
        example_confidence_filtering()
        example_processing_trace()
        
        print("\n" + "=" * 70)
        print("All Examples Completed Successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Try on real document images (PNG, JPEG, PDF pages)")
        print("  2. Tune PSM modes and preprocessing for your documents")
        print("  3. Implement structure detectors (tables, lists, headings)")
        print("  4. Use OCRTextResult output for element detection phase")
        print()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
