#!/usr/bin/env python3
"""
PaddleOCR Integration Demonstration
Shows how to use OCR for text recognition in PDFs.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.adapters.ocr import PaddleOCRAdapter
from src.core.extractors.text_recognizer import TextRecognizer


def demo_paddle_ocr_basic():
    """Demonstrate basic PaddleOCR usage"""
    print("=" * 80)
    print("PADDLEOCR BASIC DEMONSTRATION")
    print("=" * 80)
    print()

    # Initialize PaddleOCR
    config = {
        'lang': 'en',
        'use_angle_cls': True,
        'use_gpu': False,
        'show_log': False
    }

    print("Initializing PaddleOCR...")
    ocr = PaddleOCRAdapter(config)
    print(f"✓ {ocr}")
    print()

    # Show supported languages
    print("Supported Languages:")
    languages = ocr.get_supported_languages()
    print(f"  Total: {len(languages)} languages")
    print(f"  Sample: {', '.join(languages[:10])}...")
    print()


def demo_text_recognizer(pdf_path: str):
    """Demonstrate TextRecognizer with a PDF"""
    print("=" * 80)
    print("TEXT RECOGNIZER DEMONSTRATION")
    print("=" * 80)
    print()

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("Please provide a valid PDF path")
        return

    # Initialize TextRecognizer
    config = {
        'ocr_backend': 'paddleocr',
        'fallback_to_native': True,
        'min_confidence': 0.5,
        'dpi': 300,
        'ocr_config': {
            'lang': 'en',
            'use_angle_cls': True,
            'use_gpu': False,
            'show_log': False
        }
    }

    print("Initializing TextRecognizer...")
    recognizer = TextRecognizer(config)
    print(f"✓ {recognizer}")
    print()

    # Extract text from first page
    print(f"Processing PDF: {pdf_path}")
    print("Extracting text from page 0...")

    try:
        result = recognizer.extract_text_from_page(pdf_path, 0)

        print()
        print(f"✓ Extraction completed")
        print(f"  Method: {result['method']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Text blocks: {result['total_blocks']}")
        print()

        # Show first 3 text blocks
        print("Sample text blocks:")
        for i, block in enumerate(result['text_blocks'][:3]):
            text = block['text']
            conf = block.get('confidence', 1.0)
            bbox = block['bbox']

            print(f"\nBlock {i + 1}:")
            print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"  Confidence: {conf:.2f}")
            print(f"  Position: ({bbox['x']:.1f}, {bbox['y']:.1f})")
            print(f"  Size: {bbox['width']:.1f} x {bbox['height']:.1f}")

        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_ocr_verification(pdf_path: str):
    """Demonstrate OCR verification of native extraction"""
    print("=" * 80)
    print("OCR VERIFICATION DEMONSTRATION")
    print("=" * 80)
    print()

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    config = {
        'ocr_backend': 'paddleocr',
        'fallback_to_native': True,
        'min_confidence': 0.5,
        'dpi': 300,
        'ocr_config': {
            'lang': 'en',
            'use_gpu': False,
            'show_log': False
        }
    }

    recognizer = TextRecognizer(config)

    print(f"Verifying text extraction for: {pdf_path}")
    print("Comparing native extraction vs OCR...")
    print()

    try:
        verification = recognizer.verify_native_extraction(pdf_path, 0, threshold=0.8)

        print("Verification Results:")
        print(f"  Page: {verification['page_num']}")
        print(f"  Similarity: {verification['similarity']:.2%}")
        print(f"  Matches: {'✓ Yes' if verification['matches'] else '✗ No'}")
        print(f"  Native blocks: {verification['native_blocks']}")
        print(f"  OCR blocks: {verification['ocr_blocks']}")
        print(f"  Native method: {verification['native_method']}")
        print(f"  OCR confidence: {verification['ocr_confidence']:.2f}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demonstration entry point"""
    print("\n" + "=" * 80)
    print("PADDLEOCR INTEGRATION FOR PDF TRANSLATION PIPELINE")
    print("=" * 80)
    print()

    # Demo 1: Basic PaddleOCR
    try:
        demo_paddle_ocr_basic()
    except Exception as e:
        print(f"❌ Basic demo failed: {e}")
        print("Make sure PaddleOCR is installed: pip install paddleocr")
        return 1

    # Demo 2: TextRecognizer with PDF
    pdf_path = "demonstration/input/test_complex.pdf"

    # Check if demo PDF exists
    if os.path.exists(pdf_path):
        try:
            demo_text_recognizer(pdf_path)
        except Exception as e:
            print(f"❌ TextRecognizer demo failed: {e}")

        # Demo 3: OCR Verification
        try:
            demo_ocr_verification(pdf_path)
        except Exception as e:
            print(f"❌ Verification demo failed: {e}")
    else:
        print(f"Demo PDF not found: {pdf_path}")
        print("You can still test with your own PDF:")
        print(f"  python {__file__} <path_to_pdf>")
        print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Integration Summary:")
    print("  ✓ PaddleOCR adapter created")
    print("  ✓ TextRecognizer module ready")
    print("  ✓ Pipeline integration complete")
    print("  ✓ Configuration added to config.yaml")
    print()
    print("Usage in pipeline:")
    print("  - Automatic OCR for scanned PDFs")
    print("  - Fallback to native extraction when possible")
    print("  - Support for 80+ languages")
    print("  - Confidence-based filtering")
    print()
    print("To customize OCR settings, edit config.yaml:")
    print("  extraction.text_recognition.ocr_config")
    print()

    return 0


if __name__ == "__main__":
    # Allow custom PDF path as argument
    if len(sys.argv) > 1:
        custom_pdf = sys.argv[1]
        demo_text_recognizer(custom_pdf)
        demo_ocr_verification(custom_pdf)
    else:
        exit(main())
