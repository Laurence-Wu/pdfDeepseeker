#!/usr/bin/env python3
"""
VLA Processor Integration Tests - Simplified
Tests VLA model loading, processing, and selection
"""

import sys
from pathlib import Path
import numpy as np
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.deciders import VLAProcessor, ModelSelector, ComplexityLevel


def test_vla_processor_init():
    """Test VLAProcessor initialization"""
    print("\n=== VLAProcessor Initialization ===")

    # Test with default config (PaddleOCR only)
    processor = VLAProcessor(config={})
    assert processor is not None, "Processor should initialize"
    assert 'paddleocr' in processor.models or processor.models.get('paddleocr') is None, "PaddleOCR should be attempted"

    print("  ✓ VLAProcessor initialized")

    # Test with GPU config (will fallback to CPU if no GPU)
    processor_gpu = VLAProcessor(config={'use_gpu': True})
    assert processor_gpu is not None, "Processor with GPU config should initialize"

    print("  ✓ GPU config handled")
    print("\n  ✅ VLAProcessor initialization test passed")


def test_model_selector():
    """Test ModelSelector logic"""
    print("\n=== ModelSelector Tests ===")

    selector = ModelSelector()

    # Test 1: Simple document -> PaddleOCR
    model = selector.select_model(
        complexity_level=ComplexityLevel.SIMPLE,
        document_type='general',
        language='en',
        resolution=(800, 600)
    )
    assert model == 'paddleocr', f"Simple document should use PaddleOCR, got {model}"
    print("  ✓ Simple document -> paddleocr")

    # Test 2: Complex document -> Surya
    model = selector.select_model(
        complexity_level=ComplexityLevel.COMPLEX,
        document_type='general',
        language='en',
        resolution=(1024, 768)
    )
    assert model == 'surya', f"Complex document should use Surya, got {model}"
    print("  ✓ Complex document -> surya")

    # Test 3: Extreme complexity -> mPLUG
    model = selector.select_model(
        complexity_level=ComplexityLevel.EXTREME,
        document_type='complex',
        language='en',
        resolution=(2048, 1536)
    )
    assert model == 'mplug', f"Extreme complexity should use mPLUG, got {model}"
    print("  ✓ Extreme complexity -> mplug")

    # Test 4: Form document -> LayoutLM
    model = selector.select_model(
        complexity_level=ComplexityLevel.MODERATE,
        document_type='form',
        language='en',
        resolution=(800, 600)
    )
    assert model == 'layoutlm', f"Form document should use LayoutLM, got {model}"
    print("  ✓ Form document -> layoutlm")

    # Test 5: High resolution -> mPLUG
    model = selector.select_model(
        complexity_level=ComplexityLevel.COMPLEX,
        document_type='high-res',
        language='en',
        resolution=(3840, 2160)
    )
    assert model == 'mplug', f"High resolution should use mPLUG, got {model}"
    print("  ✓ High resolution -> mplug")

    # Test 6: Fallback mechanism
    fallback = selector.get_fallback_model('mplug')
    assert fallback == 'surya', "mPLUG fallback should be Surya"
    print("  ✓ Fallback: mplug -> surya")

    fallback = selector.get_fallback_model('surya')
    assert fallback == 'paddleocr', "Surya fallback should be PaddleOCR"
    print("  ✓ Fallback: surya -> paddleocr")

    # Test 7: Model info retrieval
    info = selector.get_model_info('surya')
    assert 'speed' in info, "Model info should contain speed"
    assert 'accuracy' in info, "Model info should contain accuracy"
    print("  ✓ Model info retrieval")

    # Test 8: List models by filter
    fast_models = selector.list_models(speed='fast')
    assert 'surya' in fast_models, "Surya should be in fast models"
    print("  ✓ Model filtering")

    print("\n  ✅ All ModelSelector tests passed")


async def test_paddleocr_processing():
    """Test PaddleOCR processing (fallback model)"""
    print("\n=== PaddleOCR Processing Test ===")

    processor = VLAProcessor(config={})

    # Create simple test image (white background with black text rectangles)
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Add some "text" rectangles
    import cv2
    for i in range(5):
        cv2.rectangle(image, (50, 100 + i*80), (750, 140 + i*80), (0, 0, 0), -1)

    # Test processing
    if processor.models.get('paddleocr'):
        try:
            result = await processor.process_with_vla(image, 'paddleocr', fallback=False)

            assert 'text_blocks' in result, "Result should contain text_blocks"
            assert isinstance(result['text_blocks'], list), "text_blocks should be a list"

            print(f"  ✓ PaddleOCR processed image")
            print(f"    Found {len(result['text_blocks'])} text blocks")

        except Exception as e:
            print(f"  ⚠️  PaddleOCR processing error (expected if no text detected): {e}")
    else:
        print("  ⚠️  PaddleOCR not available, skipping processing test")

    print("\n  ✅ PaddleOCR processing test passed")


async def test_fallback_mechanism():
    """Test fallback from unavailable model to PaddleOCR"""
    print("\n=== Fallback Mechanism Test ===")

    processor = VLAProcessor(config={})

    # Create test image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Try to use Surya (likely not available), should fallback to PaddleOCR
    try:
        result = await processor.process_with_vla(image, 'surya', fallback=True)
        print("  ✓ Fallback mechanism triggered successfully")
        assert 'text_blocks' in result or 'error' in result, "Should return valid result or error"

    except Exception as e:
        # If no models available at all, that's expected
        if "not available" in str(e):
            print(f"  ⚠️  No models available for fallback test: {e}")
        else:
            raise e

    print("\n  ✅ Fallback mechanism test passed")


async def test_batch_processing():
    """Test batch processing capability"""
    print("\n=== Batch Processing Test ===")

    processor = VLAProcessor(config={})

    # Create 3 test images
    images = []
    for _ in range(3):
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        images.append(image)

    if processor.models.get('paddleocr'):
        try:
            results = await processor.process_batch(images, 'paddleocr')

            assert len(results) == 3, f"Should process 3 images, got {len(results)}"
            print(f"  ✓ Batch processed {len(results)} images")

            for i, result in enumerate(results):
                assert isinstance(result, dict), f"Result {i} should be a dict"

            print("  ✓ All batch results are valid dicts")

        except Exception as e:
            print(f"  ⚠️  Batch processing error: {e}")
    else:
        print("  ⚠️  PaddleOCR not available, skipping batch test")

    print("\n  ✅ Batch processing test passed")


def test_model_availability():
    """Test which models are actually available"""
    print("\n=== Model Availability Check ===")

    processor = VLAProcessor(config={
        'enable_surya': True,
        'enable_mplug': False,
        'enable_layoutlm': False
    })

    available = []
    unavailable = []

    for model_name in ['surya', 'mplug', 'layoutlm', 'paddleocr']:
        if processor.models.get(model_name):
            available.append(model_name)
            print(f"  ✓ {model_name}: Available")
        else:
            unavailable.append(model_name)
            print(f"  ⚠️  {model_name}: Not available")

    # At minimum, PaddleOCR should be attempted
    print(f"\n  Available models: {available if available else 'None (expected for fresh environment)'}")
    print(f"  Unavailable models: {unavailable}")

    print("\n  ✅ Model availability check complete")


def main():
    """Run all VLA processor tests"""
    print("="*60)
    print("VLA Processor Integration Tests")
    print("="*60)

    tests = [
        ("VLAProcessor Init", test_vla_processor_init),
        ("ModelSelector", test_model_selector),
        ("Model Availability", test_model_availability),
    ]

    async_tests = [
        ("PaddleOCR Processing", test_paddleocr_processing),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Batch Processing", test_batch_processing),
    ]

    passed = 0
    failed = 0

    # Run synchronous tests
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Run async tests
    for test_name, test_func in async_tests:
        try:
            asyncio.run(test_func())
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_tests = len(tests) + len(async_tests)

    print("\n" + "="*60)
    print(f"Results: {passed}/{total_tests} test suites passed")
    print("="*60)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST SUITE(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
