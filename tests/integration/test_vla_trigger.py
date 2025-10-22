#!/usr/bin/env python3
"""
VLA Trigger Integration Tests - Simplified
Tests document complexity analysis and VLA decision making
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.deciders import VLATrigger, ComplexityLevel


def get_test_case(case_type):
    """Factory for test images and extraction data"""
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255

    if case_type == "simple":
        # Single column, plain text
        for i in range(10):
            cv2.rectangle(image, (50, 100 + i*60), (550, 130 + i*60), (0, 0, 0), -1)
        extraction = {
            'text_blocks': [{'text': 'text', 'confidence': 0.95, 'font': 'Arial', 'font_size': 12}] * 10,
            'images': [], 'tables': [], 'formulas': [], 'charts': []
        }

    elif case_type == "complex":
        # Multi-column + mixed content
        for i in range(8):
            cv2.rectangle(image, (30, 50 + i*90), (270, 110 + i*90), (0, 0, 0), -1)
            cv2.rectangle(image, (320, 50 + i*90), (570, 110 + i*90), (0, 0, 0), -1)
        cv2.rectangle(image, (100, 650), (500, 750), (128, 128, 128), -1)
        extraction = {
            'text_blocks': [{'text': f'text{i}', 'confidence': 0.9, 'font': 'Arial', 'font_size': 12}] * 20,
            'images': [{'bbox': [100, 650, 500, 750]}],
            'tables': [{'bbox': [50, 100, 300, 400]}],
            'formulas': [], 'charts': []
        }

    elif case_type == "blurry":
        # Quality issues - blur
        for i in range(10):
            cv2.rectangle(image, (50, 100 + i*60), (550, 130 + i*60), (0, 0, 0), -1)
        image = cv2.GaussianBlur(image, (25, 25), 0)
        extraction = {
            'text_blocks': [{'text': 'blur', 'confidence': 0.6, 'font': 'Arial', 'font_size': 12}] * 10,
            'images': [], 'tables': [], 'formulas': [], 'charts': []
        }

    elif case_type == "skewed":
        # Quality issues - skew
        for i in range(10):
            cv2.rectangle(image, (50, 100 + i*60), (550, 130 + i*60), (0, 0, 0), -1)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
        extraction = {
            'text_blocks': [{'text': 'skew', 'confidence': 0.75, 'font': 'Arial', 'font_size': 12}] * 10,
            'images': [], 'tables': [], 'formulas': [], 'charts': []
        }

    return image, extraction


def test_document_analysis():
    """Test VLA trigger on various document types"""
    print("\n=== Document Analysis Tests ===")

    trigger = VLATrigger()
    test_cases = [
        ("simple", False, [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE], "Simple document"),
        ("complex", None, [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME], "Complex document"),
        ("blurry", None, None, "Blurry document (quality issues)"),
        ("skewed", None, None, "Skewed document (quality issues)"),
    ]

    for case_type, expect_vla, expect_levels, desc in test_cases:
        image, extraction = get_test_case(case_type)
        decision = trigger.analyze_document(image, extraction)

        print(f"\n  {desc}:")
        print(f"    Level: {decision.complexity_level.name}, VLA: {decision.use_vla}, Model: {decision.recommended_model}")

        # Validate expectations
        if expect_vla is not None:
            assert decision.use_vla == expect_vla, f"{desc}: expected VLA={expect_vla}"
        if expect_levels:
            assert decision.complexity_level in expect_levels, \
                f"{desc}: expected one of {[l.name for l in expect_levels]}, got {decision.complexity_level.name}"

        print(f"    ✓ Passed")

    print("\n  ✅ All document analysis tests passed")


def test_core_logic():
    """Test analyzers, thresholds, and model selection"""
    print("\n=== Core Logic Tests ===")

    trigger = VLATrigger()

    # Test 1: Mixed content analyzer
    extraction_mixed = {
        'text_blocks': [{'text': 'text'}],
        'images': [{'bbox': [0, 0, 100, 100]}],
        'tables': [{'bbox': [0, 0, 200, 200]}],
        'formulas': [], 'charts': []
    }
    assert trigger._analyze_mixed_content(extraction_mixed) == 0.6, "Mixed content should return 0.6 for 3 types"
    print("  ✓ Mixed content analyzer")

    # Test 2: OCR confidence analyzer
    extraction_low_conf = {
        'text_blocks': [
            {'text': 'text', 'confidence': 0.5},
            {'text': 'text', 'confidence': 0.6},
            {'text': 'text', 'confidence': 0.55},
        ]
    }
    assert trigger._analyze_ocr_confidence(extraction_low_conf) > 0.3, "Low confidence should be detected"
    print("  ✓ OCR confidence analyzer")

    # Test 3: Layout complexity analyzer
    image, _ = get_test_case("complex")
    assert trigger._analyze_layout_complexity(image) >= 0, "Layout complexity should be valid"
    print("  ✓ Layout complexity analyzer")

    # Test 4: Model selection
    model_tests = [
        (ComplexityLevel.SIMPLE, 'paddleocr'),
        (ComplexityLevel.MODERATE, 'surya'),
        (ComplexityLevel.COMPLEX, 'mplug-docowl'),
        (ComplexityLevel.EXTREME, 'internvl-2.0'),
    ]
    for level, expected_model in model_tests:
        assert trigger.model_mapping[level] == expected_model, f"Wrong model for {level.name}"
    print("  ✓ Model selection")

    # Test 5: Decision thresholds
    threshold_tests = [
        (0.2, ComplexityLevel.SIMPLE, False),
        (0.4, ComplexityLevel.MODERATE, False),
        (0.6, ComplexityLevel.COMPLEX, True),
        (0.9, ComplexityLevel.EXTREME, True),
    ]
    for score, expected_level, expected_vla in threshold_tests:
        factors = {k: score for k in trigger.complexity_weights.keys()}
        decision = trigger.make_vla_decision(factors)
        assert decision.complexity_level == expected_level, \
            f"Score {score}: expected {expected_level.name}, got {decision.complexity_level.name}"
        assert decision.use_vla == expected_vla, \
            f"Score {score}: expected VLA={expected_vla}, got {decision.use_vla}"
    print("  ✓ Decision thresholds")

    print("\n  ✅ All core logic tests passed")


def main():
    """Run all VLA trigger tests"""
    print("="*60)
    print("VLA Trigger Integration Tests")
    print("="*60)

    tests = [
        test_document_analysis,
        test_core_logic,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} test suites passed")
    print("="*60)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST SUITE(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
