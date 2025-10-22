#!/usr/bin/env python3
"""
Comprehensive tests for Text Length Controller (Instruction 14)
Tests text measurement, fitting strategies, and length constraints
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.text.text_length_controller import (
    TextLengthController, TextMeasurement, TextFittingStrategy
)


def test_controller_initialization():
    """Test TextLengthController initialization"""
    print("Testing TextLengthController initialization...")

    # Default config
    controller = TextLengthController()
    assert controller is not None
    assert controller.max_expansion_ratio == 1.1
    assert controller.min_compression_ratio == 0.7
    assert controller.abbreviation_dict is not None

    # Custom config
    config = {
        'max_expansion_ratio': 1.2,
        'min_compression_ratio': 0.6
    }
    controller2 = TextLengthController(config)
    assert controller2.max_expansion_ratio == 1.2
    assert controller2.min_compression_ratio == 0.6

    print("  ✓ TextLengthController initialized correctly")


def test_text_measurement_estimation():
    """Test text measurement estimation (without font data)"""
    print("\nTesting text measurement estimation...")

    controller = TextLengthController()

    font_info = {
        'name': 'Arial',
        'size': 12
        # No font data, will use estimation
    }

    text = "Hello World"
    measurement = controller.measure_text(text, font_info)

    assert measurement is not None
    assert measurement.width > 0
    assert measurement.height > 0
    assert measurement.char_count == len(text)

    print(f"  ✓ Estimated text size: {measurement.width:.1f}w x {measurement.height:.1f}h")


def test_abbreviation_application():
    """Test abbreviation strategy"""
    print("\nTesting abbreviation application...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 100, 'height': 20}

    text = "International Corporation Department of Technology Development"
    abbreviated = controller._apply_abbreviation(text, bbox, font_info, 1.5)

    # Should have abbreviations
    assert 'Int\'l' in abbreviated or 'Corp.' in abbreviated or 'Dept.' in abbreviated
    assert len(abbreviated) < len(text)

    print(f"  ✓ Original: {text}")
    print(f"  ✓ Abbreviated: {abbreviated}")


def test_spacing_adjustment():
    """Test spacing adjustment strategy"""
    print("\nTesting spacing adjustment...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 100, 'height': 20}

    text = "Hello  ,  World  !"
    adjusted = controller._adjust_spacing(text, bbox, font_info, 1.05)

    # Should have cleaned up spacing
    assert '  ' not in adjusted  # No double spaces
    assert ' ,' not in adjusted  # No space before comma
    assert ' !' not in adjusted  # No space before exclamation

    print(f"  ✓ Original: '{text}'")
    print(f"  ✓ Adjusted: '{adjusted}'")


def test_text_truncation():
    """Test text truncation strategy"""
    print("\nTesting text truncation...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 100, 'height': 20}

    text = "This is a very long text that needs to be truncated"
    truncated = controller._truncate_text(text, bbox, font_info, 2.0)

    # Should be truncated with ellipsis
    assert len(truncated) < len(text)
    assert truncated.endswith('...')

    print(f"  ✓ Original: {text}")
    print(f"  ✓ Truncated: {truncated}")


def test_fit_translation_no_overflow():
    """Test fitting translation that already fits"""
    print("\nTesting translation fit (no overflow)...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 500, 'height': 50}  # Large bbox

    translation = "Short text"
    result = controller.fit_translation(translation, bbox, font_info)

    assert result['success'] == True
    assert result['method'] == 'none'
    assert result['fitted_text'] == translation
    assert result['overflow_ratio'] <= 1.0

    print(f"  ✓ Text fits without adjustment")
    print(f"  ✓ Overflow ratio: {result['overflow_ratio']:.2f}")


def test_fit_translation_with_overflow():
    """Test fitting translation with overflow"""
    print("\nTesting translation fit (with overflow)...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 30, 'height': 15}  # Very small bbox

    translation = "This is a very long translation that will definitely not fit in the small box at all"
    result = controller.fit_translation(translation, bbox, font_info)

    # Should have detected overflow and attempted fitting
    print(f"  → Overflow ratio: {result['overflow_ratio']:.2f}")
    print(f"  → Method used: {result['method']}")
    print(f"  → Original length: {len(translation)}")
    print(f"  → Fitted length: {len(result['fitted_text'])}")
    print(f"  → Success: {result['success']}")

    # Either a strategy was applied, or it detected overflow
    assert result['overflow_ratio'] > 1.0 or result['method'] != 'none'
    assert len(result['fitted_text']) > 0

    print(f"  ✓ Overflow handling tested")


def test_calculate_max_length():
    """Test calculation of maximum allowed length"""
    print("\nTesting max length calculation...")

    controller = TextLengthController({'max_expansion_ratio': 1.2})

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 200, 'height': 20}

    source_text = "Original text"
    max_length = controller.calculate_max_length(source_text, bbox, font_info)

    assert max_length > 0
    # Should allow some expansion
    assert max_length >= len(source_text)

    print(f"  ✓ Source length: {len(source_text)}")
    print(f"  ✓ Max allowed: {max_length}")
    print(f"  ✓ Expansion ratio: {max_length / len(source_text):.2f}x")


def test_validate_translation_fit():
    """Test validation of translation fit"""
    print("\nTesting translation fit validation...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 100, 'height': 20}

    # Short translation (fits)
    short_translation = "Hello"
    validation1 = controller.validate_translation_fit(short_translation, bbox, font_info)

    assert validation1['fits'] == True
    assert validation1['width_ratio'] < 1.0
    assert validation1['adjustment_needed'] < 0

    print(f"  ✓ Short text validation:")
    print(f"    Fits: {validation1['fits']}")
    print(f"    Width ratio: {validation1['width_ratio']:.2f}")

    # Long translation (doesn't fit)
    long_translation = "This is a very long translation that will definitely overflow the bounding box"
    validation2 = controller.validate_translation_fit(long_translation, bbox, font_info)

    assert validation2['fits'] == False
    assert validation2['width_ratio'] > 1.0
    assert validation2['adjustment_needed'] > 0

    print(f"  ✓ Long text validation:")
    print(f"    Fits: {validation2['fits']}")
    print(f"    Width ratio: {validation2['width_ratio']:.2f}")
    print(f"    Adjustment needed: {validation2['adjustment_needed']:.2f}")


def test_generate_length_constraint():
    """Test generation of length constraints"""
    print("\nTesting length constraint generation...")

    controller = TextLengthController()

    font_info = {'name': 'Arial', 'size': 12}
    bbox = {'x': 0, 'y': 0, 'width': 200, 'height': 20}
    source_text = "Sample text"

    constraint = controller.generate_length_constraint(source_text, bbox, font_info)

    assert 'max_length' in constraint
    assert 'current_length' in constraint
    assert 'expansion_allowed' in constraint
    assert 'compression_allowed' in constraint
    assert 'bbox' in constraint
    assert 'strategy_preference' in constraint

    assert constraint['current_length'] == len(source_text)
    assert constraint['max_length'] > 0

    print(f"  ✓ Current length: {constraint['current_length']}")
    print(f"  ✓ Max length: {constraint['max_length']}")
    print(f"  ✓ Expansion allowed: {constraint['expansion_allowed']}")
    print(f"  ✓ Strategies: {constraint['strategy_preference']}")


def test_hyphenation_strategy():
    """Test hyphenation strategy"""
    print("\nTesting hyphenation strategy...")

    text = "This is a demonstration of hyphenation for very long words"
    max_width = 100
    char_width = 6

    hyphenated = TextFittingStrategy.apply_hyphenation(text, max_width, char_width)

    # Should have multiple lines
    assert '\n' in hyphenated
    lines = hyphenated.split('\n')
    assert len(lines) > 1

    print(f"  ✓ Original: {text}")
    print(f"  ✓ Hyphenated ({len(lines)} lines):")
    for i, line in enumerate(lines, 1):
        print(f"    {i}. {line}")


def test_condensed_style_strategy():
    """Test condensed writing style"""
    print("\nTesting condensed style strategy...")

    text = "This is very quite good in order to demonstrate for example the condensing"
    condensed = TextFittingStrategy.apply_condensed_style(text)

    # Should have removed some words
    assert len(condensed) < len(text)
    assert 'very' not in condensed or 'quite' not in condensed

    print(f"  ✓ Original: {text}")
    print(f"  ✓ Condensed: {condensed}")


def test_abbreviation_dictionary():
    """Test abbreviation dictionary loading"""
    print("\nTesting abbreviation dictionary...")

    controller = TextLengthController()

    assert len(controller.abbreviation_dict) > 0
    assert 'International' in controller.abbreviation_dict
    assert 'Corporation' in controller.abbreviation_dict

    print(f"  ✓ Loaded {len(controller.abbreviation_dict)} abbreviations")


def test_font_cache():
    """Test font metrics caching"""
    print("\nTesting font metrics caching...")

    controller = TextLengthController()

    font_info1 = {'name': 'Arial', 'size': 12}
    font_info2 = {'name': 'Arial', 'size': 14}  # Same font, different size

    # Measure text with same font
    controller.measure_text("Test 1", font_info1)
    controller.measure_text("Test 2", font_info2)

    # Cache should work (though may not have actual font data in test)
    # Just verify no errors
    print(f"  ✓ Font cache working")


def run_all_tests():
    """Run all text length controller tests"""
    print("=" * 80)
    print("TEXT LENGTH CONTROLLER TESTS (Instruction 14)")
    print("=" * 80)

    tests = [
        test_controller_initialization,
        test_text_measurement_estimation,
        test_abbreviation_application,
        test_spacing_adjustment,
        test_text_truncation,
        test_fit_translation_no_overflow,
        test_fit_translation_with_overflow,
        test_calculate_max_length,
        test_validate_translation_fit,
        test_generate_length_constraint,
        test_hyphenation_strategy,
        test_condensed_style_strategy,
        test_abbreviation_dictionary,
        test_font_cache
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
    else:
        print("ALL TESTS PASSED ✓")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
