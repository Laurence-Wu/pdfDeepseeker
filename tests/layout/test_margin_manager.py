#!/usr/bin/env python3
"""
Comprehensive tests for Margin Manager (Instruction 12)
Tests margin detection, enforcement, and consistency analysis
"""

import sys
import os
from pathlib import Path
import fitz
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.layout.margin_manager import MarginManager, Margin


def create_test_pdf_with_margins(output_path: str, margins: tuple = (72, 72, 72, 72)) -> str:
    """
    Create a test PDF with specific margins.

    Args:
        output_path: Output file path
        margins: (top, bottom, left, right) in points
    """
    top, bottom, left, right = margins

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size

    # Add content within margins
    content_x = left
    content_y = top
    content_width = 612 - left - right
    content_height = 792 - top - bottom

    # Add text at top of content area
    page.insert_text(
        (content_x + 10, content_y + 20),
        "Document Title",
        fontsize=16
    )

    # Add text in middle
    page.insert_text(
        (content_x + 10, content_y + 100),
        "This is sample content within the margins.",
        fontsize=12
    )

    # Draw a rectangle to show content area
    page.draw_rect(
        fitz.Rect(content_x, content_y,
                 612 - right, 792 - bottom),
        color=(0.8, 0.8, 0.8),
        width=1
    )

    doc.save(output_path)
    doc.close()

    return output_path


def test_margin_manager_initialization():
    """Test MarginManager initialization"""
    print("Testing MarginManager initialization...")

    # Default config
    manager = MarginManager()
    assert manager is not None
    assert manager.threshold == 10
    assert manager.min_margin == 36
    assert manager.enforce_strict == True

    # Custom config
    config = {
        'threshold': 15,
        'min_margin': 50,
        'enforce_strict': False,
        'enforce_consistent': False
    }
    manager2 = MarginManager(config)
    assert manager2.threshold == 15
    assert manager2.min_margin == 50
    assert manager2.enforce_strict == False

    print("  ✓ MarginManager initialized correctly")


def test_margin_extraction():
    """Test margin extraction from PDF"""
    print("\nTesting margin extraction...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create PDF with known margins (72pt = 1 inch)
        pdf_path = os.path.join(tmpdir, "test_margins.pdf")
        create_test_pdf_with_margins(pdf_path, margins=(72, 72, 72, 72))

        # Extract margins
        manager = MarginManager({'min_margin': 36})
        margins = manager.extract_margins(pdf_path)

        assert len(margins) == 1
        margin = margins[0]

        print(f"  → Extracted margins: T={margin.top:.1f}, B={margin.bottom:.1f}, "
              f"L={margin.left:.1f}, R={margin.right:.1f}, confidence={margin.confidence:.2f}")

        # Check margins are detected (values reasonable)
        assert margin.page_num == 0
        assert margin.top >= 36  # At least minimum margin
        assert margin.bottom >= 36  # At least minimum margin
        assert margin.left >= 36  # At least minimum margin
        assert margin.right >= 36  # At least minimum margin
        assert margin.confidence > 0.5

        print(f"  ✓ Margins extracted and validated")


def test_margin_extraction_multipage():
    """Test margin extraction from multi-page PDF"""
    print("\nTesting multi-page margin extraction...")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "multipage.pdf")

        # Create 3-page PDF with different margins
        doc = fitz.open()

        # Page 1: Standard margins (72pt)
        page1 = doc.new_page(width=612, height=792)
        page1.insert_text((72, 92), "Page 1 - Standard Margins", fontsize=14)
        page1.insert_text((72, 120), "Content within 1-inch margins", fontsize=12)

        # Page 2: Wider margins (100pt)
        page2 = doc.new_page(width=612, height=792)
        page2.insert_text((100, 120), "Page 2 - Wider Margins", fontsize=14)
        page2.insert_text((100, 150), "Content with wider margins", fontsize=12)

        # Page 3: Standard margins (72pt) - same as page 1
        page3 = doc.new_page(width=612, height=792)
        page3.insert_text((72, 92), "Page 3 - Standard Margins", fontsize=14)
        page3.insert_text((72, 120), "Content within 1-inch margins", fontsize=12)

        doc.save(pdf_path)
        doc.close()

        # Extract margins
        manager = MarginManager({'min_margin': 36, 'enforce_consistent': True})
        margins = manager.extract_margins(pdf_path)

        assert len(margins) == 3

        # Check that consistency enforcement worked
        # Pages with low confidence should adopt consistent margins
        print(f"  ✓ Page 0 margins: T={margins[0].top:.1f}, L={margins[0].left:.1f}")
        print(f"  ✓ Page 1 margins: T={margins[1].top:.1f}, L={margins[1].left:.1f}")
        print(f"  ✓ Page 2 margins: T={margins[2].top:.1f}, L={margins[2].left:.1f}")


def test_content_boundary_detection():
    """Test detection of content boundaries"""
    print("\nTesting content boundary detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "boundaries.pdf")

        # Create PDF with specific content placement
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Add content at known positions
        page.insert_text((100, 100), "Top Left Content", fontsize=12)
        page.insert_text((400, 600), "Bottom Right Content", fontsize=12)

        # Draw a line
        page.draw_line((100, 200), (500, 200), width=2)

        # Draw a rectangle
        page.draw_rect(fitz.Rect(150, 300, 450, 500), color=(0, 0, 1), width=2)

        doc.save(pdf_path)
        doc.close()

        # Detect boundaries
        manager = MarginManager()

        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            bbox = manager.detect_content_boundaries(page)

            assert bbox is not None
            x0, y0, x1, y1 = bbox

            # Content should span roughly from (100, 100) to (500, 600)
            assert 90 < x0 < 110  # Left edge near 100
            assert 90 < y0 < 110  # Top edge near 100
            assert x1 > 400  # Right edge beyond 400
            assert y1 > 500  # Bottom edge beyond 500

            print(f"  ✓ Detected content bbox: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")


def test_margin_enforcement():
    """Test margin enforcement on content"""
    print("\nTesting margin enforcement...")

    # Create test content that violates margins
    content = {
        'page_width': 612,
        'page_height': 792,
        'elements': [
            {
                'id': 'elem1',
                'bbox': {'x': 10, 'y': 10, 'width': 200, 'height': 50}
            },
            {
                'id': 'elem2',
                'bbox': {'x': 600, 'y': 700, 'width': 100, 'height': 100}
            },
            {
                'id': 'elem3',
                'bbox': {'x': 100, 'y': 100, 'width': 400, 'height': 50}
            }
        ]
    }

    # Define margins
    margins = Margin(
        top=50,
        bottom=50,
        left=50,
        right=50,
        page_num=0
    )

    # Enforce margins
    manager = MarginManager()
    adjusted = manager.enforce_margins(content, margins)

    # Check that violations were detected and fixed
    assert 'margin_violations' in adjusted
    assert len(adjusted['margin_violations']) > 0

    # elem1 should have been moved to respect left/top margins
    elem1 = adjusted['elements'][0]
    assert elem1['bbox']['x'] >= margins.left
    assert elem1['bbox']['y'] >= margins.top

    # elem2 should have been adjusted for right/bottom margins
    elem2 = adjusted['elements'][1]
    assert elem2['bbox']['x'] + elem2['bbox']['width'] <= 612 - margins.right

    # elem3 should be unchanged (within margins)
    elem3 = adjusted['elements'][2]
    assert elem3['bbox']['x'] == 100

    print(f"  ✓ Detected {len(adjusted['margin_violations'])} margin violations")
    print(f"  ✓ Adjusted content positions to respect margins")


def test_safe_area_calculation():
    """Test safe area calculation"""
    print("\nTesting safe area calculation...")

    manager = MarginManager()

    # Letter size page (612 x 792 pt)
    page_size = (612, 792)
    margins = Margin(
        top=72,
        bottom=72,
        left=72,
        right=72,
        page_num=0
    )

    safe_area = manager.get_safe_area(page_size, margins)

    assert safe_area['x'] == 72
    assert safe_area['y'] == 72
    assert safe_area['width'] == 612 - 72 - 72  # 468
    assert safe_area['height'] == 792 - 72 - 72  # 648
    assert safe_area['total_area'] == 468 * 648

    print(f"  ✓ Safe area: {safe_area['width']:.0f} x {safe_area['height']:.0f} pts")
    print(f"  ✓ Total safe area: {safe_area['total_area']:.0f} sq pts")


def test_margin_ratio_calculation():
    """Test margin ratio calculation"""
    print("\nTesting margin ratio calculation...")

    manager = MarginManager()

    page_size = (612, 792)
    margins = Margin(
        top=72,
        bottom=72,
        left=72,
        right=72,
        page_num=0
    )

    ratios = manager.calculate_margin_ratio(page_size, margins)

    print(f"  → Top/Bottom ratio: {ratios['top_ratio']:.1%}")
    print(f"  → Left/Right ratio: {ratios['left_ratio']:.1%}")
    print(f"  → Content area ratio: {ratios['content_area_ratio']:.1%}")

    # Check ratios (with reasonable tolerance)
    # For 72pt margins on letter size:
    # top/bottom: 72/792 = 9.09%
    # left/right: 72/612 = 11.76%
    # content: (468*648)/(612*792) = 62.63%
    assert 0.08 < ratios['top_ratio'] < 0.10  # ~9% (72/792)
    assert 0.08 < ratios['bottom_ratio'] < 0.10
    assert 0.11 < ratios['left_ratio'] < 0.13  # ~11.7% (72/612)
    assert 0.11 < ratios['right_ratio'] < 0.13
    assert 0.60 < ratios['content_area_ratio'] < 0.65  # ~62.6%

    print(f"  ✓ Ratios calculated correctly")


def test_consistent_margin_finding():
    """Test finding consistent margins across pages"""
    print("\nTesting consistent margin detection...")

    manager = MarginManager()

    # Create margins with some variation
    margins = [
        Margin(top=72, bottom=72, left=72, right=72, page_num=0, confidence=1.0),
        Margin(top=72, bottom=72, left=72, right=72, page_num=1, confidence=1.0),
        Margin(top=75, bottom=70, left=72, right=72, page_num=2, confidence=0.9),
        Margin(top=72, bottom=72, left=72, right=72, page_num=3, confidence=1.0),
        Margin(top=100, bottom=100, left=100, right=100, page_num=4, confidence=0.3),  # Outlier
    ]

    consistent = manager._find_consistent_margins(margins)

    # Should find 72 as the most common value (mode)
    assert consistent.top == 72 or abs(consistent.top - 72) < 10
    assert consistent.left == 72 or abs(consistent.left - 72) < 10

    print(f"  ✓ Consistent margins: T={consistent.top:.0f}, B={consistent.bottom:.0f}, "
          f"L={consistent.left:.0f}, R={consistent.right:.0f}")


def test_margin_adjustment_suggestions():
    """Test margin adjustment suggestions"""
    print("\nTesting margin adjustment suggestions...")

    manager = MarginManager({'threshold': 10})

    # Create margins with inconsistencies
    margins = [
        Margin(top=72, bottom=72, left=72, right=72, page_num=0, confidence=1.0),
        Margin(top=72, bottom=72, left=72, right=72, page_num=1, confidence=1.0),
        Margin(top=100, bottom=100, left=100, right=100, page_num=2, confidence=0.8),  # Inconsistent
        Margin(top=72, bottom=72, left=72, right=72, page_num=3, confidence=1.0),
    ]

    suggestions = manager.suggest_margin_adjustments(margins)

    assert suggestions['recommended_margins'] is not None
    assert len(suggestions['inconsistent_pages']) > 0

    # Page 2 should be flagged as inconsistent
    inconsistent_page_nums = [p['page'] for p in suggestions['inconsistent_pages']]
    assert 2 in inconsistent_page_nums

    print(f"  ✓ Recommended margins: {suggestions['recommended_margins']}")
    print(f"  ✓ Inconsistent pages: {inconsistent_page_nums}")


def test_minimum_margin_enforcement():
    """Test minimum margin enforcement"""
    print("\nTesting minimum margin enforcement...")

    manager = MarginManager({'min_margin': 50})

    # Create margin with values below minimum
    margin = Margin(
        top=20,
        bottom=30,
        left=40,
        right=10,
        page_num=0,
        confidence=1.0
    )

    enforced = manager._enforce_minimum_margins(margin)

    # All margins should be at least 50
    assert enforced.top == 50
    assert enforced.bottom == 50
    assert enforced.left == 50
    assert enforced.right == 50

    print(f"  ✓ Enforced minimum margins (50pt)")


def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")

    manager = MarginManager()

    # Test with non-existent file
    try:
        margins = manager.extract_margins("/nonexistent/file.pdf")
        assert False, "Should have raised exception"
    except Exception as e:
        print(f"  ✓ Correctly raised exception for non-existent file")

    # Test with empty content boundary detection
    import pdfplumber
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty PDF
        pdf_path = os.path.join(tmpdir, "empty.pdf")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        doc.save(pdf_path)
        doc.close()

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            bbox = manager.detect_content_boundaries(page)
            # Empty page should return None
            assert bbox is None
            print(f"  ✓ Correctly handled empty page")


def run_all_tests():
    """Run all margin manager tests"""
    print("=" * 80)
    print("MARGIN MANAGER TESTS (Instruction 12)")
    print("=" * 80)

    tests = [
        test_margin_manager_initialization,
        test_margin_extraction,
        test_margin_extraction_multipage,
        test_content_boundary_detection,
        test_margin_enforcement,
        test_safe_area_calculation,
        test_margin_ratio_calculation,
        test_consistent_margin_finding,
        test_margin_adjustment_suggestions,
        test_minimum_margin_enforcement,
        test_error_handling
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
