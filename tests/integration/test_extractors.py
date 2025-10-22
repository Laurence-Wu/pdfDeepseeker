#!/usr/bin/env python3
"""
Test script for Extract Format Classes (Instruction 08)
Tests FontExtractor, FormulaExtractor, TableExtractor, and WatermarkExtractor
"""

import sys
from pathlib import Path
import tempfile
import fitz  # PyMuPDF

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.extractors.font_extractor import FontExtractor
from src.core.extractors.formula_extractor import FormulaExtractor
from src.core.extractors.table_extractor import TableExtractor
from src.core.extractors.watermark_extractor import WatermarkExtractor


def create_test_pdf() -> str:
    """Create a simple test PDF with text and tables"""

    # Create a temporary PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size

    # Add text with different fonts
    page.insert_text((72, 100), "Test Document Title", fontsize=24, fontname="helv")
    page.insert_text((72, 150), "This is regular text with Arial font.", fontsize=12, fontname="helv")
    page.insert_text((72, 180), "This is text with Times New Roman.", fontsize=12, fontname="tiro")

    # Add some math-like text
    page.insert_text((72, 220), "Formula example: E = mc²", fontsize=12, fontname="helv")
    page.insert_text((72, 250), "Equation: x² + y² = r²", fontsize=12, fontname="helv")

    # Add watermark-like text
    page.insert_text((200, 400), "CONFIDENTIAL", fontsize=48, fontname="helv")

    doc.save(temp_path)
    doc.close()

    return temp_path


def test_font_extractor():
    """Test FontExtractor"""

    print("=== FontExtractor Tests ===\n")

    # Create test PDF
    test_pdf = create_test_pdf()

    try:
        extractor = FontExtractor(config={'cache_fonts': True})
        fonts = extractor.extract_all_fonts(test_pdf)

        print(f"✓ FontExtractor initialized")
        print(f"  Embedded fonts found: {len(fonts['embedded_fonts'])}")
        print(f"  Font usage entries: {len(fonts['font_usage'])}")
        print(f"  Font mapping entries: {len(fonts['font_mapping'])}")
        print(f"  Fallback chain length: {len(fonts['fallback_chain'])}")

        # Check structure
        assert 'embedded_fonts' in fonts
        assert 'font_usage' in fonts
        assert 'font_mapping' in fonts
        assert 'fallback_chain' in fonts

        print(f"✓ All expected keys present")

        # Test font metrics
        if len(fonts['embedded_fonts']) > 0:
            first_font_ref = list(fonts['embedded_fonts'].keys())[0]
            font_info = fonts['embedded_fonts'][first_font_ref]

            assert 'name' in font_info
            assert 'type' in font_info
            assert 'metrics' in font_info

            print(f"✓ Font metadata structure correct")
            print(f"  Sample font: {font_info['name']}")
            print(f"  Font type: {font_info['type']}")

        # Test text measurement
        if extractor.font_cache:
            width = extractor.measure_text("Test", list(extractor.font_cache.keys())[0], 12)
            print(f"✓ Text measurement working: width={width:.2f}")

        print("\n✅ FontExtractor tests passed\n")
        return True

    except Exception as e:
        print(f"\n❌ FontExtractor tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(test_pdf).unlink(missing_ok=True)


def test_formula_extractor():
    """Test FormulaExtractor"""

    print("=== FormulaExtractor Tests ===\n")

    # Create test PDF
    test_pdf = create_test_pdf()

    try:
        extractor = FormulaExtractor(config={
            'confidence_threshold': 0.5,
            'preserve_as_image': False
        })

        print(f"✓ FormulaExtractor initialized")

        # Test formula detection (may not find any in simple PDF)
        formulas = extractor.extract_formulas(test_pdf)

        print(f"  Formulas found: {len(formulas)}")

        # Test text checking
        assert extractor.is_formula_text("E = mc²") == True
        assert extractor.is_formula_text("x + y = 10") == True
        assert extractor.is_formula_text("Hello World") == False

        print(f"✓ Formula text detection working")

        # Check formula structure if any found
        for formula in formulas[:3]:
            assert 'page' in formula
            assert 'bbox' in formula
            assert 'latex' in formula
            assert 'confidence' in formula
            assert 'type' in formula

            print(f"  Formula type: {formula['type']}, confidence: {formula['confidence']:.2f}")

        print("\n✅ FormulaExtractor tests passed\n")
        return True

    except Exception as e:
        print(f"\n❌ FormulaExtractor tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(test_pdf).unlink(missing_ok=True)


def test_table_extractor():
    """Test TableExtractor"""

    print("=== TableExtractor Tests ===\n")

    # Create test PDF with table
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    test_pdf = temp_file.name
    temp_file.close()

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Draw a simple table manually
    y_start = 100
    x_start = 72
    col_width = 150
    row_height = 30

    # Table header
    page.insert_text((x_start, y_start), "Name", fontsize=12, fontname="helv")
    page.insert_text((x_start + col_width, y_start), "Age", fontsize=12, fontname="helv")
    page.insert_text((x_start + col_width*2, y_start), "City", fontsize=12, fontname="helv")

    # Table rows
    page.insert_text((x_start, y_start + row_height), "Alice", fontsize=11, fontname="helv")
    page.insert_text((x_start + col_width, y_start + row_height), "30", fontsize=11, fontname="helv")
    page.insert_text((x_start + col_width*2, y_start + row_height), "NYC", fontsize=11, fontname="helv")

    doc.save(test_pdf)
    doc.close()

    try:
        extractor = TableExtractor(config={'confidence_threshold': 0.5})

        print(f"✓ TableExtractor initialized")

        # Extract tables
        tables = extractor.extract_tables(test_pdf)

        print(f"  Tables found: {len(tables)}")

        # Check table structure
        for idx, table in enumerate(tables[:3]):
            assert 'page' in table
            assert 'bbox' in table
            assert 'rows' in table
            assert 'columns' in table
            assert 'headers' in table
            assert 'data' in table
            assert 'style' in table

            print(f"  Table {idx}: {table['columns']} columns, {len(table['rows'])} rows")

            # Test markdown conversion
            if table['rows']:
                markdown = extractor.table_to_markdown(table)
                assert '|' in markdown
                print(f"✓ Markdown conversion working")

                # Test HTML conversion
                html = extractor.table_to_html(table)
                assert '<table>' in html
                print(f"✓ HTML conversion working")

        print("\n✅ TableExtractor tests passed\n")
        return True

    except Exception as e:
        print(f"\n❌ TableExtractor tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(test_pdf).unlink(missing_ok=True)


def test_watermark_extractor():
    """Test WatermarkExtractor"""

    print("=== WatermarkExtractor Tests ===\n")

    # Create test PDF
    test_pdf = create_test_pdf()

    try:
        extractor = WatermarkExtractor(config={
            'detect_visible': True,
            'detect_invisible': False  # Skip invisible for basic test
        })

        print(f"✓ WatermarkExtractor initialized")

        # Extract watermarks
        watermarks = extractor.extract_watermarks(test_pdf)

        print(f"  Watermarks found: {len(watermarks)}")

        # Check watermark structure
        for wm in watermarks[:3]:
            assert 'page' in wm
            assert 'type' in wm

            if wm['type'] == 'visible_text':
                assert 'text' in wm
                assert 'bbox' in wm
                print(f"  Visible text watermark: '{wm['text'][:30]}...'")

            print(f"  Detection reason: {wm.get('detection_reason', 'N/A')}")

        # Test has_watermark
        has_wm = extractor.has_watermark(test_pdf)
        print(f"✓ Has watermark check: {has_wm}")

        print("\n✅ WatermarkExtractor tests passed\n")
        return True

    except Exception as e:
        print(f"\n❌ WatermarkExtractor tests failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        Path(test_pdf).unlink(missing_ok=True)


def main():
    """Run all extractor tests"""

    print("=" * 70)
    print("EXTRACT FORMAT CLASSES - INTEGRATION TESTS (Instruction 08)")
    print("=" * 70)
    print()

    results = {
        'FontExtractor': test_font_extractor(),
        'FormulaExtractor': test_formula_extractor(),
        'TableExtractor': test_table_extractor(),
        'WatermarkExtractor': test_watermark_extractor()
    }

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component}: {status}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
