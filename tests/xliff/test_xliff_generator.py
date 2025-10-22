#!/usr/bin/env python3
"""
Comprehensive tests for XLIFF Generator (Instruction 10)
Tests XLIFF 2.1 generation and parsing functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.xliff.xliff_generator import XLIFFGenerator, XLIFFValidator, XLIFFUnit
from lxml import etree


def test_xliff_generator_initialization():
    """Test XLIFFGenerator initialization"""
    print("Testing XLIFF Generator initialization...")

    generator = XLIFFGenerator()
    assert generator is not None
    assert generator.XLIFF_NS == "urn:oasis:names:tc:xliff:document:2.1"
    assert generator.PDF_NS == "urn:custom:pdf:metadata:1.0"
    assert 'xliff' in generator.nsmap

    print("  ✓ XLIFFGenerator initialized correctly")


def test_create_simple_xliff():
    """Test creating simple XLIFF document"""
    print("\nTesting simple XLIFF creation...")

    generator = XLIFFGenerator()

    content = {
        'source_file': 'test.pdf',
        'translation_units': [
            {
                'id': 'unit_1',
                'source': 'Hello World',
                'target': '你好世界',
                'metadata': {
                    'confidence': 0.95,
                    'page': 1
                }
            }
        ]
    }

    xliff = generator.create_xliff(
        content=content,
        source_lang='en',
        target_lang='zh'
    )

    assert xliff is not None
    assert isinstance(xliff, str)
    assert 'xliff' in xliff
    assert 'version="2.1"' in xliff
    assert 'srcLang="en"' in xliff
    assert 'trgLang="zh"' in xliff
    assert 'Hello World' in xliff
    assert '你好世界' in xliff

    print("  ✓ Simple XLIFF document created successfully")
    print(f"  ✓ XLIFF length: {len(xliff)} bytes")


def test_create_xliff_with_metadata():
    """Test creating XLIFF with rich metadata"""
    print("\nTesting XLIFF creation with metadata...")

    generator = XLIFFGenerator()

    content = {
        'source_file': 'document.pdf',
        'translation_units': [
            {
                'id': 'p0_u1',
                'source': 'Scientific Research',
                'target': '科学研究',
                'metadata': {
                    'confidence': 0.92,
                    'page': 0,
                    'bbox': [100, 50, 200, 70]
                }
            },
            {
                'id': 'p0_u2',
                'source': 'Abstract',
                'target': '摘要',
                'metadata': {
                    'confidence': 0.98,
                    'page': 0,
                    'bbox': [50, 100, 150, 120]
                }
            }
        ],
        'fonts': {
            'embedded_fonts': {
                '1': {'name': 'Arial', 'type': 'TrueType'}
            }
        },
        'formulas': [],
        'tables': []
    }

    xliff = generator.create_xliff(
        content=content,
        source_lang='en',
        target_lang='zh',
        document_metadata={'datatype': 'pdf'}
    )

    assert xliff is not None
    assert 'Scientific Research' in xliff
    assert '科学研究' in xliff
    assert 'skeleton' in xliff
    assert 'fonts' in xliff

    print("  ✓ XLIFF with metadata created successfully")


def test_xliff_with_page_structure():
    """Test creating XLIFF with page structure"""
    print("\nTesting XLIFF with page structure...")

    generator = XLIFFGenerator()

    content = {
        'source_file': 'multipage.pdf',
        'pages': [
            {
                'text_blocks': [
                    {
                        'text': 'Page 1 content',
                        'bbox': {'x': 50, 'y': 50, 'width': 200, 'height': 20},
                        'style': {
                            'font': 'Arial',
                            'size': 12,
                            'color': '#000000'
                        }
                    }
                ],
                'dimensions': {'width': 612, 'height': 792}
            },
            {
                'text_blocks': [
                    {
                        'text': 'Page 2 content',
                        'bbox': {'x': 50, 'y': 50, 'width': 200, 'height': 20},
                        'style': {
                            'font': 'Arial',
                            'size': 12,
                            'color': '#000000'
                        }
                    }
                ],
                'dimensions': {'width': 612, 'height': 792}
            }
        ]
    }

    xliff = generator.create_xliff(
        content=content,
        source_lang='en',
        target_lang='fr'
    )

    assert xliff is not None
    assert 'page_1' in xliff or 'Page 1' in xliff
    assert 'page_2' in xliff or 'Page 2' in xliff
    assert 'Page 1 content' in xliff
    assert 'Page 2 content' in xliff

    print("  ✓ XLIFF with page structure created successfully")


def test_xliff_parsing():
    """Test parsing XLIFF document"""
    print("\nTesting XLIFF parsing...")

    generator = XLIFFGenerator()

    # Create XLIFF
    content = {
        'source_file': 'test.pdf',
        'translation_units': [
            {
                'id': 'unit_1',
                'source': 'Original text',
                'target': 'Translated text',
                'metadata': {'confidence': 0.90}
            },
            {
                'id': 'unit_2',
                'source': 'Another text',
                'target': '另一个文本',
                'metadata': {'confidence': 0.85}
            }
        ]
    }

    xliff_str = generator.create_xliff(content, 'en', 'zh')

    # Parse XLIFF
    parsed = generator.parse_xliff(xliff_str)

    assert parsed is not None
    assert parsed['source_lang'] == 'en'
    assert parsed['target_lang'] == 'zh'
    assert len(parsed['files']) > 0
    assert len(parsed['files'][0]['units']) == 2

    # Check first unit
    unit1 = parsed['files'][0]['units'][0]
    assert unit1['id'] == 'unit_1'
    assert unit1['source'] == 'Original text'
    assert unit1['target'] == 'Translated text'

    # Check second unit
    unit2 = parsed['files'][0]['units'][1]
    assert unit2['id'] == 'unit_2'
    assert unit2['source'] == 'Another text'
    assert unit2['target'] == '另一个文本'

    print("  ✓ XLIFF parsed correctly")
    print(f"  ✓ Extracted {len(parsed['files'][0]['units'])} translation units")


def test_xliff_with_tables():
    """Test XLIFF generation with table content"""
    print("\nTesting XLIFF with tables...")

    generator = XLIFFGenerator()

    content = {
        'source_file': 'table.pdf',
        'pages': [
            {
                'tables': [
                    {
                        'rows': [
                            [
                                {'text': 'Header 1'},
                                {'text': 'Header 2'}
                            ],
                            [
                                {'text': 'Data 1'},
                                {'text': 'Data 2'}
                            ]
                        ]
                    }
                ],
                'text_blocks': []
            }
        ]
    }

    xliff = generator.create_xliff(content, 'en', 'es')

    assert xliff is not None
    assert 'Header 1' in xliff
    assert 'Header 2' in xliff
    assert 'Data 1' in xliff
    assert 'Data 2' in xliff
    assert 'translate="no"' in xliff  # Headers should not be translated

    print("  ✓ XLIFF with tables created successfully")


def test_xliff_validator():
    """Test XLIFF validation"""
    print("\nTesting XLIFF validation...")

    generator = XLIFFGenerator()
    validator = XLIFFValidator()

    # Create valid XLIFF
    content = {
        'source_file': 'valid.pdf',
        'translation_units': [
            {
                'id': 'unit_1',
                'source': 'Test',
                'target': '测试'
            }
        ]
    }

    xliff_str = generator.create_xliff(content, 'en', 'zh')

    # Validate
    is_valid, errors = validator.validate(xliff_str)

    assert is_valid == True
    assert len(errors) == 0

    print("  ✓ XLIFF validation passed")

    # Test invalid XLIFF (missing version)
    invalid_xliff = '''<?xml version="1.0" encoding="UTF-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:2.1" srcLang="en" trgLang="zh">
</xliff>'''

    is_valid, errors = validator.validate(invalid_xliff)

    assert is_valid == False
    assert len(errors) > 0

    print(f"  ✓ Invalid XLIFF correctly rejected ({len(errors)} errors)")


def test_xliff_with_special_characters():
    """Test XLIFF with special characters and unicode"""
    print("\nTesting XLIFF with special characters...")

    generator = XLIFFGenerator()

    content = {
        'source_file': 'unicode.pdf',
        'translation_units': [
            {
                'id': 'unit_1',
                'source': 'Special chars: <>&"\'',
                'target': '特殊字符: <>&"\''
            },
            {
                'id': 'unit_2',
                'source': 'Math: ∫∑∏√∞',
                'target': '数学: ∫∑∏√∞'
            },
            {
                'id': 'unit_3',
                'source': 'Emoji: 🌍🚀💡',
                'target': '表情: 🌍🚀💡'
            }
        ]
    }

    xliff = generator.create_xliff(content, 'en', 'zh')

    assert xliff is not None
    # Special chars should be escaped in XML
    assert '&lt;' in xliff or '<![CDATA[' in xliff
    # Unicode should be preserved
    assert '∫' in xliff or '&#' in xliff
    assert '🌍' in xliff or '&#' in xliff

    # Parse back
    parsed = generator.parse_xliff(xliff)
    assert len(parsed['files'][0]['units']) == 3

    print("  ✓ Special characters handled correctly")


def test_xliff_unit_dataclass():
    """Test XLIFFUnit dataclass"""
    print("\nTesting XLIFFUnit dataclass...")

    unit = XLIFFUnit(
        id='test_unit',
        source='Source text',
        target='Target text',
        metadata={'confidence': 0.95},
        translate=True,
        preserve_space=False,
        max_length=100,
        notes=['Note 1', 'Note 2']
    )

    assert unit.id == 'test_unit'
    assert unit.source == 'Source text'
    assert unit.target == 'Target text'
    assert unit.metadata['confidence'] == 0.95
    assert unit.translate == True
    assert len(unit.notes) == 2

    print("  ✓ XLIFFUnit dataclass works correctly")


def run_all_tests():
    """Run all XLIFF generator tests"""
    print("=" * 80)
    print("XLIFF GENERATOR TESTS (Instruction 10)")
    print("=" * 80)

    tests = [
        test_xliff_generator_initialization,
        test_create_simple_xliff,
        test_create_xliff_with_metadata,
        test_xliff_with_page_structure,
        test_xliff_parsing,
        test_xliff_with_tables,
        test_xliff_validator,
        test_xliff_with_special_characters,
        test_xliff_unit_dataclass
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} error: {e}")
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
