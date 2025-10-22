#!/usr/bin/env python3
"""
Comprehensive tests for Integrated Pipeline (Instruction 15)
Tests the complete 8-phase translation pipeline
"""

import sys
import os
from pathlib import Path
import tempfile
import fitz
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.pipeline.integrated_pipeline import IntegratedPDFTranslationPipeline


def create_simple_test_pdf(output_path: str) -> str:
    """Create a simple test PDF"""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Add title
    page.insert_text((50, 50), "Test Document", fontsize=16)

    # Add body text
    page.insert_text((50, 100), "This is a test document for pipeline testing.", fontsize=12)

    # Add more text
    page.insert_text((50, 150), "It contains multiple text blocks.", fontsize=12)

    doc.save(output_path)
    doc.close()

    return output_path


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("Testing pipeline initialization...")

    # Default config
    pipeline = IntegratedPDFTranslationPipeline()
    assert pipeline is not None
    assert pipeline.margin_manager is not None
    assert pipeline.layout_manager is not None
    assert pipeline.text_controller is not None
    assert pipeline.metrics is not None

    # Custom config
    config = {
        'margins': {'min_margin': 50},
        'layout': {'column_threshold': 0.9}
    }
    pipeline2 = IntegratedPDFTranslationPipeline(config)
    assert pipeline2.margin_manager.min_margin == 50

    print("  ✓ Pipeline initialized correctly")


def test_text_block_structuring():
    """Test text block structuring"""
    print("\nTesting text block structuring...")

    pipeline = IntegratedPDFTranslationPipeline()

    # Create mock text_dict
    text_dict = {
        'blocks': [
            {
                'type': 0,
                'lines': [
                    {
                        'spans': [
                            {
                                'text': 'Hello World',
                                'bbox': (10, 10, 100, 30),
                                'font': 'Helvetica',
                                'size': 12,
                                'color': 0,
                                'flags': 0
                            }
                        ]
                    }
                ]
            }
        ]
    }

    blocks = pipeline._structure_text_blocks(text_dict)

    assert len(blocks) == 1
    assert blocks[0]['text'] == 'Hello World'
    assert blocks[0]['bbox']['x'] == 10
    assert blocks[0]['font_size'] == 12

    print(f"  ✓ Structured {len(blocks)} text blocks")


def test_document_type_determination():
    """Test document type determination"""
    print("\nTesting document type determination...")

    pipeline = IntegratedPDFTranslationPipeline()

    # General document
    pages_layout = [
        {'elements': []},
        {'elements': []}
    ]
    doc_type = pipeline._determine_document_type(pages_layout)
    assert doc_type in ['general', 'technical', 'scientific']

    print(f"  ✓ Determined document type: {doc_type}")


def test_layout_consistency_analysis():
    """Test layout consistency analysis"""
    print("\nTesting layout consistency analysis...")

    pipeline = IntegratedPDFTranslationPipeline()

    pages_layout = [
        {'margins': {}, 'columns': []},
        {'margins': {}, 'columns': []},
        {'margins': {}, 'columns': []}
    ]

    consistency = pipeline._analyze_layout_consistency(pages_layout)

    assert 'consistent_margins' in consistency
    assert 'consistent_columns' in consistency

    print(f"  ✓ Consistency analysis: {consistency}")


def test_extraction_phase():
    """Test Phase 1: Extraction"""
    print("\nTesting Phase 1: Extraction...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test.pdf")
        create_simple_test_pdf(pdf_path)

        pipeline = IntegratedPDFTranslationPipeline()

        # Run extraction
        async def run_extraction():
            result = await pipeline._extract_all_elements(pdf_path)
            return result

        extraction = asyncio.run(run_extraction())

        assert 'source_file' in extraction
        assert 'pages' in extraction
        assert 'margins' in extraction
        assert 'fonts' in extraction
        assert len(extraction['pages']) == 1

        print(f"  ✓ Extracted {len(extraction['pages'])} pages")
        print(f"  ✓ Found {len(extraction['pages'][0]['text_blocks'])} text blocks")


def test_layout_analysis_phase():
    """Test Phase 2: Layout Analysis"""
    print("\nTesting Phase 2: Layout Analysis...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test.pdf")
        create_simple_test_pdf(pdf_path)

        pipeline = IntegratedPDFTranslationPipeline()

        # Run extraction and analysis
        async def run_analysis():
            extraction = await pipeline._extract_all_elements(pdf_path)
            analysis = pipeline._analyze_complete_layout(extraction)
            return analysis

        layout_analysis = asyncio.run(run_analysis())

        assert 'pages' in layout_analysis
        assert 'document_type' in layout_analysis
        assert 'consistency' in layout_analysis

        print(f"  ✓ Analyzed {len(layout_analysis['pages'])} pages")
        print(f"  ✓ Document type: {layout_analysis['document_type']}")


def test_edge_case_detection_phase():
    """Test Phase 3: Edge Case Detection"""
    print("\nTesting Phase 3: Edge Case Detection...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test.pdf")
        create_simple_test_pdf(pdf_path)

        pipeline = IntegratedPDFTranslationPipeline()

        # Run extraction and edge case detection
        async def run_detection():
            extraction = await pipeline._extract_all_elements(pdf_path)
            edge_cases = pipeline._detect_all_edge_cases(extraction)
            return edge_cases

        edge_cases = asyncio.run(run_detection())

        assert isinstance(edge_cases, list)

        print(f"  ✓ Detected {len(edge_cases)} edge cases")


def test_xliff_generation_phase():
    """Test Phase 5: XLIFF Generation"""
    print("\nTesting Phase 5: XLIFF Generation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test.pdf")
        create_simple_test_pdf(pdf_path)

        pipeline = IntegratedPDFTranslationPipeline()

        # Run extraction, analysis, and XLIFF generation
        async def run_xliff_gen():
            extraction = await pipeline._extract_all_elements(pdf_path)
            analysis = pipeline._analyze_complete_layout(extraction)
            edge_cases = pipeline._detect_all_edge_cases(extraction)
            xliff = pipeline._generate_xliff_with_constraints(
                extraction, analysis, edge_cases, 'en', 'zh'
            )
            return xliff

        xliff_document = asyncio.run(run_xliff_gen())

        assert xliff_document is not None
        assert isinstance(xliff_document, str)
        assert '<xliff' in xliff_document

        print(f"  ✓ Generated XLIFF document ({len(xliff_document)} bytes)")


def test_layout_validation_phase():
    """Test Phase 7: Layout Validation"""
    print("\nTesting Phase 7: Layout Validation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test PDF
        pdf_path = os.path.join(tmpdir, "test.pdf")
        create_simple_test_pdf(pdf_path)

        pipeline = IntegratedPDFTranslationPipeline()

        # Run full pipeline up to validation
        async def run_validation():
            extraction = await pipeline._extract_all_elements(pdf_path)
            analysis = pipeline._analyze_complete_layout(extraction)
            edge_cases = pipeline._detect_all_edge_cases(extraction)
            xliff = pipeline._generate_xliff_with_constraints(
                extraction, analysis, edge_cases, 'en', 'zh'
            )

            # For testing, use the XLIFF as-is (no actual translation)
            validated = pipeline._validate_layout_preservation(xliff)
            return validated

        validation_result = asyncio.run(run_validation())

        assert 'valid' in validation_result
        assert 'issues' in validation_result
        assert 'parsed_xliff' in validation_result

        print(f"  ✓ Validation: {validation_result['valid']}")
        print(f"  ✓ Issues found: {len(validation_result['issues'])}")


def test_metrics_tracking():
    """Test metrics tracking"""
    print("\nTesting metrics tracking...")

    pipeline = IntegratedPDFTranslationPipeline()

    assert 'total_pages' in pipeline.metrics
    assert 'processing_time' in pipeline.metrics
    assert 'extraction_time' in pipeline.metrics
    assert 'translation_time' in pipeline.metrics
    assert 'reconstruction_time' in pipeline.metrics
    assert 'errors' in pipeline.metrics

    print(f"  ✓ Metrics initialized: {list(pipeline.metrics.keys())}")


def test_component_integration():
    """Test that all components are properly integrated"""
    print("\nTesting component integration...")

    pipeline = IntegratedPDFTranslationPipeline()

    # Check all components exist
    assert hasattr(pipeline, 'margin_manager')
    assert hasattr(pipeline, 'layout_manager')
    assert hasattr(pipeline, 'text_controller')
    assert hasattr(pipeline, 'font_extractor')
    assert hasattr(pipeline, 'formula_extractor')
    assert hasattr(pipeline, 'table_extractor')
    assert hasattr(pipeline, 'watermark_extractor')
    assert hasattr(pipeline, 'gemini_client')
    assert hasattr(pipeline, 'xliff_generator')
    assert hasattr(pipeline, 'xliff_validator')
    assert hasattr(pipeline, 'pdf_reconstructor')

    print(f"  ✓ All 11 components integrated")


def run_all_tests():
    """Run all pipeline tests"""
    print("=" * 80)
    print("INTEGRATED PIPELINE TESTS (Instruction 15)")
    print("=" * 80)

    tests = [
        test_pipeline_initialization,
        test_text_block_structuring,
        test_document_type_determination,
        test_layout_consistency_analysis,
        test_extraction_phase,
        test_layout_analysis_phase,
        test_edge_case_detection_phase,
        test_xliff_generation_phase,
        test_layout_validation_phase,
        test_metrics_tracking,
        test_component_integration
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
