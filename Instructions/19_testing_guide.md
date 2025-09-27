# PDF Translation Pipeline - Comprehensive Testing Guide

## Overview
Complete testing strategy for ensuring 98%+ accuracy and layout preservation.

## 1. Unit Testing

### Test Structure
```
tests/
├── unit/
│   ├── test_margin_manager.py
│   ├── test_layout_manager.py
│   ├── test_text_controller.py
│   ├── test_edge_handler.py
│   ├── test_font_extractor.py
│   ├── test_formula_extractor.py
│   ├── test_table_extractor.py
│   ├── test_vla_trigger.py
│   ├── test_gemini_client.py
│   ├── test_xliff_generator.py
│   └── test_pdf_reconstructor.py
├── integration/
│   ├── test_extraction_pipeline.py
│   ├── test_translation_flow.py
│   ├── test_reconstruction_flow.py
│   └── test_end_to_end.py
├── performance/
│   ├── test_load.py
│   ├── test_memory.py
│   └── test_concurrency.py
└── fixtures/
    ├── sample_pdfs/
    ├── expected_outputs/
    └── test_data.py
```

## 2. Unit Tests Implementation

### Test Margin Manager

```python
import pytest
from margin_manager import MarginManager, Margin
import numpy as np

class TestMarginManager:
    """Test margin detection and enforcement"""
    
    @pytest.fixture
    def manager(self):
        return MarginManager({
            'threshold': 10,
            'enforce_strict': True
        })
    
    def test_extract_margins_simple(self, manager):
        """Test margin extraction from simple PDF"""
        margins = manager.extract_margins('tests/fixtures/simple.pdf')
        
        assert len(margins) > 0
        assert all(isinstance(m, Margin) for m in margins)
        assert all(m.top >= 0 for m in margins)
        assert all(m.confidence > 0 for m in margins)
    
    def test_detect_content_boundaries(self, manager):
        """Test content boundary detection"""
        # Mock page data
        page_mock = create_mock_page([
            {'x0': 50, 'y0': 50, 'x1': 550, 'y1': 750}
        ])
        
        bbox = manager.detect_content_boundaries(page_mock)
        
        assert bbox == (50, 50, 550, 750)
    
    def test_enforce_margins(self, manager):
        """Test margin enforcement"""
        content = {
            'elements': [
                {'id': '1', 'bbox': {'x': 20, 'y': 100}},  # Violates left margin
                {'id': '2', 'bbox': {'x': 100, 'y': 20}}   # Violates top margin
            ],
            'page_width': 612,
            'page_height': 792
        }
        
        margin = Margin(top=50, bottom=50, left=50, right=50, page_num=0)
        
        adjusted = manager.enforce_margins(content, margin)
        
        assert adjusted['elements'][0]['bbox']['x'] == 50
        assert adjusted['elements'][1]['bbox']['y'] == 50
        assert 'margin_violations' in adjusted
    
    def test_find_consistent_margins(self, manager):
        """Test finding consistent margins across pages"""
        margins = [
            Margin(top=72, bottom=72, left=72, right=72, page_num=0),
            Margin(top=72, bottom=72, left=72, right=72, page_num=1),
            Margin(top=70, bottom=72, left=72, right=72, page_num=2),  # Slight variation
        ]
        
        consistent = manager._find_consistent_margins(margins)
        
        assert consistent.top == 72  # Should find mode
        assert consistent.left == 72
```

### Test VLA Trigger

```python
import pytest
from vla_trigger import VLATrigger, ComplexityLevel, VLADecision
import numpy as np

class TestVLATrigger:
    """Test VLA triggering logic"""
    
    @pytest.fixture
    def trigger(self):
        return VLATrigger({
            'simple_threshold': 0.3,
            'moderate_threshold': 0.5,
            'complex_threshold': 0.7
        })
    
    def test_calculate_complexity_factors(self, trigger):
        """Test complexity calculation"""
        # Create test image
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        
        # Mock extraction result
        extraction = {
            'text_blocks': [
                {'confidence': 0.95, 'text': 'Test'},
                {'confidence': 0.85, 'text': 'Text'}
            ],
            'images': [],
            'tables': []
        }
        
        factors = trigger.calculate_complexity_factors(image, extraction)
        
        assert 'layout_complexity' in factors
        assert 'mixed_content' in factors
        assert 'ocr_confidence' in factors
        assert all(0 <= v <= 1 for v in factors.values())
    
    def test_make_vla_decision_simple(self, trigger):
        """Test decision for simple document"""
        factors = {
            'layout_complexity': 0.1,
            'mixed_content': 0.0,
            'ocr_confidence': 0.1,
            'visual_elements': 0.1,
            'text_structure': 0.1,
            'quality_issues': 0.0
        }
        
        decision = trigger.make_vla_decision(factors)
        
        assert decision.use_vla == False
        assert decision.complexity_level == ComplexityLevel.SIMPLE
        assert decision.recommended_model == 'paddleocr'
    
    def test_make_vla_decision_complex(self, trigger):
        """Test decision for complex document"""
        factors = {
            'layout_complexity': 0.8,
            'mixed_content': 0.7,
            'ocr_confidence': 0.6,
            'visual_elements': 0.7,
            'text_structure': 0.5,
            'quality_issues': 0.3
        }
        
        decision = trigger.make_vla_decision(factors)
        
        assert decision.use_vla == True
        assert decision.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME]
        assert decision.recommended_model in ['surya', 'mplug-docowl', 'internvl-2.0']
```

### Test Text Length Controller

```python
import pytest
from text_length_controller import TextLengthController, TextMeasurement

class TestTextLengthController:
    """Test text length control"""
    
    @pytest.fixture
    def controller(self):
        return TextLengthController({
            'max_expansion_ratio': 1.1,
            'min_compression_ratio': 0.7
        })
    
    def test_measure_text(self, controller):
        """Test text measurement"""
        font_info = {
            'name': 'Arial',
            'size': 12
        }
        
        measurement = controller.measure_text("Hello World", font_info)
        
        assert isinstance(measurement, TextMeasurement)
        assert measurement.width > 0
        assert measurement.height > 0
        assert measurement.char_count == 11
    
    def test_fit_translation_no_overflow(self, controller):
        """Test fitting when no overflow"""
        translation = "Short text"
        bbox = {'width': 200, 'height': 20}
        font_info = {'size': 12}
        
        result = controller.fit_translation(translation, bbox, font_info)
        
        assert result['success'] == True
        assert result['method'] == 'none'
        assert result['fitted_text'] == translation
    
    def test_fit_translation_with_overflow(self, controller):
        """Test fitting with overflow"""
        translation = "This is a very long text that definitely won't fit in a small box"
        bbox = {'width': 50, 'height': 20}
        font_info = {'size': 12}
        
        result = controller.fit_translation(translation, bbox, font_info)
        
        assert len(result['fitted_text']) <= len(translation)
        assert result['method'] in ['abbreviation', 'spacing', 'font_size', 'truncation']
    
    def test_apply_abbreviation(self, controller):
        """Test abbreviation strategy"""
        text = "International Corporation Department Management"
        
        abbreviated = controller._apply_abbreviation(text, {}, {}, 1.2)
        
        assert "Int'l" in abbreviated
        assert "Corp." in abbreviated
        assert "Dept." in abbreviated
        assert len(abbreviated) < len(text)
```

## 3. Integration Tests

### Test End-to-End Pipeline

```python
import pytest
import asyncio
from integrated_pipeline import IntegratedPDFTranslationPipeline

class TestIntegratedPipeline:
    """Test complete pipeline integration"""
    
    @pytest.fixture
    async def pipeline(self):
        return IntegratedPDFTranslationPipeline('tests/config/test_config.yaml')
    
    @pytest.mark.asyncio
    async def test_simple_document_translation(self, pipeline):
        """Test translation of simple document"""
        result = await pipeline.process_pdf(
            pdf_path='tests/fixtures/simple.pdf',
            target_lang='zh',
            output_path='tests/output/simple_translated.pdf',
            source_lang='en'
        )
        
        assert result['success'] == True
        assert result['total_pages'] > 0
        assert os.path.exists('tests/output/simple_translated.pdf')
    
    @pytest.mark.asyncio
    async def test_complex_layout_preservation(self, pipeline):
        """Test preservation of complex layout"""
        result = await pipeline.process_pdf(
            pdf_path='tests/fixtures/complex_layout.pdf',
            target_lang='es',
            output_path='tests/output/complex_translated.pdf'
        )
        
        # Verify layout metrics
        assert result['success'] == True
        
        # Check layout preservation
        layout_score = verify_layout_preservation(
            'tests/fixtures/complex_layout.pdf',
            'tests/output/complex_translated.pdf'
        )
        
        assert layout_score > 0.95  # 95% layout preservation
    
    @pytest.mark.asyncio
    async def test_formula_preservation(self, pipeline):
        """Test formula preservation"""
        result = await pipeline.process_pdf(
            pdf_path='tests/fixtures/scientific.pdf',
            target_lang='fr',
            output_path='tests/output/scientific_translated.pdf'
        )
        
        # Verify formulas are preserved
        formulas_original = extract_formulas('tests/fixtures/scientific.pdf')
        formulas_translated = extract_formulas('tests/output/scientific_translated.pdf')
        
        assert len(formulas_original) == len(formulas_translated)
        
        # Check LaTeX preservation
        for orig, trans in zip(formulas_original, formulas_translated):
            assert orig['latex'] == trans['latex']
```

## 4. Performance Tests

### Load Testing

```python
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.performance
    async def test_concurrent_translations(self):
        """Test concurrent translation handling"""
        pipeline = IntegratedPDFTranslationPipeline()
        
        # Create 10 concurrent translation tasks
        tasks = []
        for i in range(10):
            task = pipeline.process_pdf(
                pdf_path=f'tests/fixtures/sample_{i % 3}.pdf',
                target_lang='zh',
                output_path=f'tests/output/concurrent_{i}.pdf'
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should succeed
        assert all(r['success'] for r in results)
        
        # Check performance
        total_time = end_time - start_time
        avg_time = total_time / len(tasks)
        
        assert avg_time < 60  # Less than 1 minute average
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large document
        pipeline = IntegratedPDFTranslationPipeline()
        asyncio.run(pipeline.process_pdf(
            pdf_path='tests/fixtures/large_document.pdf',
            target_lang='zh',
            output_path='tests/output/large_translated.pdf'
        ))
        
        # Check memory after
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - baseline_memory
        
        assert memory_increase < 1000  # Less than 1GB increase
```

## 5. Test Data Preparation

### Create Test Fixtures

```python
# tests/fixtures/test_data.py

def create_test_pdfs():
    """Create various test PDFs"""
    
    test_cases = [
        {
            'name': 'simple.pdf',
            'pages': 1,
            'content': 'Simple text content',
            'layout': 'single-column'
        },
        {
            'name': 'complex_layout.pdf',
            'pages': 5,
            'content': 'Multi-column with images',
            'layout': 'multi-column',
            'images': 3,
            'tables': 2
        },
        {
            'name': 'scientific.pdf',
            'pages': 10,
            'content': 'Scientific paper with formulas',
            'formulas': 15,
            'citations': 20
        },
        {
            'name': 'edge_cases.pdf',
            'pages': 3,
            'features': [
                'rotated_text',
                'vertical_text',
                'watermarks',
                'footnotes',
                'form_fields'
            ]
        }
    ]
    
    for test_case in test_cases:
        create_pdf_from_spec(test_case)

def create_mock_page(elements):
    """Create mock page for testing"""
    class MockPage:
        def __init__(self, elements):
            self.chars = elements
            self.width = 612
            self.height = 792
            
        def find_tables(self):
            return []
            
        @property
        def images(self):
            return []
    
    return MockPage(elements)
```

## 6. Quality Metrics Testing

### Translation Quality

```python
from sacrebleu import BLEU
from bert_score import score as bert_score

def test_translation_quality():
    """Test translation quality metrics"""
    
    # Reference translations
    references = load_reference_translations()
    
    # Generate translations
    translations = []
    for doc in test_documents:
        result = translate_document(doc)
        translations.append(result)
    
    # Calculate BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(translations, [references])
    
    assert bleu_score.score > 85  # 85+ BLEU score
    
    # Calculate BERT score
    P, R, F1 = bert_score(translations, references, lang='en')
    
    assert F1.mean() > 0.90  # 90%+ BERT score

def test_layout_preservation():
    """Test layout preservation accuracy"""
    
    for test_file in test_files:
        original = extract_layout(test_file)
        translated = extract_layout(f"{test_file}_translated")
        
        # Calculate IoU for each element
        iou_scores = []
        for orig_elem, trans_elem in zip(original, translated):
            iou = calculate_iou(orig_elem.bbox, trans_elem.bbox)
            iou_scores.append(iou)
        
        avg_iou = np.mean(iou_scores)
        assert avg_iou > 0.95  # 95%+ layout preservation
```

## 7. Test Execution

### Run All Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/ -m performance

# Run with parallel execution
pytest tests/ -n 4

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/unit/test_margin_manager.py::TestMarginManager::test_extract_margins_simple
```

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 8. Test Coverage Requirements

### Minimum Coverage Targets
- Unit Tests: 90%
- Integration Tests: 80%
- End-to-End Tests: 70%
- Overall Coverage: 85%

### Critical Path Coverage
- Margin detection: 100%
- Layout analysis: 95%
- Text measurement: 100%
- Translation: 90%
- Reconstruction: 95%

## 9. Test Documentation

### Test Case Template
```yaml
test_id: TC001
name: Test margin extraction
category: unit
component: MarginManager
priority: high
preconditions:
  - Valid PDF file available
  - MarginManager initialized
steps:
  1. Load test PDF
  2. Call extract_margins()
  3. Verify returned margins
expected_results:
  - Margins detected correctly
  - Confidence score > 0.8
  - All margins >= minimum threshold
```

## 10. Performance Benchmarks

### Expected Performance
- Simple document (1-10 pages): < 30 seconds
- Complex document (10-50 pages): < 3 minutes
- Large document (50-200 pages): < 10 minutes
- Memory usage: < 2GB for most documents
- CPU usage: < 80% average
- Concurrent jobs: 10+ simultaneous

### Success Criteria
- Translation accuracy: >98%
- Layout preservation: >95%
- Formula preservation: 100%
- Table structure: >98%
- Font preservation: >95%
- Processing speed: <1 minute/page
- Error rate: <2%
- Crash rate: <0.1%