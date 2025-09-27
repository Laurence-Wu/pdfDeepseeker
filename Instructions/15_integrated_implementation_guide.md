# Integrated Implementation Guide - Complete PDF Translation Pipeline

## Overview
Complete guide for integrating all components into a production-ready PDF translation pipeline.

## System Integration Architecture

```python
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path
import json

class IntegratedPDFTranslationPipeline:
    """
    Complete integrated pipeline with all components working together.
    Orchestrates the entire translation process from input to output.
    """
    
    def __init__(self, config_path: str = 'config/pipeline_config.yaml'):
        """
        Initialize integrated pipeline with all components.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize all managers
        self.margin_manager = MarginManager(self.config.get('margins', {}))
        self.layout_manager = LayoutManager(self.config.get('layout', {}))
        self.text_controller = TextLengthController(self.config.get('text_control', {}))
        self.edge_handler = EdgeCaseHandler(self.config.get('edge_cases', {}))
        
        # Initialize extractors
        self.font_extractor = FontExtractor(self.config.get('fonts', {}))
        self.formula_extractor = FormulaExtractor(self.config.get('formulas', {}))
        self.table_extractor = TableExtractor(self.config.get('tables', {}))
        self.watermark_extractor = WatermarkExtractor(self.config.get('watermarks', {}))
        
        # Initialize VLA components
        self.vla_trigger = VLATrigger(self.config.get('vla', {}))
        self.vla_processor = VLAProcessor(self.config.get('vla', {}))
        self.vla_pipeline = VLAProcessingPipeline(self.config.get('vla', {}))
        
        # Initialize translation components
        self.gemini_client = GeminiClient(
            api_key=self.config.get('translation', {}).get('api_key'),
            config=self.config.get('translation', {})
        )
        self.prompt_engine = PromptEngine(self.config.get('prompts', {}))
        
        # Initialize XLIFF components
        self.xliff_generator = XLIFFGenerator(self.config.get('xliff', {}))
        self.xliff_validator = XLIFFValidator()
        
        # Initialize reconstruction
        self.pdf_reconstructor = PDFReconstructor(self.config.get('reconstruction', {}))
        
        # Performance tracking
        self.metrics = {
            'total_pages': 0,
            'processing_time': 0,
            'extraction_time': 0,
            'translation_time': 0,
            'reconstruction_time': 0,
            'vla_usage': 0,
            'errors': []
        }
    
    async def process_pdf(self,
                          pdf_path: str,
                          target_lang: str,
                          output_path: str,
                          source_lang: str = 'auto') -> Dict:
        """
        Main entry point for PDF translation.
        
        Args:
            pdf_path: Input PDF path
            target_lang: Target language code
            output_path: Output PDF path
            source_lang: Source language code (auto-detect if 'auto')
            
        Returns:
            Processing result with metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Phase 1: Deep Extraction
            print("Phase 1: Extracting document content...")
            extraction_start = time.time()
            extraction_result = await self._extract_all_elements(pdf_path)
            self.metrics['extraction_time'] = time.time() - extraction_start
            
            # Phase 2: Layout Analysis
            print("Phase 2: Analyzing document layout...")
            layout_analysis = self._analyze_complete_layout(extraction_result)
            
            # Phase 3: Edge Case Detection
            print("Phase 3: Detecting edge cases...")
            edge_cases = self._detect_all_edge_cases(extraction_result)
            
            # Phase 4: VLA Processing (if needed)
            print("Phase 4: Checking VLA requirements...")
            vla_enhanced = await self._process_with_vla_if_needed(
                extraction_result, 
                pdf_path
            )
            
            # Phase 5: XLIFF Generation
            print("Phase 5: Generating XLIFF document...")
            xliff_document = self._generate_xliff_with_constraints(
                vla_enhanced or extraction_result,
                layout_analysis,
                edge_cases,
                source_lang,
                target_lang
            )
            
            # Phase 6: Smart Translation
            print("Phase 6: Translating content...")
            translation_start = time.time()
            translated_xliff = await self._smart_translate(
                xliff_document,
                target_lang
            )
            self.metrics['translation_time'] = time.time() - translation_start
            
            # Phase 7: Layout Validation
            print("Phase 7: Validating layout preservation...")
            validated = self._validate_layout_preservation(translated_xliff)
            
            # Phase 8: Reconstruction
            print("Phase 8: Reconstructing PDF...")
            reconstruction_start = time.time()
            success = await self._reconstruct_with_exact_layout(
                pdf_path,
                validated,
                output_path
            )
            self.metrics['reconstruction_time'] = time.time() - reconstruction_start
            
            # Update metrics
            self.metrics['processing_time'] = time.time() - start_time
            self.metrics['success'] = success
            
            print(f"✓ Translation complete! Total time: {self.metrics['processing_time']:.2f}s")
            
            return self.metrics
            
        except Exception as e:
            self.metrics['errors'].append(str(e))
            print(f"✗ Translation failed: {e}")
            raise e
    
    async def _extract_all_elements(self, pdf_path: str) -> Dict:
        """
        Extract all elements from PDF with complete metadata.
        """
        extraction_result = {
            'source_file': pdf_path,
            'pages': [],
            'margins': [],
            'fonts': {},
            'global_elements': {}
        }
        
        # Extract margins
        extraction_result['margins'] = self.margin_manager.extract_margins(pdf_path)
        
        # Extract fonts
        extraction_result['fonts'] = self.font_extractor.extract_all_fonts(pdf_path)
        
        # Extract formulas
        formulas = self.formula_extractor.extract_formulas(pdf_path)
        
        # Extract tables
        tables = self.table_extractor.extract_tables(pdf_path)
        
        # Extract watermarks
        watermarks = self.watermark_extractor.extract_watermarks(pdf_path)
        
        extraction_result['global_elements'] = {
            'formulas': formulas,
            'tables': tables,
            'watermarks': watermarks
        }
        
        # Extract page-by-page content
        import fitz
        doc = fitz.open(pdf_path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text with positions
            text_dict = page.get_text("dict")
            
            # Structure page content
            page_content = {
                'page_num': page_num,
                'dimensions': {
                    'width': page.rect.width,
                    'height': page.rect.height
                },
                'rotation': page.rotation,
                'text_blocks': self._structure_text_blocks(text_dict),
                'images': self._extract_page_images(page),
                'tables': [t for t in tables if t.get('page') == page_num],
                'formulas': [f for f in formulas if f.get('page') == page_num]
            }
            
            extraction_result['pages'].append(page_content)
        
        doc.close()
        self.metrics['total_pages'] = len(extraction_result['pages'])
        
        return extraction_result
    
    def _analyze_complete_layout(self, extraction_result: Dict) -> Dict:
        """
        Perform comprehensive layout analysis.
        """
        layout_analysis = {
            'pages': []
        }
        
        for page in extraction_result['pages']:
            # Analyze page layout
            page_layout = self.layout_manager.analyze_layout(page)
            
            # Add margin information
            page_num = page['page_num']
            if page_num < len(extraction_result['margins']):
                page_layout['margins'] = extraction_result['margins'][page_num]
            
            layout_analysis['pages'].append(page_layout)
        
        # Analyze document-level layout patterns
        layout_analysis['document_type'] = self._determine_document_type(
            layout_analysis['pages']
        )
        layout_analysis['consistency'] = self._analyze_layout_consistency(
            layout_analysis['pages']
        )
        
        return layout_analysis
    
    def _detect_all_edge_cases(self, extraction_result: Dict) -> List[Dict]:
        """
        Detect all edge cases across the document.
        """
        all_edge_cases = []
        
        for page in extraction_result['pages']:
            # Detect edge cases for this page
            page_edge_cases = self.edge_handler.detect_edge_cases(page)
            
            # Add page number to each edge case
            for edge_case in page_edge_cases:
                edge_case.page_num = page['page_num']
            
            all_edge_cases.extend(page_edge_cases)
        
        return all_edge_cases
    
    async def _process_with_vla_if_needed(self, 
                                         extraction_result: Dict,
                                         pdf_path: str) -> Optional[Dict]:
        """
        Process with VLA if document complexity requires it.
        """
        # Check each page for VLA requirements
        vla_needed_pages = []
        
        for page in extraction_result['pages']:
            # Create page image for analysis
            page_image = self._render_page_to_image(pdf_path, page['page_num'])
            
            # Check if VLA needed
            decision = self.vla_trigger.analyze_document(
                page_image,
                page
            )
            
            if decision.use_vla:
                vla_needed_pages.append((page['page_num'], decision))
                self.metrics['vla_usage'] += 1
        
        if not vla_needed_pages:
            return None
        
        print(f"VLA processing needed for {len(vla_needed_pages)} pages")
        
        # Process with VLA
        enhanced_result = extraction_result.copy()
        
        for page_num, decision in vla_needed_pages:
            page_image = self._render_page_to_image(pdf_path, page_num)
            
            # Process with VLA
            vla_result = await self.vla_pipeline.process_document(
                page_image,
                {'page_num': page_num, 'complexity': decision.complexity_level}
            )
            
            if vla_result.success:
                # Merge VLA results
                enhanced_result['pages'][page_num]['vla_enhanced'] = vla_result.data
        
        return enhanced_result
    
    def _generate_xliff_with_constraints(self,
                                        extraction_result: Dict,
                                        layout_analysis: Dict,
                                        edge_cases: List,
                                        source_lang: str,
                                        target_lang: str) -> str:
        """
        Generate XLIFF with all constraints and metadata.
        """
        # Prepare content for XLIFF
        xliff_content = {
            'source_file': extraction_result['source_file'],
            'pages': []
        }
        
        for page_idx, page in enumerate(extraction_result['pages']):
            page_content = {
                'text_blocks': [],
                'dimensions': page['dimensions'],
                'rotation': page.get('rotation', 0)
            }
            
            # Process text blocks with constraints
            for block in page.get('text_blocks', []):
                # Calculate length constraint
                constraint = self.text_controller.generate_length_constraint(
                    block.get('text', ''),
                    block.get('bbox', {}),
                    {'name': block.get('font'), 'size': block.get('font_size', 12)}
                )
                
                # Add constraint to block
                block['max_length'] = constraint['max_length']
                block['constraint'] = constraint
                
                page_content['text_blocks'].append(block)
            
            # Add tables and form fields
            page_content['tables'] = page.get('tables', [])
            page_content['form_fields'] = page.get('form_fields', [])
            
            xliff_content['pages'].append(page_content)
        
        # Add skeleton data
        xliff_content['skeleton'] = {
            'fonts': extraction_result.get('fonts', {}),
            'margins': extraction_result.get('margins', []),
            'layout': layout_analysis,
            'edge_cases': [self._serialize_edge_case(ec) for ec in edge_cases],
            'global_elements': extraction_result.get('global_elements', {})
        }
        
        # Generate XLIFF
        xliff_document = self.xliff_generator.create_xliff(
            xliff_content,
            source_lang,
            target_lang,
            {'document_type': layout_analysis.get('document_type', 'general')}
        )
        
        # Validate XLIFF
        is_valid, errors = self.xliff_validator.validate(xliff_document)
        if not is_valid:
            print(f"XLIFF validation errors: {errors}")
        
        return xliff_document
    
    async def _smart_translate(self, xliff_document: str, target_lang: str) -> str:
        """
        Translate XLIFF content with smart strategies.
        """
        # Parse XLIFF
        parsed_xliff = self.xliff_generator.parse_xliff(xliff_document)
        
        # Translate each unit
        for file_data in parsed_xliff.get('files', []):
            for unit in file_data.get('units', []):
                if not unit.get('translate', True):
                    continue
                
                # Generate optimized prompt
                prompt = self.prompt_engine.generate_prompt(
                    text=unit['source'],
                    source_lang=parsed_xliff['source_lang'],
                    target_lang=target_lang,
                    document_type=DocumentType.GENERAL,  # Could be more specific
                    constraints=unit.get('metadata', {}).get('constraint'),
                    metadata=unit.get('metadata', {})
                )
                
                # Translate
                request = TranslationRequest(
                    text=unit['source'],
                    source_lang=parsed_xliff['source_lang'],
                    target_lang=target_lang,
                    max_length=unit.get('metadata', {}).get('constraint', {}).get('max_length')
                )
                
                response = await self.gemini_client.translate(request)
                
                # Validate and adjust if needed
                if response.translated_text:
                    # Check length
                    validation = self.text_controller.validate_translation_fit(
                        response.translated_text,
                        unit.get('metadata', {}).get('position', {}),
                        {'size': 12}  # Default font size
                    )
                    
                    if not validation['fits']:
                        # Apply fitting strategies
                        fitting_result = self.text_controller.fit_translation(
                            response.translated_text,
                            unit.get('metadata', {}).get('position', {}),
                            {'size': 12}
                        )
                        unit['target'] = fitting_result['fitted_text']
                    else:
                        unit['target'] = response.translated_text
        
        # Regenerate XLIFF with translations
        translated_xliff = self.xliff_generator.create_xliff(
            parsed_xliff,
            parsed_xliff['source_lang'],
            target_lang
        )
        
        return translated_xliff
    
    def _validate_layout_preservation(self, translated_xliff: str) -> Dict:
        """
        Validate that translations preserve layout.
        """
        parsed = self.xliff_generator.parse_xliff(translated_xliff)
        
        validation_results = {
            'valid': True,
            'issues': [],
            'adjusted_units': []
        }
        
        for file_data in parsed.get('files', []):
            skeleton = file_data.get('skeleton', {})
            margins = skeleton.get('margins', [])
            
            for unit in file_data.get('units', []):
                if not unit.get('target'):
                    continue
                
                # Check margin compliance
                position = unit.get('metadata', {}).get('position', {})
                if position and margins:
                    # Get page margins
                    page_num = self._extract_page_num(unit['id'])
                    if page_num < len(margins):
                        page_margins = margins[page_num]
                        
                        # Check violations
                        if position['x'] < page_margins.left:
                            validation_results['issues'].append({
                                'unit_id': unit['id'],
                                'type': 'margin_violation',
                                'side': 'left'
                            })
                
                # Check text overflow
                if unit.get('metadata', {}).get('constraint'):
                    max_length = unit['metadata']['constraint'].get('max_length')
                    if max_length and len(unit['target']) > max_length:
                        validation_results['issues'].append({
                            'unit_id': unit['id'],
                            'type': 'text_overflow',
                            'excess': len(unit['target']) - max_length
                        })
        
        validation_results['valid'] = len(validation_results['issues']) == 0
        
        return validation_results
    
    async def _reconstruct_with_exact_layout(self,
                                            original_pdf: str,
                                            validated_xliff: Dict,
                                            output_path: str) -> bool:
        """
        Reconstruct PDF with exact layout preservation.
        """
        # Parse validated XLIFF
        if isinstance(validated_xliff, str):
            parsed_content = self.xliff_generator.parse_xliff(validated_xliff)
        else:
            parsed_content = validated_xliff
        
        # Reconstruct PDF
        success = self.pdf_reconstructor.reconstruct_pdf(
            original_pdf,
            parsed_content,
            output_path
        )
        
        return success
    
    # Helper methods
    
    def _load_configuration(self, config_path: str) -> Dict:
        """Load configuration from file"""
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load environment variables
        import os
        config['translation'] = {
            'api_key': os.getenv('OPENROUTER_API_KEY'),
            'base_url': os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            'model': os.getenv('OPENROUTER_MODEL', 'google/gemini-pro-1.5')
        }
        
        return config
    
    def _structure_text_blocks(self, text_dict: Dict) -> List[Dict]:
        """Structure text blocks from PyMuPDF text dict"""
        blocks = []
        
        for block in text_dict.get('blocks', []):
            if block['type'] == 0:  # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        blocks.append({
                            'text': span.get('text', ''),
                            'bbox': {
                                'x': span['bbox'][0],
                                'y': span['bbox'][1],
                                'width': span['bbox'][2] - span['bbox'][0],
                                'height': span['bbox'][3] - span['bbox'][1]
                            },
                            'font': span.get('font'),
                            'font_size': span.get('size'),
                            'flags': span.get('flags'),
                            'color': span.get('color')
                        })
        
        return blocks
    
    def _extract_page_images(self, page) -> List[Dict]:
        """Extract images from page"""
        images = []
        
        for img in page.get_images(full=True):
            img_rect = page.get_image_bbox(img[7])
            images.append({
                'bbox': {
                    'x': img_rect.x0,
                    'y': img_rect.y0,
                    'width': img_rect.width,
                    'height': img_rect.height
                },
                'xref': img[0]
            })
        
        return images
    
    def _render_page_to_image(self, pdf_path: str, page_num: int) -> np.ndarray:
        """Render PDF page to image"""
        import fitz
        import cv2
        
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render at high DPI
        mat = fitz.Matrix(2, 2)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        doc.close()
        
        return img
    
    def _determine_document_type(self, pages_layout: List[Dict]) -> str:
        """Determine document type from layout analysis"""
        
        # Count layout features
        has_formulas = any(
            page.get('special_layouts', {}).get('has_formulas')
            for page in pages_layout
        )
        has_tables = any(
            len(page.get('tables', [])) > 0
            for page in pages_layout
        )
        
        if has_formulas:
            return 'scientific'
        elif has_tables:
            return 'technical'
        else:
            return 'general'
    
    def _analyze_layout_consistency(self, pages_layout: List[Dict]) -> Dict:
        """Analyze layout consistency across pages"""
        
        return {
            'consistent_margins': all(
                page.get('margins') for page in pages_layout
            ),
            'consistent_columns': len(set(
                len(page.get('columns', [])) for page in pages_layout
            )) == 1
        }
    
    def _serialize_edge_case(self, edge_case) -> Dict:
        """Serialize edge case for storage"""
        
        return {
            'type': edge_case.type,
            'element_id': edge_case.element_id,
            'confidence': edge_case.confidence,
            'metadata': edge_case.metadata,
            'handling_strategy': edge_case.handling_strategy,
            'page_num': getattr(edge_case, 'page_num', -1)
        }
    
    def _extract_page_num(self, unit_id: str) -> int:
        """Extract page number from unit ID"""
        
        # Assuming unit ID format: p{page_num}_u{unit_num}
        if unit_id.startswith('p'):
            parts = unit_id.split('_')
            if parts:
                return int(parts[0][1:])
        
        return 0
```

## Complete Usage Example

```python
async def main():
    # Initialize pipeline
    pipeline = IntegratedPDFTranslationPipeline('config/pipeline_config.yaml')
    
    # Process PDF
    result = await pipeline.process_pdf(
        pdf_path='input/document.pdf',
        target_lang='zh',
        output_path='output/translated.pdf',
        source_lang='en'
    )
    
    # Print results
    print("\n=== Translation Complete ===")
    print(f"Total pages: {result['total_pages']}")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"- Extraction: {result['extraction_time']:.2f}s")
    print(f"- Translation: {result['translation_time']:.2f}s")
    print(f"- Reconstruction: {result['reconstruction_time']:.2f}s")
    print(f"VLA pages processed: {result['vla_usage']}")
    
    if result.get('errors'):
        print(f"Errors: {result['errors']}")

# Run
if __name__ == "__main__":
    asyncio.run(main())
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    libmagickwand-dev \
    tesseract-ocr \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENROUTER_API_KEY=${OPENROUTER_API_KEY}

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Checklist

- [ ] Configure OpenRouter API key
- [ ] Set up Redis for caching
- [ ] Configure PostgreSQL for translation memory
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up error tracking (Sentry)
- [ ] Configure rate limiting
- [ ] Set up backup strategy
- [ ] Configure auto-scaling
- [ ] Set up CI/CD pipeline
- [ ] Security audit
- [ ] Performance testing
- [ ] Load testing
- [ ] Documentation complete
- [ ] Training materials ready

## Performance Optimization

1. **Parallel Processing**: Process pages in parallel
2. **Caching**: Cache translations and extractions
3. **Batch Translation**: Send multiple units in one request
4. **Model Selection**: Use appropriate models for complexity
5. **Resource Management**: Monitor and limit resource usage

## Success Metrics

- Translation Accuracy: >98%
- Layout Preservation: >95%
- Processing Speed: <1 minute per page
- Error Rate: <2%
- User Satisfaction: >90%
