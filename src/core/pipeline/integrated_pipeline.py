#!/usr/bin/env python3
"""
Integrated PDF Translation Pipeline (Instruction 15)
Complete pipeline integrating all components (Instructions 00-14).
"""

from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path
import json
import time
import logging
import fitz
import numpy as np

from ..layout.margin_manager import MarginManager
from ..layout.layout_manager import LayoutManager
from ..text.text_length_controller import TextLengthController
from ..extractors.font_extractor import FontExtractor
from ..extractors.formula_extractor import FormulaExtractor
from ..extractors.table_extractor import TableExtractor
from ..extractors.watermark_extractor import WatermarkExtractor
from ..extractors.text_recognizer import TextRecognizer
from ..translation.gemini_client import GeminiClient, TranslationRequest
from ..xliff.xliff_generator import XLIFFGenerator, XLIFFValidator
from ..reconstruction.pdf_reconstructor import PDFReconstructor
from src.utils.config_loader import load_translation_config

logger = logging.getLogger(__name__)


class IntegratedPDFTranslationPipeline:
    """
    Complete integrated pipeline with all components working together.
    Orchestrates the entire translation process from input to output.

    8-Phase Pipeline:
    1. Deep Extraction
    2. Layout Analysis
    3. Edge Case Detection
    4. VLA Processing (if needed)
    5. XLIFF Generation
    6. Smart Translation
    7. Layout Validation
    8. Reconstruction
    """

    def __init__(self, config: Dict = None):
        """
        Initialize integrated pipeline with all components.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Load translation config for rate limiting
        self.translation_config = load_translation_config()

        # Initialize all managers
        self.margin_manager = MarginManager(self.config.get('margins', {}))
        self.layout_manager = LayoutManager(self.config.get('layout', {}))
        self.text_controller = TextLengthController(self.config.get('text_control', {}))

        # Initialize extractors
        self.font_extractor = FontExtractor(self.config.get('fonts', {}))
        self.formula_extractor = FormulaExtractor(self.config.get('formulas', {}))
        self.table_extractor = TableExtractor(self.config.get('tables', {}))
        self.watermark_extractor = WatermarkExtractor(self.config.get('watermarks', {}))

        # Initialize OCR text recognizer if enabled
        text_recognition_config = self.config.get('text_recognition', {})
        if text_recognition_config.get('enabled', False):
            try:
                self.text_recognizer = TextRecognizer(text_recognition_config)
                logger.info("OCR text recognizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize text recognizer: {e}")
                self.text_recognizer = None
        else:
            self.text_recognizer = None

        # Initialize translation components
        api_key = self.config.get('translation', {}).get('api_key')
        if not api_key:
            import os
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('OPENROUTER_API_KEY')

        self.gemini_client = GeminiClient(
            api_key=api_key,
            config=self.config.get('translation', {})
        )

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
                          source_lang: str = 'en') -> Dict:
        """
        Main entry point for PDF translation.

        Args:
            pdf_path: Input PDF path
            target_lang: Target language code
            output_path: Output PDF path
            source_lang: Source language code

        Returns:
            Processing result with metrics
        """
        start_time = time.time()

        try:
            logger.info(f"Starting PDF translation pipeline: {pdf_path}")

            # Phase 1: Deep Extraction
            logger.info("Phase 1: Extracting document content...")
            extraction_start = time.time()
            extraction_result = await self._extract_all_elements(pdf_path)
            self.metrics['extraction_time'] = time.time() - extraction_start

            # Phase 2: Layout Analysis
            logger.info("Phase 2: Analyzing document layout...")
            layout_analysis = self._analyze_complete_layout(extraction_result)

            # Phase 3: Edge Case Detection (simplified - can be expanded)
            logger.info("Phase 3: Detecting edge cases...")
            edge_cases = self._detect_all_edge_cases(extraction_result)

            # Phase 4: VLA Processing (placeholder - requires VLA implementation)
            logger.info("Phase 4: Checking VLA requirements...")
            vla_enhanced = None  # Placeholder for VLA integration

            # Phase 5: XLIFF Generation
            logger.info("Phase 5: Generating XLIFF document...")
            xliff_document = self._generate_xliff_with_constraints(
                vla_enhanced or extraction_result,
                layout_analysis,
                edge_cases,
                source_lang,
                target_lang
            )

            # Phase 6: Smart Translation
            logger.info("Phase 6: Translating content...")
            translation_start = time.time()
            translated_xliff = await self._smart_translate(
                xliff_document,
                target_lang,
                source_lang
            )
            self.metrics['translation_time'] = time.time() - translation_start

            # Phase 7: Layout Validation
            logger.info("Phase 7: Validating layout preservation...")
            validated = self._validate_layout_preservation(translated_xliff)

            # Phase 8: Reconstruction
            logger.info("Phase 8: Reconstructing PDF...")
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

            logger.info(f"✓ Translation complete! Total time: {self.metrics['processing_time']:.2f}s")

            return self.metrics

        except Exception as e:
            self.metrics['errors'].append(str(e))
            logger.error(f"✗ Translation failed: {e}", exc_info=True)
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
        try:
            extraction_result['margins'] = self.margin_manager.extract_margins(pdf_path)
        except Exception as e:
            logger.warning(f"Margin extraction failed: {e}")
            extraction_result['margins'] = []

        # Extract fonts
        try:
            extraction_result['fonts'] = self.font_extractor.extract_all_fonts(pdf_path)
        except Exception as e:
            logger.warning(f"Font extraction failed: {e}")
            extraction_result['fonts'] = {}

        # Extract formulas
        try:
            formulas = self.formula_extractor.extract_formulas(pdf_path)
        except Exception as e:
            logger.warning(f"Formula extraction failed: {e}")
            formulas = []

        # Extract tables
        try:
            tables = self.table_extractor.extract_tables(pdf_path)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            tables = []

        # Extract watermarks
        try:
            watermarks = self.watermark_extractor.extract_watermarks(pdf_path)
        except Exception as e:
            logger.warning(f"Watermark extraction failed: {e}")
            watermarks = []

        extraction_result['global_elements'] = {
            'formulas': formulas,
            'tables': tables,
            'watermarks': watermarks
        }

        # Extract page-by-page content
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Extract text with positions
            text_dict = page.get_text("dict")

            # Check if OCR is needed for this page
            use_ocr = False
            if self.text_recognizer:
                # Check if page has native text
                has_native_text = self._has_meaningful_text(text_dict)

                # Use OCR for scanned pages or if configured
                use_for_config = self.config.get('text_recognition', {}).get('use_for', {})
                if not has_native_text and use_for_config.get('scanned_pages', True):
                    use_ocr = True
                    logger.info(f"Page {page_num}: Using OCR (scanned page detected)")

            # Get text blocks
            if use_ocr:
                # Use OCR extraction
                ocr_result = self.text_recognizer.extract_text_from_page(
                    pdf_path,
                    page_num,
                    use_ocr=True
                )
                text_blocks = ocr_result['text_blocks']
                extraction_method = 'ocr'
            else:
                # Use native extraction
                text_blocks = self._structure_text_blocks(text_dict)
                extraction_method = 'native'

            # Structure page content
            page_content = {
                'page_num': page_num,
                'dimensions': {
                    'width': page.rect.width,
                    'height': page.rect.height
                },
                'rotation': page.rotation,
                'text_blocks': text_blocks,
                'extraction_method': extraction_method,
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
        Simplified implementation - can be expanded with edge case handler.
        """
        all_edge_cases = []

        for page in extraction_result['pages']:
            # Detect edge cases for this page
            # Placeholder - can integrate EdgeCaseHandler when available
            page_edge_cases = []

            # Example: Detect very small text
            for block in page.get('text_blocks', []):
                if block.get('font_size', 12) < 6:
                    page_edge_cases.append({
                        'type': 'small_text',
                        'element_id': block.get('text', '')[:20],
                        'page_num': page['page_num'],
                        'metadata': {'font_size': block.get('font_size')}
                    })

            all_edge_cases.extend(page_edge_cases)

        return all_edge_cases

    def _serialize_for_xliff(self, obj):
        """Serialize objects for XLIFF (handle bytes, numpy types)"""
        import base64
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, dict):
            return {k: self._serialize_for_xliff(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_xliff(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_xliff(obj.__dict__)
        else:
            return obj

    def _generate_xliff_with_constraints(self,
                                        extraction_result: Dict,
                                        layout_analysis: Dict,
                                        edge_cases: List,
                                        source_lang: str,
                                        target_lang: str) -> str:
        """
        Generate XLIFF with all constraints and metadata.
        """
        # Prepare translation units
        translation_units = []
        unit_id = 0

        for page_idx, page in enumerate(extraction_result['pages']):
            for block in page.get('text_blocks', []):
                text = block.get('text', '').strip()
                if not text or len(text) < 2:
                    continue

                # Calculate length constraint
                bbox = block.get('bbox', {})
                font_info = {
                    'name': block.get('font', 'Helvetica'),
                    'size': block.get('font_size', 12)
                }

                constraint = self.text_controller.generate_length_constraint(
                    text,
                    bbox,
                    font_info
                )

                # Create translation unit
                translation_units.append({
                    'id': f"p{page_idx}_u{unit_id}",
                    'source': text,
                    'target': '',  # Will be filled during translation
                    'metadata': {
                        'page': page_idx,
                        'position': bbox,
                        'style': {
                            'font': font_info['name'],
                            'size': font_info['size'],
                            'color': block.get('color', '#000000')
                        },
                        'constraint': constraint
                    }
                })
                unit_id += 1

        # Serialize data (handle bytes)
        fonts_serialized = self._serialize_for_xliff(extraction_result.get('fonts', {}))
        formulas_serialized = self._serialize_for_xliff(extraction_result.get('global_elements', {}).get('formulas', []))
        tables_serialized = self._serialize_for_xliff(extraction_result.get('global_elements', {}).get('tables', []))
        watermarks_serialized = self._serialize_for_xliff(extraction_result.get('global_elements', {}).get('watermarks', []))

        # Prepare content structure
        xliff_content = {
            'source_file': extraction_result['source_file'],
            'translation_units': translation_units,
            'fonts': fonts_serialized,
            'formulas': formulas_serialized,
            'tables': tables_serialized,
            'watermarks': watermarks_serialized
        }

        # Generate XLIFF
        xliff_document = self.xliff_generator.create_xliff(
            xliff_content,
            source_lang,
            target_lang
        )

        # Validate XLIFF
        is_valid, errors = self.xliff_validator.validate(xliff_document)
        if not is_valid:
            logger.warning(f"XLIFF validation errors: {errors}")

        return xliff_document

    async def _smart_translate(self, xliff_document: str, target_lang: str, source_lang: str) -> str:
        """
        Translate XLIFF content with smart strategies.
        """
        # Parse XLIFF
        parsed_xliff = self.xliff_generator.parse_xliff(xliff_document)

        # Translate each unit
        async with self.gemini_client:
            for file_data in parsed_xliff.get('files', []):
                for unit in file_data.get('units', []):
                    if not unit.get('source'):
                        continue

                    # Create translation request
                    metadata = unit.get('metadata', {})
                    constraint = metadata.get('constraint', {})

                    doc_type = self.translation_config['translation']['default_document_type']
                    request = TranslationRequest(
                        text=unit['source'],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_length=constraint.get('max_length'),
                        document_type=doc_type
                    )

                    # Add delay to avoid rate limits (from config)
                    rate_delay = self.translation_config['rate_limiting']['delay_between_requests']
                    await asyncio.sleep(rate_delay)

                    # Translate
                    try:
                        response = await self.gemini_client.translate(request)

                        if response.translated_text:
                            # Validate and adjust if needed
                            position = metadata.get('position', {})
                            if position:
                                validation = self.text_controller.validate_translation_fit(
                                    response.translated_text,
                                    position,
                                    metadata.get('style', {'size': 12})
                                )

                                if not validation['fits']:
                                    # Apply fitting strategies
                                    fitting_result = self.text_controller.fit_translation(
                                        response.translated_text,
                                        position,
                                        metadata.get('style', {'size': 12})
                                    )
                                    unit['target'] = fitting_result['fitted_text']
                                else:
                                    unit['target'] = response.translated_text
                            else:
                                unit['target'] = response.translated_text
                        else:
                            unit['target'] = unit['source']  # Fallback to source

                    except Exception as e:
                        logger.warning(f"Translation failed for unit {unit['id']}: {e}")
                        unit['target'] = unit['source']  # Fallback to source

        # Regenerate XLIFF with translations
        translated_content = {
            'source_file': parsed_xliff['files'][0]['original'],
            'translation_units': [],
            'fonts': parsed_xliff['files'][0]['skeleton'].get('fonts', {}),
            'formulas': parsed_xliff['files'][0]['skeleton'].get('formulas', []),
            'tables': parsed_xliff['files'][0]['skeleton'].get('tables', []),
            'watermarks': parsed_xliff['files'][0]['skeleton'].get('watermarks', [])
        }

        for unit in parsed_xliff['files'][0]['units']:
            translated_content['translation_units'].append(unit)

        translated_xliff = self.xliff_generator.create_xliff(
            translated_content,
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
            'parsed_xliff': parsed
        }

        for file_data in parsed.get('files', []):
            for unit in file_data.get('units', []):
                if not unit.get('target'):
                    continue

                # Check text overflow
                metadata = unit.get('metadata', {})
                constraint = metadata.get('constraint', {})
                if constraint:
                    max_length = constraint.get('max_length')
                    if max_length and len(unit['target']) > max_length * 1.5:  # Allow 50% overflow
                        validation_results['issues'].append({
                            'unit_id': unit['id'],
                            'type': 'text_overflow',
                            'excess': len(unit['target']) - max_length
                        })

        validation_results['valid'] = len(validation_results['issues']) == 0

        return validation_results

    async def _reconstruct_with_exact_layout(self,
                                            original_pdf: str,
                                            validated: Dict,
                                            output_path: str) -> bool:
        """
        Reconstruct PDF with exact layout preservation.
        """
        # Get parsed content
        parsed_content = validated.get('parsed_xliff')

        if not parsed_content:
            logger.error("No parsed content found for reconstruction")
            return False

        # Reconstruct PDF
        success = self.pdf_reconstructor.reconstruct_pdf(
            original_pdf,
            parsed_content,
            output_path
        )

        return success

    # Helper methods

    def _structure_text_blocks(self, text_dict: Dict) -> List[Dict]:
        """Structure text blocks from PyMuPDF text dict"""
        blocks = []

        for block in text_dict.get('blocks', []):
            if block['type'] == 0:  # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        if text:
                            blocks.append({
                                'text': text,
                                'bbox': {
                                    'x': span['bbox'][0],
                                    'y': span['bbox'][1],
                                    'width': span['bbox'][2] - span['bbox'][0],
                                    'height': span['bbox'][3] - span['bbox'][1]
                                },
                                'font': span.get('font'),
                                'font_size': span.get('size'),
                                'flags': span.get('flags'),
                                'color': f"#{span.get('color', 0):06x}"
                            })

        return blocks

    def _extract_page_images(self, page) -> List[Dict]:
        """Extract images from page"""
        images = []

        try:
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
        except Exception as e:
            logger.warning(f"Image extraction error: {e}")

        return images

    def _determine_document_type(self, pages_layout: List[Dict]) -> str:
        """Determine document type from layout analysis"""

        # Count layout features
        has_formulas = any(
            len(page.get('elements', [])) > 0 and
            any(e.type == 'formula' for e in page.get('elements', []))
            for page in pages_layout
        )
        has_tables = any(
            len([e for e in page.get('elements', []) if hasattr(e, 'type') and e.type == 'table']) > 0
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
            )) <= 2  # Allow for some variation
        }

    def _has_meaningful_text(self, text_dict: Dict) -> bool:
        """Check if page has meaningful text content"""
        if not text_dict or 'blocks' not in text_dict:
            return False

        text_blocks = [
            block for block in text_dict['blocks']
            if block.get('type') == 0  # Text block
        ]

        if not text_blocks:
            return False

        # Check total text length
        total_text = 0
        for block in text_blocks:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    total_text += len(span.get('text', ''))

        # Consider meaningful if > 10 characters
        return total_text > 10
