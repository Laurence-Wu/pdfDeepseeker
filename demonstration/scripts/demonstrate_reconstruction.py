#!/usr/bin/env python3
"""
Complete PDF Translation and Reconstruction Demonstration
Shows the full workflow: PDF → Extraction → Translation → XLIFF → Reconstruction

This demonstrates Instructions 00-11 working together.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.extractors.font_extractor import FontExtractor
from src.core.extractors.formula_extractor import FormulaExtractor
from src.core.extractors.table_extractor import TableExtractor
from src.core.extractors.watermark_extractor import WatermarkExtractor
from src.core.extractors.citation_extractor import CitationExtractor
from src.core.extractors.content_classifier import ContentClassifier
from src.core.translation.gemini_client import GeminiClient, TranslationRequest
from src.core.translation.translation_strategies import MultiStrategyTranslator
from src.core.translation.translation_tracker import TranslationTracker, TranslationIssue, IssueType, IssueSeverity
from src.core.xliff.xliff_generator import XLIFFGenerator
from src.core.reconstruction.pdf_reconstructor import PDFReconstructor
from src.utils.config_loader import load_translation_config
import fitz


class ReconstructionDemonstration:
    """Complete reconstruction pipeline demonstration"""

    def __init__(self, pdf_path, target_lang="zh", output_dir="demonstration/output_reconstruction"):
        self.pdf_path = pdf_path
        self.target_lang = target_lang
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / "reconstruction_log.txt"
        self.log_buffer = []

        # Load API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key or self.api_key.startswith('your_'):
            raise ValueError("GEMINI_API_KEY not configured in .env")

        # Load configuration
        self.config = load_translation_config()

        # Translation tracking
        self.tracker = TranslationTracker()

        # Data storage
        self.extraction_data = {}
        self.translations = []
        self.xliff_content = None
        self.parsed_xliff = None

    def log(self, message, console=True):
        """Log message to file and optionally console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.log_buffer.append(log_msg)

        if console:
            print(message)

    async def run_demonstration(self):
        """Run the complete demonstration"""

        self.log("\n" + "=" * 80)
        self.log("PDF TRANSLATION & RECONSTRUCTION DEMONSTRATION")
        self.log("Instructions 00-11 Complete Workflow")
        self.log("=" * 80)
        self.log(f"\nInput PDF: {self.pdf_path}")
        self.log(f"Target Language: {self.target_lang}")
        self.log(f"Output Directory: {self.output_dir}")
        self.log("")

        start_time = datetime.now()

        try:
            # Step 1: Extract Content
            await self.step1_extract_content()

            # Step 2: Translate Content
            await self.step2_translate_content()

            # Step 3: Generate XLIFF
            await self.step3_generate_xliff()

            # Step 4: Parse XLIFF (simulate round-trip)
            await self.step4_parse_xliff()

            # Step 5: Reconstruct PDF
            await self.step5_reconstruct_pdf()

            # Step 6: Validate Results
            await self.step6_validate_results()

            # Step 7: Generate Translation Quality Report
            self.log("")
            self.log("=" * 80)
            self.log("STEP 7: TRANSLATION QUALITY REPORT")
            self.log("=" * 80)
            self.log("")

            # Print summary to console and log
            summary = self.tracker.get_summary()
            self.log(f"Total Translations: {summary['total_translations']}")
            self.log(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
            self.log(f"Failed: {summary['failed']}")
            self.log(f"Low Confidence: {summary['low_confidence']}")
            self.log(f"Total Issues: {summary['total_issues']}")

            if summary['issues_by_severity']:
                self.log(f"\nIssues by Severity:")
                for severity, count in summary['issues_by_severity'].items():
                    self.log(f"  {severity.upper()}: {count}")

            if summary['issues_by_type']:
                self.log(f"\nIssues by Type:")
                for issue_type, count in summary['issues_by_type'].items():
                    self.log(f"  {issue_type}: {count}")

            # Save detailed report
            report_path = self.output_dir / "translation_quality_report.json"
            self.tracker.save_report(report_path)
            self.log(f"\n  ✓ Detailed report saved: {report_path}")

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            self.log(f"\n{'=' * 80}")
            self.log(f"DEMONSTRATION COMPLETED SUCCESSFULLY")
            self.log(f"Total Time: {duration:.2f}s")
            self.log(f"{'=' * 80}\n")

            # Save log
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_buffer))

            return True

        except Exception as e:
            self.log(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def step1_extract_content(self):
        """Step 1: Extract all content from PDF"""

        self.log("=" * 80)
        self.log("STEP 1: CONTENT EXTRACTION")
        self.log("=" * 80)
        self.log("")

        # Extract fonts
        self.log("Extracting fonts...")
        font_extractor = FontExtractor()
        fonts = font_extractor.extract_all_fonts(self.pdf_path)
        self.extraction_data['fonts'] = fonts
        self.log(f"  ✓ Fonts extracted: {len(fonts.get('embedded_fonts', {}))}")

        # Extract formulas
        self.log("Extracting formulas...")
        formula_extractor = FormulaExtractor()
        formulas = formula_extractor.extract_formulas(self.pdf_path)
        self.extraction_data['formulas'] = formulas
        self.log(f"  ✓ Formulas extracted: {len(formulas)}")

        # Extract tables
        self.log("Extracting tables...")
        table_extractor = TableExtractor()
        tables = table_extractor.extract_tables(self.pdf_path)
        self.extraction_data['tables'] = tables
        self.log(f"  ✓ Tables extracted: {len(tables)}")

        # Extract watermarks
        self.log("Extracting watermarks...")
        watermark_extractor = WatermarkExtractor()
        watermarks = watermark_extractor.extract_watermarks(self.pdf_path)
        self.extraction_data['watermarks'] = watermarks
        self.log(f"  ✓ Watermarks extracted: {len(watermarks)}")

        # Extract citations
        self.log("Extracting citations...")
        citation_extractor = CitationExtractor()
        citations = citation_extractor.extract_citations(self.pdf_path)
        self.extraction_data['citations'] = citations
        self.log(f"  ✓ Citations extracted: {len(citations)}")

        # Initialize content classifier
        self.content_classifier = ContentClassifier()

        self.log("")

    async def step2_translate_content(self):
        """Step 2: Translate text content"""

        self.log("=" * 80)
        self.log("STEP 2: TRANSLATION")
        self.log("=" * 80)
        self.log("")

        config = {
            'use_openrouter': False,
            'model': 'gemini-2.0-flash-exp',
            'temperature': 0.3,
            'max_tokens': 500,
            'use_advanced_prompts': True
        }

        doc = fitz.open(self.pdf_path)

        async with GeminiClient(api_key=self.api_key, config=config) as client:
            for page_num, page in enumerate(doc):
                self.log(f"Translating page {page_num + 1}...")

                # Extract text blocks with font information
                page_dict = page.get_text("dict")
                blocks_simple = page.get_text("blocks")  # For backward compatibility

                # Create a mapping of block positions to font info
                block_fonts = {}
                for block in page_dict.get('blocks', []):
                    if block['type'] == 0:  # Text block
                        bbox_key = tuple(block['bbox'])
                        # Get font info from first span
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                block_fonts[bbox_key] = {
                                    'font': span.get('font', 'Helvetica'),
                                    'size': span.get('size', 12),
                                    'color': span.get('color', 0)  # RGB as int
                                }
                                break
                            if bbox_key in block_fonts:
                                break

                # Get config values
                min_text_len = self.config['text_processing']['min_text_length']
                trans_threshold = self.config['text_processing']['translation_threshold']
                rate_delay = self.config['rate_limiting']['delay_between_requests']
                max_retries = self.config['rate_limiting']['max_retries']
                retry_waits = self.config['rate_limiting']['retry_wait_times']
                doc_type = self.config['translation']['default_document_type']

                for block_num, block in enumerate(blocks_simple):
                    if len(block) >= 5:  # Has text
                        text = block[4].strip()

                        if len(text) > min_text_len:  # Process text longer than minimum
                            # Get font info for this block
                            bbox_key = tuple(block[:4])
                            font_info = block_fonts.get(bbox_key, {'font': 'helv', 'size': 12, 'color': 0})

                            # Convert color int to hex
                            color_int = font_info['color']
                            color_hex = f"#{color_int:06x}"

                            # Translate substantial text, keep short text as-is
                            if len(text) > trans_threshold:
                                location = f"page {page_num + 1}, block {block_num + 1}"

                                # Classify content to determine if it should be translated
                                classification = self.content_classifier.classify_text(text, {
                                    'page': page_num,
                                    'block': block_num,
                                    'location': location
                                })

                                if not classification['should_translate']:
                                    # Preserve this content as-is
                                    translated_text = text
                                    confidence = 1.0
                                    self.log(f"  ✓ Block {block_num + 1}: preserved ({classification['preserve_reason']})")

                                    # Record as preserved, not as translation issue
                                    self.tracker.record_issue(TranslationIssue(
                                        issue_type=IssueType.NO_TRANSLATION,
                                        severity=IssueSeverity.INFO,
                                        source_text=text,
                                        translated_text=text,
                                        confidence=classification['confidence'],
                                        location=location,
                                        attempted_strategies=["content_classifier"],
                                        error_message=f"Preserved: {classification['preserve_reason']}"
                                    ))
                                else:
                                    # Proceed with translation
                                    request = TranslationRequest(
                                        text=text,
                                        source_lang="en",
                                        target_lang=self.target_lang,
                                        document_type=doc_type
                                    )

                                    # Add delay to avoid rate limits
                                    await asyncio.sleep(rate_delay)

                                    # Try translation with retry on rate limit
                                    for retry in range(max_retries):
                                        try:
                                            response = await client.translate(request)
                                            translated_text = response.translated_text
                                            confidence = response.confidence

                                            # Track successful translation
                                            if translated_text != text:
                                                self.tracker.record_success(text, translated_text, confidence, location)
                                            else:
                                                # Text unchanged - record as issue
                                                self.tracker.record_issue(TranslationIssue(
                                                    issue_type=IssueType.NO_TRANSLATION,
                                                    severity=IssueSeverity.INFO,
                                                    source_text=text,
                                                    translated_text=translated_text,
                                                    confidence=confidence,
                                                    location=location,
                                                    attempted_strategies=["direct"],
                                                    error_message="Text unchanged after translation"
                                                ))
                                            break
                                        except Exception as e:
                                            error_str = str(e)
                                            if '429' in error_str and retry < max_retries - 1:
                                                # Rate limited, wait longer and retry
                                                wait_time = retry_waits[retry] if retry < len(retry_waits) else retry_waits[-1]
                                                self.log(f"    Rate limit hit for block {block_num + 1}, waiting {wait_time}s before retry {retry + 1}/{max_retries}")
                                                await asyncio.sleep(wait_time)
                                                continue
                                            else:
                                                # Failed, use original text as fallback
                                                if '429' in error_str:
                                                    self.log(f"    Block {block_num + 1} translation failed after {max_retries} retries (rate limit)")
                                                else:
                                                    self.log(f"    Block {block_num + 1} translation error: {error_str[:150]}")
                                                translated_text = text
                                                confidence = 0.0
                                                # Track failure
                                                self.tracker.record_failure(text, location, error_str, ["direct"])
                                                break
                            else:
                                # Short text (formulas, labels) - keep as-is
                                translated_text = text
                                confidence = 1.0  # Mark as "preserved"

                            self.translations.append({
                                'page': page_num,
                                'block': block_num,
                                'original': text,
                                'translated': translated_text,
                                'confidence': confidence,
                                'bbox': list(block[:4]),
                                'font': font_info['font'],
                                'size': font_info['size'],
                                'color': color_hex,
                                'unit_id': f"p{page_num}_b{block_num}"
                            })

                            if len(text) > 20:
                                self.log(f"  ✓ Block {block_num + 1}: {confidence:.2f} confidence")
                            else:
                                self.log(f"  ✓ Block {block_num + 1}: preserved as-is")

        doc.close()
        self.log(f"\n  ✓ Total translations: {len(self.translations)}")
        self.log("")

        # Translate table contents
        await self.step2b_translate_tables(client)

    async def step2b_translate_tables(self, client):
        """Step 2b: Translate table cell contents"""

        tables = self.extraction_data.get('tables', [])
        if not tables:
            return

        self.log("Translating tables...")

        # Load config values
        min_cell_length = self.config['table_translation']['min_cell_length']
        rate_delay = self.config['rate_limiting']['delay_between_requests']
        max_retries = self.config['rate_limiting']['max_retries']
        retry_waits = self.config['rate_limiting']['retry_wait_times']
        doc_type = self.config['translation']['default_document_type']

        for table in tables:
            if not table.get('translatable', False):
                continue

            table_cells = table.get('cells', [])
            translated_cells = []

            # Group cells by row for batch translation
            rows = {}
            for cell in table_cells:
                row_idx = cell.get('row', 0)
                if row_idx not in rows:
                    rows[row_idx] = []
                rows[row_idx].append(cell)

            # Translate each row as a batch
            for row_idx in sorted(rows.keys()):
                row_cells = rows[row_idx]

                # Separate translatable and non-translatable cells
                translatable_cells = []
                translatable_texts = []

                for cell in row_cells:
                    cell_text = cell.get('text', '')
                    if cell.get('translatable', False) and len(cell_text) > min_cell_length:
                        translatable_cells.append(cell)
                        translatable_texts.append(cell_text)
                    else:
                        # Add non-translatable cells directly
                        translated_cells.append({
                            **cell,
                            'translated_text': cell_text
                        })
                        self.log(f"      Skipping cell: \"{cell_text}\" (translatable={cell.get('translatable')}, len={len(cell_text)})")

                # If no translatable cells in this row, continue
                if not translatable_cells:
                    continue

                # Create batch translation request with full context
                is_header = translatable_cells[0].get('is_header', False)
                row_type = "header row" if is_header else f"data row {row_idx}"

                self.log(f"      Translating {row_type} with {len(translatable_cells)} cells...")

                # Use multi-strategy translator for table cells
                multi_translator = MultiStrategyTranslator(client, self.config)

                # Format: "Column1: Value1 | Column2: Value2 | Column3: Value3"
                batch_text = " | ".join(translatable_texts)

                location = f"table {table['index']}, {row_type}"
                context = {
                    "document_type": "technical",
                    "location": location,
                    "type": "table_row",
                    "row_index": row_idx,
                    "is_header": is_header
                }

                await asyncio.sleep(rate_delay)

                try:
                    # Try batch translation first
                    # For table cells, try all strategies since they often fail with direct translation
                    result, attempted = await multi_translator.translate_with_fallback(
                        text=batch_text,
                        source_lang="en",
                        target_lang=self.target_lang,
                        context=context,
                        delay_between_attempts=rate_delay,
                        try_all_strategies=True  # Try all strategies for table cells
                    )

                    translated_batch = result.translated_text

                    # Parse the batch translation
                    if result.success and translated_batch:
                        # Split by separator
                        translated_parts = [part.strip() for part in translated_batch.split('|')]

                        # Match parts to cells
                        for i, cell in enumerate(translatable_cells):
                            cell_text = cell.get('text', '')

                            if i < len(translated_parts):
                                translated_text = translated_parts[i]
                            else:
                                translated_text = cell_text

                            translated_cells.append({
                                **cell,
                                'translated_text': translated_text
                            })

                            # Track translation
                            cell_location = f"{location}, cell {i}"
                            if translated_text != cell_text:
                                self.log(f"        ✓ \"{cell_text}\" -> \"{translated_text}\"")
                                self.tracker.record_success(cell_text, translated_text, result.confidence, cell_location)
                            else:
                                # Cell unchanged - record issue
                                self.tracker.record_issue(TranslationIssue(
                                    issue_type=IssueType.TECHNICAL_TERM,
                                    severity=IssueSeverity.WARNING,
                                    source_text=cell_text,
                                    translated_text=translated_text,
                                    confidence=result.confidence,
                                    location=cell_location,
                                    attempted_strategies=attempted,
                                    error_message="Table cell unchanged after translation"
                                ))
                    else:
                        # Fallback: record failures
                        for i, cell in enumerate(translatable_cells):
                            cell_text = cell.get('text', '')
                            translated_cells.append({
                                **cell,
                                'translated_text': cell_text
                            })

                            cell_location = f"{location}, cell {i}"
                            self.tracker.record_failure(
                                cell_text, cell_location,
                                result.error_message or "Batch translation failed",
                                attempted
                            )

                except Exception as e:
                    self.log(f"    Warning: Row translation failed: {e}")
                    # Fallback: record failures
                    for i, cell in enumerate(translatable_cells):
                        cell_text = cell.get('text', '')
                        translated_cells.append({
                            **cell,
                            'translated_text': cell_text
                        })

                        cell_location = f"{location}, cell {i}"
                        self.tracker.record_failure(
                            cell_text, cell_location,
                            str(e), ["exception"]
                        )

            # Update table with translations
            table['translated_cells'] = translated_cells

            self.log(f"  ✓ Table {table['index']}: {len(translated_cells)} cells processed")

        self.log("")

    async def step3_generate_xliff(self):
        """Step 3: Generate XLIFF from translations"""

        self.log("=" * 80)
        self.log("STEP 3: XLIFF GENERATION")
        self.log("=" * 80)
        self.log("")

        xliff_generator = XLIFFGenerator()

        # Prepare translation units
        translation_units = []
        for trans in self.translations:
            # Convert bbox from [x0, y0, x1, y1] to position dict
            bbox = trans['bbox']
            translation_units.append({
                'source': trans['original'],
                'target': trans['translated'],
                'id': trans['unit_id'],
                'metadata': {
                    'confidence': trans['confidence'],
                    'page': trans['page'],
                    'bbox': bbox,
                    'position': {
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    },
                    'style': {
                        'font': trans.get('font', 'helv'),
                        'size': trans.get('size', 12),
                        'color': trans.get('color', '#000000')
                    }
                }
            })

        # Serialize extraction data (remove bytes for JSON compatibility)
        import base64

        def serialize_data(obj):
            """Convert bytes to base64 for JSON serialization"""
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode('utf-8')
            elif isinstance(obj, dict):
                return {k: serialize_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_data(item) for item in obj]
            return obj

        # Create content dict for XLIFF
        content = {
            'source_file': str(self.pdf_path),
            'translation_units': translation_units,
            'fonts': serialize_data(self.extraction_data.get('fonts', {})),
            'formulas': serialize_data(self.extraction_data.get('formulas', [])),
            'tables': serialize_data(self.extraction_data.get('tables', [])),
            'watermarks': serialize_data(self.extraction_data.get('watermarks', []))
        }

        # Generate XLIFF
        self.xliff_content = xliff_generator.create_xliff(
            content=content,
            source_lang="en",
            target_lang=self.target_lang
        )

        # Save XLIFF
        xliff_path = self.output_dir / "translation.xliff"
        with open(xliff_path, 'w', encoding='utf-8') as f:
            f.write(self.xliff_content)

        self.log(f"  ✓ XLIFF file generated: {xliff_path}")
        self.log(f"  ✓ Translation units: {len(translation_units)}")
        self.log(f"  ✓ XLIFF size: {len(self.xliff_content)} bytes")
        self.log("")

    async def step4_parse_xliff(self):
        """Step 4: Parse XLIFF (simulate round-trip)"""

        self.log("=" * 80)
        self.log("STEP 4: XLIFF PARSING (Round-trip Test)")
        self.log("=" * 80)
        self.log("")

        xliff_generator = XLIFFGenerator()

        # Parse XLIFF
        self.parsed_xliff = xliff_generator.parse_xliff(self.xliff_content)

        self.log(f"  ✓ XLIFF parsed successfully")
        self.log(f"  ✓ Source language: {self.parsed_xliff['source_lang']}")
        self.log(f"  ✓ Target language: {self.parsed_xliff['target_lang']}")
        self.log(f"  ✓ Files: {len(self.parsed_xliff['files'])}")

        if self.parsed_xliff['files']:
            units_count = len(self.parsed_xliff['files'][0]['units'])
            self.log(f"  ✓ Translation units: {units_count}")

            # Check skeleton data
            skeleton = self.parsed_xliff['files'][0].get('skeleton')
            if skeleton:
                self.log(f"  ✓ Skeleton data preserved")
                if 'fonts' in skeleton:
                    self.log(f"    - Fonts: {len(skeleton.get('fonts', {}).get('embedded_fonts', {}))}")
                if 'formulas' in skeleton:
                    self.log(f"    - Formulas: {len(skeleton.get('formulas', []))}")
                if 'tables' in skeleton:
                    self.log(f"    - Tables: {len(skeleton.get('tables', []))}")

        self.log("")

    async def step5_reconstruct_pdf(self):
        """Step 5: Reconstruct PDF with translations"""

        self.log("=" * 80)
        self.log("STEP 5: PDF RECONSTRUCTION")
        self.log("=" * 80)
        self.log("")

        reconstructor = PDFReconstructor()

        output_pdf = self.output_dir / "translated_document.pdf"

        success = reconstructor.reconstruct_pdf(
            original_pdf=self.pdf_path,
            translated_content=self.parsed_xliff,
            output_path=str(output_pdf)
        )

        if success:
            self.log(f"  ✓ PDF reconstructed: {output_pdf}")

            # Get file size
            file_size = os.path.getsize(output_pdf)
            self.log(f"  ✓ Output size: {file_size} bytes ({file_size / 1024:.1f} KB)")

            # Verify with fitz
            doc = fitz.open(str(output_pdf))
            self.log(f"  ✓ Pages: {doc.page_count}")

            # Extract text from first page
            if doc.page_count > 0:
                page_text = doc[0].get_text()
                self.log(f"  ✓ First page text length: {len(page_text)} chars")

            doc.close()
        else:
            self.log("  ✗ PDF reconstruction failed")

        self.log("")

    async def step6_validate_results(self):
        """Step 6: Validate all results"""

        self.log("=" * 80)
        self.log("STEP 6: VALIDATION")
        self.log("=" * 80)
        self.log("")

        # Validate XLIFF
        from src.core.xliff.xliff_generator import XLIFFValidator
        validator = XLIFFValidator()
        is_valid, errors = validator.validate(self.xliff_content)

        if is_valid:
            self.log("  ✓ XLIFF validation: PASSED")
        else:
            self.log(f"  ✗ XLIFF validation: FAILED ({len(errors)} errors)")
            for error in errors:
                self.log(f"    - {error}")

        # Check all files exist
        xliff_file = self.output_dir / "translation.xliff"
        pdf_file = self.output_dir / "translated_document.pdf"

        if xliff_file.exists():
            self.log(f"  ✓ XLIFF file exists: {xliff_file}")
        else:
            self.log(f"  ✗ XLIFF file missing: {xliff_file}")

        if pdf_file.exists():
            self.log(f"  ✓ Reconstructed PDF exists: {pdf_file}")
        else:
            self.log(f"  ✗ Reconstructed PDF missing: {pdf_file}")

        self.log("")


async def main():
    """Main entry point"""

    # Get PDF path
    pdf_path = "demonstration/input/sample_paper.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return 1

    # Create and run demonstration
    demo = ReconstructionDemonstration(
        pdf_path=pdf_path,
        target_lang="zh",
        output_dir="demonstration/output_reconstruction"
    )

    success = await demo.run_demonstration()

    if success:
        print("\n" + "=" * 80)
        print("COMPLETE WORKFLOW DEMONSTRATED:")
        print("=" * 80)
        print(f"  Original PDF:       {pdf_path}")
        print(f"  XLIFF output:       demonstration/output_reconstruction/translation.xliff")
        print(f"  Reconstructed PDF:  demonstration/output_reconstruction/translated_document.pdf")
        print(f"  Log file:           demonstration/output_reconstruction/reconstruction_log.txt")
        print("=" * 80)
        print("\n✅ Instructions 00-11 Complete Workflow Successfully Demonstrated!")
        print("=" * 80)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
