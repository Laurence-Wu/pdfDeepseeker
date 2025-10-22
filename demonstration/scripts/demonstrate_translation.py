#!/usr/bin/env python3
"""
PDF Translation Pipeline Demonstration
Shows the complete workflow from PDF input to translated output
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
from src.core.deciders.vla_trigger import VLATrigger
from src.core.handlers.edge_case_handler import EdgeCaseHandler
from src.core.translation.gemini_client import GeminiClient, TranslationRequest
from src.core.xliff.xliff_generator import XLIFFGenerator
from src.utils.config_loader import load_translation_config
import fitz
import cv2
import numpy as np


class TranslationDemonstration:
    """Complete translation pipeline demonstration"""

    def __init__(self, pdf_path, target_lang="zh", output_dir="demonstration/output"):
        self.pdf_path = pdf_path
        self.target_lang = target_lang
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / "translation_log.txt"
        self.log_buffer = []

        # Load translation config
        self.config = load_translation_config()

        # Load API key
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key or self.api_key.startswith('your_'):
            raise ValueError("GEMINI_API_KEY not configured in .env")

        # Extracted data
        self.fonts = []
        self.formulas = []
        self.tables = []
        self.watermarks = []
        self.edge_cases = []
        self.translations = []
        self.vla_decisions = []

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
        self.log("PDF TRANSLATION PIPELINE DEMONSTRATION")
        self.log("=" * 80)
        self.log(f"\nInput PDF: {self.pdf_path}")
        self.log(f"Target Language: {self.target_lang}")
        self.log(f"Output Directory: {self.output_dir}")
        self.log("")

        # Start timing
        start_time = datetime.now()

        try:
            # Step 1: Extract Content
            await self.step1_extract_content()

            # Step 2: Analyze Complexity
            await self.step2_analyze_complexity()

            # Step 3: Detect Edge Cases
            await self.step3_detect_edge_cases()

            # Step 4: Translate Content
            await self.step4_translate_content()

            # Step 5: Generate XLIFF
            await self.step5_generate_xliff()

            # Step 6: Save Results
            await self.step6_save_results()

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
        self.fonts = font_extractor.extract_all_fonts(self.pdf_path)
        self.log(f"  ✓ Fonts extracted: {len(self.fonts)}")

        # Extract formulas
        self.log("Extracting formulas...")
        formula_extractor = FormulaExtractor()
        self.formulas = formula_extractor.extract_formulas(self.pdf_path)
        self.log(f"  ✓ Formulas extracted: {len(self.formulas)}")

        # Extract tables
        self.log("Extracting tables...")
        table_extractor = TableExtractor()
        self.tables = table_extractor.extract_tables(self.pdf_path)
        self.log(f"  ✓ Tables extracted: {len(self.tables)}")

        # Extract watermarks
        self.log("Extracting watermarks...")
        watermark_extractor = WatermarkExtractor()
        self.watermarks = watermark_extractor.extract_watermarks(self.pdf_path)
        self.log(f"  ✓ Watermarks extracted: {len(self.watermarks)}")

        self.log("")

    async def step2_analyze_complexity(self):
        """Step 2: Analyze document complexity with VLA"""

        self.log("=" * 80)
        self.log("STEP 2: VLA COMPLEXITY ANALYSIS")
        self.log("=" * 80)
        self.log("")

        vla_trigger = VLATrigger()
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(doc):
            self.log(f"Analyzing page {page_num + 1}...")

            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get page text for context
            page_dict = page.get_text("dict")

            # Analyze complexity
            decision = vla_trigger.analyze_document(img, page_dict)
            self.vla_decisions.append({
                'page': page_num + 1,
                'complexity_level': str(decision.complexity_level),
                'use_vla': decision.use_vla,
                'recommended_model': decision.recommended_model,
                'confidence': decision.confidence,
                'reasons': decision.reasons
            })

            self.log(f"  ✓ Page {page_num + 1}: {decision.complexity_level}")
            self.log(f"    Model: {decision.recommended_model}, Confidence: {decision.confidence:.2f}")

        doc.close()
        self.log("")

    async def step3_detect_edge_cases(self):
        """Step 3: Detect edge cases"""

        self.log("=" * 80)
        self.log("STEP 3: EDGE CASE DETECTION")
        self.log("=" * 80)
        self.log("")

        edge_handler = EdgeCaseHandler()
        doc = fitz.open(self.pdf_path)

        total_edge_cases = 0
        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict")
            edge_cases = edge_handler.detect_edge_cases(page_dict)

            self.edge_cases.extend([
                {
                    'page': page_num + 1,
                    'type': str(ec.type),
                    'confidence': ec.confidence,
                    'bbox': ec.bbox
                }
                for ec in edge_cases
            ])

            total_edge_cases += len(edge_cases)
            if edge_cases:
                self.log(f"Page {page_num + 1}: {len(edge_cases)} edge cases detected")

        doc.close()
        self.log(f"  ✓ Total edge cases detected: {total_edge_cases}")
        self.log("")

    async def step4_translate_content(self):
        """Step 4: Translate text content"""

        self.log("=" * 80)
        self.log("STEP 4: TRANSLATION")
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

                # Extract text blocks
                blocks = page.get_text("blocks")

                page_translations = []
                for block_num, block in enumerate(blocks):
                    if len(block) >= 5:  # Has text
                        text = block[4].strip()

                        if len(text) > 20:  # Only translate substantial text
                            # Determine document type (simple heuristic)
                            doc_type = "scientific" if any(word in text.lower() for word in
                                ['quantum', 'research', 'experiment', 'theory']) else "general"

                            request = TranslationRequest(
                                text=text,
                                source_lang="en",
                                target_lang=self.target_lang,
                                document_type=doc_type
                            )

                            # Add delay to avoid rate limits (from config)
                            rate_delay = self.config['rate_limiting']['delay_between_requests']
                            await asyncio.sleep(rate_delay)

                            response = await client.translate(request)

                            page_translations.append({
                                'block': block_num,
                                'original': text[:100] + "..." if len(text) > 100 else text,
                                'translated': response.translated_text[:100] + "..." if len(response.translated_text) > 100 else response.translated_text,
                                'confidence': response.confidence,
                                'tokens_used': response.tokens_used,
                                'bbox': block[:4]
                            })

                            self.log(f"  ✓ Block {block_num + 1}: {response.confidence:.2f} confidence")

                self.translations.append({
                    'page': page_num + 1,
                    'translations': page_translations
                })

        doc.close()
        self.log(f"\n  ✓ Total translations: {sum(len(p['translations']) for p in self.translations)}")
        self.log("")

    async def step5_generate_xliff(self):
        """Step 5: Generate XLIFF output"""

        self.log("=" * 80)
        self.log("STEP 5: XLIFF GENERATION")
        self.log("=" * 80)
        self.log("")

        xliff_generator = XLIFFGenerator()

        # Prepare translation units
        translation_units = []
        for page_data in self.translations:
            for trans in page_data['translations']:
                translation_units.append({
                    'source': trans['original'],
                    'target': trans['translated'],
                    'id': f"page_{page_data['page']}_block_{trans['block']}",
                    'metadata': {
                        'confidence': trans['confidence'],
                        'page': page_data['page'],
                        'bbox': trans['bbox']
                    }
                })

        # Generate XLIFF
        xliff_path = self.output_dir / "translation.xliff"

        # Create content dict for XLIFF
        content = {
            'source_file': str(self.pdf_path),
            'translation_units': translation_units
        }

        xliff_content = xliff_generator.create_xliff(
            content=content,
            source_lang="en",
            target_lang=self.target_lang
        )

        # Save XLIFF
        with open(xliff_path, 'w', encoding='utf-8') as f:
            f.write(xliff_content)

        self.log(f"  ✓ XLIFF file generated: {xliff_path}")
        self.log(f"  ✓ Translation units: {len(translation_units)}")
        self.log("")

    async def step6_save_results(self):
        """Step 6: Save all results"""

        self.log("=" * 80)
        self.log("STEP 6: SAVING RESULTS")
        self.log("=" * 80)
        self.log("")

        # Save extraction results (convert bytes to base64 strings)
        import base64

        def serialize_data(obj):
            """Convert bytes, numpy types to JSON-serializable formats"""
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode('utf-8')
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: serialize_data(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_data(item) for item in obj]
            return obj

        extraction_results = {
            'fonts': serialize_data(self.fonts),
            'formulas': serialize_data(self.formulas),
            'tables': serialize_data(self.tables),
            'watermarks': serialize_data(self.watermarks)
        }

        extraction_path = self.output_dir / "extraction_results.json"
        with open(extraction_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_results, f, indent=2, ensure_ascii=False)
        self.log(f"  ✓ Extraction results: {extraction_path}")

        # Save VLA analysis
        vla_path = self.output_dir / "vla_analysis.json"
        with open(vla_path, 'w', encoding='utf-8') as f:
            json.dump(serialize_data(self.vla_decisions), f, indent=2)
        self.log(f"  ✓ VLA analysis: {vla_path}")

        # Save edge cases
        edge_path = self.output_dir / "edge_cases.json"
        with open(edge_path, 'w', encoding='utf-8') as f:
            json.dump(serialize_data(self.edge_cases), f, indent=2)
        self.log(f"  ✓ Edge cases: {edge_path}")

        # Save translations
        trans_path = self.output_dir / "translations.json"
        with open(trans_path, 'w', encoding='utf-8') as f:
            json.dump(serialize_data(self.translations), f, indent=2, ensure_ascii=False)
        self.log(f"  ✓ Translations: {trans_path}")

        self.log("")


async def main():
    """Main entry point"""

    # Get PDF path
    pdf_path = "demonstration/input/sample_paper.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return 1

    # Create and run demonstration
    demo = TranslationDemonstration(
        pdf_path=pdf_path,
        target_lang="zh",  # Chinese
        output_dir="demonstration/output"
    )

    success = await demo.run_demonstration()

    if success:
        print("\n" + "=" * 80)
        print("DEMONSTRATION FILES GENERATED:")
        print("=" * 80)
        print(f"  Input PDF:          {pdf_path}")
        print(f"  Log file:           demonstration/output/translation_log.txt")
        print(f"  Extraction results: demonstration/output/extraction_results.json")
        print(f"  VLA analysis:       demonstration/output/vla_analysis.json")
        print(f"  Edge cases:         demonstration/output/edge_cases.json")
        print(f"  Translations:       demonstration/output/translations.json")
        print(f"  XLIFF output:       demonstration/output/translation.xliff")
        print("=" * 80)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
