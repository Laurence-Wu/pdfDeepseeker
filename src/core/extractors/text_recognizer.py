#!/usr/bin/env python3
"""
Text Recognizer using OCR
Extracts text from PDF pages using OCR when native text extraction fails.
"""

import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np

from ..adapters.ocr import PaddleOCRAdapter

logger = logging.getLogger(__name__)


class TextRecognizer:
    """
    Advanced text recognition using OCR for PDF pages.

    Use cases:
    - Scanned PDFs without embedded text
    - Images embedded in PDFs
    - Poor quality or corrupted text
    - Verification of extracted text
    - Multi-language document support
    """

    def __init__(self, config: Dict = None):
        """
        Initialize text recognizer.

        Args:
            config: Configuration dictionary with options:
                - ocr_backend: OCR backend to use ('paddleocr', 'tesseract')
                - fallback_to_native: Try native extraction first (default: True)
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - dpi: DPI for PDF rendering (default: 300)
                - ocr_config: Configuration passed to OCR backend
        """
        self.config = config or {}

        self.ocr_backend = self.config.get('ocr_backend', 'paddleocr')
        self.fallback_to_native = self.config.get('fallback_to_native', True)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.dpi = self.config.get('dpi', 300)

        # Initialize OCR backend
        ocr_config = self.config.get('ocr_config', {})

        if self.ocr_backend == 'paddleocr':
            try:
                self.ocr_adapter = PaddleOCRAdapter(ocr_config)
                logger.info("TextRecognizer initialized with PaddleOCR")
            except ImportError as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
        else:
            raise ValueError(f"Unsupported OCR backend: {self.ocr_backend}")

    def extract_text_from_page(
        self,
        pdf_path: str,
        page_num: int,
        use_ocr: bool = False
    ) -> Dict:
        """
        Extract text from a PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            use_ocr: Force OCR even if native text exists

        Returns:
            Dictionary containing:
                - text_blocks: List of text blocks with positions
                - method: Extraction method used ('native' or 'ocr')
                - confidence: Average confidence (for OCR)
                - page_num: Page number
        """
        try:
            doc = fitz.open(pdf_path)

            if page_num >= doc.page_count:
                raise ValueError(f"Page {page_num} out of range (total: {doc.page_count})")

            page = doc[page_num]

            # Try native extraction first if enabled
            if self.fallback_to_native and not use_ocr:
                native_text = page.get_text("dict")

                # Check if page has meaningful text
                if self._has_meaningful_text(native_text):
                    logger.info(f"Page {page_num}: Using native text extraction")
                    result = self._parse_native_text(native_text, page_num)
                    doc.close()
                    return result

            # Use OCR
            logger.info(f"Page {page_num}: Using OCR text extraction")

            # Render page to image
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array for OCR
            import cv2
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            doc.close()

            # Run OCR
            ocr_results = self.ocr_adapter.recognize_text(img)

            # Filter by confidence
            filtered_results = [
                r for r in ocr_results
                if r['confidence'] >= self.min_confidence
            ]

            if len(filtered_results) < len(ocr_results):
                logger.info(
                    f"Filtered {len(ocr_results) - len(filtered_results)} "
                    f"low-confidence results (< {self.min_confidence})"
                )

            # Calculate average confidence
            avg_confidence = (
                sum(r['confidence'] for r in filtered_results) / len(filtered_results)
                if filtered_results else 0.0
            )

            # Convert OCR results to text blocks
            text_blocks = self._convert_ocr_to_blocks(filtered_results, zoom)

            return {
                'text_blocks': text_blocks,
                'method': 'ocr',
                'confidence': avg_confidence,
                'page_num': page_num,
                'total_blocks': len(text_blocks)
            }

        except Exception as e:
            logger.error(f"Text extraction failed for page {page_num}: {e}")
            raise

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        use_ocr: bool = False
    ) -> List[Dict]:
        """
        Extract text from multiple PDF pages.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to process (None = all pages)
            use_ocr: Force OCR for all pages

        Returns:
            List of extraction results for each page
        """
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        if pages is None:
            pages = list(range(total_pages))

        results = []
        for page_num in pages:
            try:
                result = self.extract_text_from_page(pdf_path, page_num, use_ocr)
                results.append(result)
                logger.info(f"Processed page {page_num + 1}/{total_pages}")
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                results.append({
                    'text_blocks': [],
                    'method': 'failed',
                    'confidence': 0.0,
                    'page_num': page_num,
                    'error': str(e)
                })

        return results

    def extract_region(
        self,
        pdf_path: str,
        page_num: int,
        bbox: tuple
    ) -> Dict:
        """
        Extract text from a specific region of a page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            bbox: Bounding box (x0, y0, x1, y1)

        Returns:
            Dictionary with extracted text and confidence
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            # Render region to image
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)

            # Create clip rect
            clip_rect = fitz.Rect(bbox)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)

            # Convert to numpy array
            import cv2
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            doc.close()

            # Run OCR on region
            ocr_results = self.ocr_adapter.recognize_text(img)

            # Combine results
            combined_text = ' '.join(r['text'] for r in ocr_results)
            avg_confidence = (
                sum(r['confidence'] for r in ocr_results) / len(ocr_results)
                if ocr_results else 0.0
            )

            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'blocks': len(ocr_results),
                'bbox': bbox
            }

        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            raise

    def verify_native_extraction(
        self,
        pdf_path: str,
        page_num: int,
        threshold: float = 0.8
    ) -> Dict:
        """
        Verify native text extraction using OCR.

        Compares native extraction with OCR to detect issues.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            threshold: Similarity threshold (0-1)

        Returns:
            Dictionary with verification results
        """
        # Get native extraction
        native_result = self.extract_text_from_page(pdf_path, page_num, use_ocr=False)

        # Get OCR extraction
        ocr_result = self.extract_text_from_page(pdf_path, page_num, use_ocr=True)

        # Compare text
        native_text = ' '.join(b['text'] for b in native_result['text_blocks'])
        ocr_text = ' '.join(b['text'] for b in ocr_result['text_blocks'])

        # Calculate similarity (simple approach)
        similarity = self._calculate_text_similarity(native_text, ocr_text)

        return {
            'page_num': page_num,
            'similarity': similarity,
            'matches': similarity >= threshold,
            'native_blocks': len(native_result['text_blocks']),
            'ocr_blocks': len(ocr_result['text_blocks']),
            'native_method': native_result['method'],
            'ocr_confidence': ocr_result['confidence']
        }

    # Helper methods

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

    def _parse_native_text(self, text_dict: Dict, page_num: int) -> Dict:
        """Parse native PyMuPDF text dict to standard format"""
        text_blocks = []

        for block in text_dict.get('blocks', []):
            if block.get('type') == 0:  # Text block
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        if text:
                            bbox = span['bbox']
                            text_blocks.append({
                                'text': text,
                                'confidence': 1.0,  # Native extraction
                                'bbox': {
                                    'x': bbox[0],
                                    'y': bbox[1],
                                    'width': bbox[2] - bbox[0],
                                    'height': bbox[3] - bbox[1]
                                },
                                'font': span.get('font'),
                                'font_size': span.get('size'),
                                'color': f"#{span.get('color', 0):06x}"
                            })

        return {
            'text_blocks': text_blocks,
            'method': 'native',
            'confidence': 1.0,
            'page_num': page_num,
            'total_blocks': len(text_blocks)
        }

    def _convert_ocr_to_blocks(self, ocr_results: List[Dict], zoom: float) -> List[Dict]:
        """Convert OCR results to text block format"""
        text_blocks = []

        for result in ocr_results:
            # OCR coordinates are in image space, convert to PDF space
            bbox_coords = result['bbox']

            # Calculate bounding box in PDF coordinates
            x_coords = [p[0] / zoom for p in bbox_coords]
            y_coords = [p[1] / zoom for p in bbox_coords]

            x = min(x_coords)
            y = min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y

            text_blocks.append({
                'text': result['text'],
                'confidence': result['confidence'],
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                },
                'font': None,  # OCR doesn't provide font info
                'font_size': None,
                'color': None
            })

        return text_blocks

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        # Simple character-level similarity
        # For production, consider using difflib or rapidfuzz

        if not text1 and not text2:
            return 1.0

        if not text1 or not text2:
            return 0.0

        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Simple approach: count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        max_len = max(len(text1), len(text2))

        return matches / max_len if max_len > 0 else 0.0

    def set_language(self, lang: str):
        """
        Change OCR language.

        Args:
            lang: Language code (e.g., 'en', 'ch', 'fr')
        """
        self.ocr_adapter.set_language(lang)
        logger.info(f"OCR language changed to: {lang}")

    def __repr__(self):
        return f"TextRecognizer(backend={self.ocr_backend}, dpi={self.dpi})"
