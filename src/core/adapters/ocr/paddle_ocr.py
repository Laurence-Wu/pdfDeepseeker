#!/usr/bin/env python3
"""
PaddleOCR Adapter for Text Recognition
Provides a unified interface to PaddleOCR for text detection and recognition.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import cv2

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    logging.warning("PaddleOCR not installed. Install with: pip install paddleocr")

logger = logging.getLogger(__name__)


class PaddleOCRAdapter:
    """
    Adapter for PaddleOCR providing text detection and recognition.

    Features:
    - Multi-language support (80+ languages)
    - Text detection with bounding boxes
    - Text recognition with confidence scores
    - Angle classification for rotated text
    - GPU acceleration support
    """

    def __init__(self, config: Dict = None):
        """
        Initialize PaddleOCR adapter.

        Args:
            config: Configuration dictionary with options:
                - lang: Language code (default: 'en')
                - use_angle_cls: Enable angle classification (default: True)
                - use_gpu: Use GPU acceleration (default: False)
                - det_db_thresh: Detection threshold (default: 0.3)
                - det_db_box_thresh: Box threshold (default: 0.5)
                - rec_batch_num: Recognition batch size (default: 6)
                - show_log: Show PaddleOCR logs (default: False)
        """
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddleocr"
            )

        self.config = config or {}

        # Extract configuration
        self.lang = self.config.get('lang', 'en')
        self.use_angle_cls = self.config.get('use_angle_cls', True)
        self.use_gpu = self.config.get('use_gpu', False)
        self.det_db_thresh = self.config.get('det_db_thresh', 0.3)
        self.det_db_box_thresh = self.config.get('det_db_box_thresh', 0.5)
        self.rec_batch_num = self.config.get('rec_batch_num', 6)
        self.show_log = self.config.get('show_log', False)

        # Initialize PaddleOCR
        logger.info(f"Initializing PaddleOCR (lang={self.lang}, gpu={self.use_gpu})")

        try:
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                use_gpu=self.use_gpu,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                rec_batch_num=self.rec_batch_num,
                show_log=self.show_log
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def recognize_text(
        self,
        image: Union[str, np.ndarray, Path],
        cls: bool = None
    ) -> List[Dict]:
        """
        Recognize text from an image.

        Args:
            image: Image path, numpy array, or Path object
            cls: Override angle classification setting

        Returns:
            List of recognition results with:
                - text: Recognized text
                - confidence: Recognition confidence (0-1)
                - bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - angle: Text angle (if angle classification enabled)
        """
        try:
            # Convert Path to string
            if isinstance(image, Path):
                image = str(image)

            # Run OCR
            use_cls = cls if cls is not None else self.use_angle_cls
            results = self.ocr.ocr(image, cls=use_cls)

            if not results or not results[0]:
                logger.warning("No text detected in image")
                return []

            # Parse results
            parsed_results = []
            for line in results[0]:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)

                parsed_results.append({
                    'text': text_info[0],
                    'confidence': float(text_info[1]),
                    'bbox': bbox,
                    'center': self._calculate_center(bbox),
                    'width': self._calculate_width(bbox),
                    'height': self._calculate_height(bbox)
                })

            logger.info(f"Recognized {len(parsed_results)} text blocks")
            return parsed_results

        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            raise

    def detect_text(
        self,
        image: Union[str, np.ndarray, Path]
    ) -> List[Dict]:
        """
        Detect text regions without recognition.

        Args:
            image: Image path, numpy array, or Path object

        Returns:
            List of detection results with bounding boxes
        """
        try:
            # Convert Path to string
            if isinstance(image, Path):
                image = str(image)

            # Load image if path
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image

            # Run detection only
            from paddleocr import PPStructure

            # Alternative: use detection model directly
            # This is more efficient when you only need bounding boxes
            results = self.ocr.ocr(image, rec=False)

            if not results or not results[0]:
                return []

            detections = []
            for bbox in results[0]:
                detections.append({
                    'bbox': bbox,
                    'center': self._calculate_center(bbox),
                    'width': self._calculate_width(bbox),
                    'height': self._calculate_height(bbox)
                })

            return detections

        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise

    def recognize_from_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        dpi: int = 300
    ) -> List[Dict]:
        """
        Recognize text from a specific PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            dpi: DPI for PDF rendering (default: 300)

        Returns:
            List of recognition results
        """
        try:
            import fitz  # PyMuPDF

            # Open PDF
            doc = fitz.open(pdf_path)

            if page_num >= doc.page_count:
                raise ValueError(f"Page {page_num} out of range (total: {doc.page_count})")

            # Render page to image
            page = doc[page_num]
            zoom = dpi / 72  # 72 DPI is default
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            doc.close()

            # Run OCR
            return self.recognize_text(img)

        except Exception as e:
            logger.error(f"PDF page recognition failed: {e}")
            raise

    def batch_recognize(
        self,
        images: List[Union[str, np.ndarray, Path]],
        cls: bool = None
    ) -> List[List[Dict]]:
        """
        Batch recognize text from multiple images.

        Args:
            images: List of images (paths or arrays)
            cls: Override angle classification setting

        Returns:
            List of recognition results for each image
        """
        results = []

        for i, image in enumerate(images):
            try:
                result = self.recognize_text(image, cls=cls)
                results.append(result)
                logger.info(f"Processed image {i+1}/{len(images)}")
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append([])

        return results

    def set_language(self, lang: str):
        """
        Change OCR language.

        Supported languages: ch, en, korean, japan, french, german,
        italian, spanish, portuguese, russian, arabic, hindi, etc.

        Args:
            lang: Language code
        """
        logger.info(f"Changing language from {self.lang} to {lang}")

        self.lang = lang
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
            det_db_thresh=self.det_db_thresh,
            det_db_box_thresh=self.det_db_box_thresh,
            rec_batch_num=self.rec_batch_num,
            show_log=self.show_log
        )

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        # Common PaddleOCR supported languages
        return [
            'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin',
            'arabic', 'cyrillic', 'devanagari', 'french', 'german', 'italian',
            'spanish', 'portuguese', 'russian', 'hindi', 'bengali', 'ug', 'fa',
            'ur', 'rs_latin', 'oc', 'rsc', 'bg', 'uk', 'be', 'te', 'kn', 'ta',
            'ml', 'ne', 'si', 'mr', 'hi', 'sa', 'en_symbols'
        ]

    # Helper methods

    @staticmethod
    def _calculate_center(bbox: List[List[float]]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _calculate_width(bbox: List[List[float]]) -> float:
        """Calculate width of bounding box"""
        xs = [point[0] for point in bbox]
        return max(xs) - min(xs)

    @staticmethod
    def _calculate_height(bbox: List[List[float]]) -> float:
        """Calculate height of bounding box"""
        ys = [point[1] for point in bbox]
        return max(ys) - min(ys)

    def __repr__(self):
        return f"PaddleOCRAdapter(lang={self.lang}, gpu={self.use_gpu})"
