"""
OCR adapters for text recognition.
Supports multiple OCR backends including PaddleOCR.
"""

from .paddle_ocr import PaddleOCRAdapter

__all__ = ['PaddleOCRAdapter']
