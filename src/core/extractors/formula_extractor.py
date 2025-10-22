"""
Formula Extractor - Extract mathematical formulas using LaTeX-OCR
Preserves LaTeX representation for perfect reconstruction.
"""

from pix2tex.cli import LatexOCR
import fitz
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import base64
from PIL import Image


class FormulaExtractor:
    """
    Extract mathematical formulas using LaTeX-OCR.
    Preserves LaTeX representation for perfect reconstruction.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)
        self.preserve_as_image = self.config.get('preserve_as_image', True)  # Changed to True by default
        self.model = None
        self.dpi = self.config.get('dpi', 150)  # Store DPI for coordinate conversion

    def extract_formulas(self, pdf_path: str) -> List[Dict]:
        """
        Extract all formulas from PDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            List of formula dictionaries
        """
        if not self.model:
            try:
                self.model = LatexOCR()
            except Exception as e:
                print(f"Warning: LatexOCR failed to load: {e}")
                return []

        formulas = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=self.dpi)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Calculate scale factor from image pixels to PDF points
            # PDF points = image pixels * (72 / DPI)
            scale_to_pdf = 72.0 / self.dpi

            # Get page dimensions for coordinate validation
            page_width = page.rect.width
            page_height = page.rect.height

            # Detect formula regions
            formula_regions = self._detect_formula_regions(img)

            for region in formula_regions:
                # Extract formula image (in pixel coordinates)
                x_px, y_px, w_px, h_px = region
                formula_img = img[y_px:y_px+h_px, x_px:x_px+w_px]

                # Convert to LaTeX
                try:
                    # Convert numpy array to PIL Image for LaTeX OCR
                    formula_img_pil = Image.fromarray(cv2.cvtColor(formula_img, cv2.COLOR_BGR2RGB))
                    latex = self.model(formula_img_pil)
                    confidence = self._calculate_confidence(formula_img, latex)

                    if confidence > self.confidence_threshold:
                        # Convert pixel coordinates to PDF points
                        x_pdf = x_px * scale_to_pdf
                        y_pdf = y_px * scale_to_pdf
                        w_pdf = w_px * scale_to_pdf
                        h_pdf = h_px * scale_to_pdf

                        # Validate coordinates are within page bounds
                        if x_pdf < page_width and y_pdf < page_height:
                            formula_dict = {
                                'page': page_num,
                                'bbox': {'x': x_pdf, 'y': y_pdf, 'width': w_pdf, 'height': h_pdf},
                                'latex': latex,
                                'confidence': confidence,
                                'type': self._classify_formula(latex)
                            }

                            if self.preserve_as_image:
                                # Save formula image as base64 PNG
                                _, img_encoded = cv2.imencode('.png', formula_img)
                                formula_dict['image_data'] = base64.b64encode(img_encoded).decode('utf-8')

                            formulas.append(formula_dict)

                except Exception as e:
                    print(f"Formula extraction failed for region {region}: {e}")

        doc.close()
        return formulas

    def _detect_formula_regions(self, image: np.ndarray) -> List[Tuple]:
        """Detect regions likely containing formulas"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to find formula-like regions
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio and size
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 10 and w > 30 and h > 20:
                # Check for formula characteristics
                region = gray[y:y+h, x:x+w]
                if self._is_formula_region(region):
                    regions.append((x, y, w, h))

        return regions

    def _is_formula_region(self, region: np.ndarray) -> bool:
        """Check if region likely contains formula"""

        # Check for mathematical symbols
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Formulas typically have moderate edge density
        return 0.05 < edge_density < 0.3

    def _calculate_confidence(self, image: np.ndarray, latex: str) -> float:
        """Calculate extraction confidence"""

        # Basic confidence based on LaTeX structure
        confidence = 0.5

        # Check for valid LaTeX commands
        if '\\' in latex:
            confidence += 0.2

        # Check for mathematical operators
        if any(op in latex for op in ['+', '-', '=', '\\times', '\\div']):
            confidence += 0.15

        # Check for common formula patterns
        if any(pattern in latex for pattern in ['\\frac', '\\sqrt', '^', '_']):
            confidence += 0.15

        return min(confidence, 1.0)

    def _classify_formula(self, latex: str) -> str:
        """Classify formula type"""

        if '\\int' in latex:
            return 'integral'
        elif '\\sum' in latex:
            return 'summation'
        elif '\\frac' in latex:
            return 'fraction'
        elif '^' in latex or '_' in latex:
            return 'exponent'
        elif '\\sqrt' in latex:
            return 'root'
        else:
            return 'general'

    def is_formula_text(self, text: str) -> bool:
        """Quick check if text might contain formula"""
        formula_indicators = ['=', '+', '-', '×', '÷', '∫', '∑', '√', '∂', '∞', '≈', '≠', '≤', '≥']
        return any(indicator in text for indicator in formula_indicators)
