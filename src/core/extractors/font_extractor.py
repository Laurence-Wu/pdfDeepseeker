"""
Font Extractor - Extract and analyze embedded fonts from PDFs
Preserves font files for exact reconstruction.
"""

import fitz  # PyMuPDF
import pdfplumber
try:
    from fonttools.ttLib import TTFont
    FONTTOOLS_AVAILABLE = True
except ImportError:
    FONTTOOLS_AVAILABLE = False
    print("Warning: fonttools not available, font metrics will be limited")

from typing import Dict, List, Optional, Tuple
import io
import base64


class FontExtractor:
    """
    Extract and analyze embedded fonts from PDFs.
    Preserves font files for exact reconstruction.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.extract_embedded = self.config.get('extract_embedded', True)
        self.cache_fonts = self.config.get('cache_fonts', True)
        self.font_cache = {}

    def extract_all_fonts(self, pdf_path: str) -> Dict:
        """
        Extract all fonts from PDF with complete metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary of font information
        """
        fonts = {
            'embedded_fonts': {},
            'font_usage': [],
            'font_mapping': {},
            'fallback_chain': []
        }

        # Extract with PyMuPDF
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            # Get font list for page
            font_list = page.get_fonts(full=True)

            for font_info in font_list:
                font_ref = font_info[0]
                font_name = font_info[1]
                font_type = font_info[2]

                # Extract embedded font if not already cached
                if font_ref not in fonts['embedded_fonts']:
                    font_data = self._extract_font_data(doc, font_ref)
                    if font_data:
                        # font_data might be bytes or tuple from PyMuPDF
                        if isinstance(font_data, tuple):
                            font_bytes = font_data[0] if font_data else b''
                        else:
                            font_bytes = font_data if isinstance(font_data, bytes) else b''

                        fonts['embedded_fonts'][font_ref] = {
                            'name': font_name,
                            'type': font_type,
                            'data': base64.b64encode(font_bytes).decode() if font_bytes else '',
                            'data_bytes': font_bytes,
                            'metrics': self._analyze_font_metrics(font_bytes) if font_bytes else {}
                        }

                # Track usage
                fonts['font_usage'].append({
                    'page': page_num,
                    'font_ref': font_ref,
                    'font_name': font_name
                })

        # Extract with pdfplumber for character-level mapping
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    chars = page.chars

                    for char in chars:
                        font_name = char.get('fontname', 'unknown')
                        char_text = char.get('text', '')

                        # Map characters to fonts
                        if font_name not in fonts['font_mapping']:
                            fonts['font_mapping'][font_name] = []

                        fonts['font_mapping'][font_name].append({
                            'char': char_text,
                            'page': page_num,
                            'bbox': {
                                'x': char['x0'],
                                'y': char['top'],
                                'width': char['width'],
                                'height': char['height']
                            },
                            'size': char.get('size', 12)
                        })
        except Exception as e:
            print(f"Warning: pdfplumber extraction failed: {e}")

        # Define fallback chain
        fonts['fallback_chain'] = self._determine_fallback_chain(fonts['embedded_fonts'])

        # Cache fonts
        if self.cache_fonts:
            self.font_cache = fonts['embedded_fonts']

        doc.close()
        return fonts

    def _extract_font_data(self, doc: fitz.Document, font_ref: int) -> Optional[bytes]:
        """Extract embedded font data from PDF"""

        try:
            # Get font buffer
            font_buffer = doc.extract_font(font_ref)
            if font_buffer:
                return font_buffer[0]  # Return font binary data
        except Exception as e:
            print(f"Failed to extract font {font_ref}: {e}")

        return None

    def _analyze_font_metrics(self, font_data: bytes) -> Dict:
        """Analyze font metrics for text measurement"""

        metrics = {
            'ascent': 0,
            'descent': 0,
            'line_height': 0,
            'avg_width': 0,
            'char_widths': {}
        }

        if not FONTTOOLS_AVAILABLE:
            return metrics

        try:
            # Load font with fonttools
            font = TTFont(io.BytesIO(font_data))

            # Get metrics
            if 'hhea' in font:
                metrics['ascent'] = font['hhea'].ascent
                metrics['descent'] = font['hhea'].descent
                metrics['line_height'] = font['hhea'].lineGap

            # Get character widths
            if 'hmtx' in font:
                hmtx = font['hmtx']
                for char_name, (width, lsb) in hmtx.metrics.items():
                    metrics['char_widths'][char_name] = width

                widths = [w for w, _ in hmtx.metrics.values()]
                metrics['avg_width'] = sum(widths) / len(widths) if widths else 0

        except Exception as e:
            print(f"Failed to analyze font metrics: {e}")

        return metrics

    def _determine_fallback_chain(self, embedded_fonts: Dict) -> List[str]:
        """Determine font fallback chain"""

        # Priority order
        priority_fonts = ['Arial', 'Helvetica', 'Times', 'Times New Roman', 'Calibri']

        fallback_chain = []
        font_names = [f['name'] for f in embedded_fonts.values()]

        # Add priority fonts first
        for priority in priority_fonts:
            if any(priority.lower() in name.lower() for name in font_names):
                fallback_chain.append(priority)

        # Add remaining fonts
        for name in font_names:
            if not any(p.lower() in name.lower() for p in priority_fonts):
                fallback_chain.append(name)

        # Add system defaults
        fallback_chain.extend(['Arial', 'Times New Roman', 'sans-serif'])

        # Remove duplicates while preserving order
        seen = set()
        return [x for x in fallback_chain if not (x in seen or seen.add(x))]

    def measure_text(self, text: str, font_ref: str, font_size: float) -> float:
        """
        Measure text width using font metrics.

        Args:
            text: Text to measure
            font_ref: Font reference
            font_size: Font size in points

        Returns:
            Width in points
        """
        if font_ref not in self.font_cache:
            return len(text) * font_size * 0.5  # Fallback estimation

        font_metrics = self.font_cache[font_ref]['metrics']
        char_widths = font_metrics.get('char_widths', {})
        avg_width = font_metrics.get('avg_width', 500)

        total_width = 0
        for char in text:
            # Look up character width
            width = char_widths.get(char, avg_width)
            total_width += width

        # Scale to font size (font units to points)
        return (total_width / 1000) * font_size

    def get_font_info(self, font_ref: str) -> Optional[Dict]:
        """Get font information by reference"""
        return self.font_cache.get(font_ref)
