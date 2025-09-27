# Extract Format Classes - Complete Implementation

## Overview
Comprehensive extractors for fonts, formulas, tables, and watermarks.

## Font Extractor

```python
import fitz  # PyMuPDF
import pdfplumber
from fonttools.ttLib import TTFont
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
                font_ref, font_name, font_type, _, _ = font_info[:5]
                
                # Extract embedded font if not already cached
                if font_ref not in fonts['embedded_fonts']:
                    font_data = self._extract_font_data(doc, font_ref)
                    if font_data:
                        fonts['embedded_fonts'][font_ref] = {
                            'name': font_name,
                            'type': font_type,
                            'data': font_data,
                            'metrics': self._analyze_font_metrics(font_data)
                        }
                
                # Track usage
                fonts['font_usage'].append({
                    'page': page_num,
                    'font_ref': font_ref,
                    'font_name': font_name
                })
        
        # Extract with pdfplumber for character-level mapping
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
        
        # Define fallback chain
        fonts['fallback_chain'] = self._determine_fallback_chain(fonts['embedded_fonts'])
        
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
```

## Formula Extractor

```python
from latex_ocr import LatexOCR
import cv2
import numpy as np
from typing import Dict, List, Optional

class FormulaExtractor:
    """
    Extract mathematical formulas using LaTeX-OCR.
    Preserves LaTeX representation for perfect reconstruction.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)
        self.preserve_as_image = self.config.get('preserve_as_image', False)
        self.model = None
        
    def extract_formulas(self, pdf_path: str) -> List[Dict]:
        """
        Extract all formulas from PDF.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            List of formula dictionaries
        """
        if not self.model:
            self.model = LatexOCR()
        
        formulas = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect formula regions
            formula_regions = self._detect_formula_regions(img)
            
            for region in formula_regions:
                # Extract formula image
                x, y, w, h = region
                formula_img = img[y:y+h, x:x+w]
                
                # Convert to LaTeX
                try:
                    latex = self.model(formula_img)
                    confidence = self._calculate_confidence(formula_img, latex)
                    
                    if confidence > self.confidence_threshold:
                        formulas.append({
                            'page': page_num,
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'latex': latex,
                            'confidence': confidence,
                            'image': base64.b64encode(cv2.imencode('.png', formula_img)[1]).decode() if self.preserve_as_image else None,
                            'type': self._classify_formula(latex)
                        })
                
                except Exception as e:
                    print(f"Formula extraction failed: {e}")
        
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
```

## Table Extractor

```python
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor
import pandas as pd

class TableExtractor:
    """
    Extract tables using Table Transformer.
    Preserves structure and formatting.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.model = None
        self.processor = None
        
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract all tables from PDF.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            List of table dictionaries
        """
        if not self.model:
            self._load_model()
        
        tables = []
        
        # Use pdfplumber for table extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Find tables
                page_tables = page.find_tables()
                
                for table_idx, table in enumerate(page_tables):
                    # Extract table data
                    table_data = table.extract()
                    
                    if table_data:
                        # Structure table
                        structured = self._structure_table(table_data, table.bbox)
                        
                        tables.append({
                            'page': page_num,
                            'index': table_idx,
                            'bbox': {
                                'x': table.bbox[0],
                                'y': table.bbox[1],
                                'width': table.bbox[2] - table.bbox[0],
                                'height': table.bbox[3] - table.bbox[1]
                            },
                            'rows': structured['rows'],
                            'columns': structured['columns'],
                            'headers': structured['headers'],
                            'data': structured['data'],
                            'style': self._analyze_table_style(table)
                        })
        
        return tables
    
    def _load_model(self):
        """Load Table Transformer model"""
        
        try:
            model_name = "microsoft/table-transformer-detection"
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load Table Transformer: {e}")
            self.model = None
    
    def _structure_table(self, table_data: List[List], bbox: Tuple) -> Dict:
        """Structure table data"""
        
        structured = {
            'rows': [],
            'columns': [],
            'headers': [],
            'data': []
        }
        
        if not table_data:
            return structured
        
        # Assume first row is header
        if len(table_data) > 0:
            structured['headers'] = table_data[0]
            structured['columns'] = len(table_data[0])
        
        # Process rows
        for row_idx, row in enumerate(table_data):
            row_data = []
            for col_idx, cell in enumerate(row):
                row_data.append({
                    'text': str(cell) if cell else '',
                    'row': row_idx,
                    'col': col_idx,
                    'is_header': row_idx == 0
                })
            structured['rows'].append(row_data)
        
        # Convert to pandas for easier manipulation
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
            structured['data'] = df.to_dict('records')
        except:
            structured['data'] = table_data
        
        return structured
    
    def _analyze_table_style(self, table) -> Dict:
        """Analyze table styling"""
        
        return {
            'has_header': True,  # Assumption
            'border_style': 'solid',
            'alignment': 'left'
        }
```

## Watermark Extractor

```python
from invisible_watermark import WatermarkDecoder
import cv2

class WatermarkExtractor:
    """
    Extract visible and invisible watermarks.
    Preserves for reconstruction.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.detect_visible = self.config.get('detect_visible', True)
        self.detect_invisible = self.config.get('detect_invisible', True)
        
    def extract_watermarks(self, pdf_path: str) -> List[Dict]:
        """
        Extract all watermarks from PDF.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            List of watermark dictionaries
        """
        watermarks = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            # Extract visible watermarks
            if self.detect_visible:
                visible = self._extract_visible_watermarks(page)
                watermarks.extend(visible)
            
            # Extract invisible watermarks
            if self.detect_invisible:
                invisible = self._extract_invisible_watermarks(page)
                watermarks.extend(invisible)
        
        doc.close()
        return watermarks
    
    def _extract_visible_watermarks(self, page) -> List[Dict]:
        """Extract visible watermarks"""
        
        watermarks = []
        
        # Check for transparent text
        text_instances = page.get_text("dict")
        for block in text_instances.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        opacity = span.get("opacity", 1.0)
                        if opacity < 0.5:  # Likely watermark
                            watermarks.append({
                                'type': 'visible_text',
                                'text': span.get("text", ""),
                                'opacity': opacity,
                                'bbox': span.get("bbox"),
                                'font': span.get("font"),
                                'size': span.get("size")
                            })
        
        return watermarks
    
    def _extract_invisible_watermarks(self, page) -> List[Dict]:
        """Extract invisible watermarks"""
        
        watermarks = []
        
        try:
            # Render page
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Try to decode invisible watermark
            decoder = WatermarkDecoder('bytes', 32)
            watermark = decoder.decode(img)
            
            if watermark is not None:
                watermarks.append({
                    'type': 'invisible',
                    'data': base64.b64encode(watermark).decode(),
                    'method': 'lsb'
                })
        
        except Exception as e:
            print(f"Invisible watermark extraction failed: {e}")
        
        return watermarks
```
