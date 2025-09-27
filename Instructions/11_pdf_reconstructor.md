# PDF Reconstructor - Complete Implementation

## Overview
Reconstruct PDF with translations while preserving exact layout, fonts, and formatting.

## Implementation

```python
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from typing import Dict, List, Optional, Tuple
import io
import base64
import numpy as np

class PDFReconstructor:
    """
    Reconstruct PDF documents with translated content.
    Preserves original layout, fonts, and all visual elements.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.preserve_margins = self.config.get('preserve_margins', True)
        self.preserve_fonts = self.config.get('preserve_fonts', True)
        self.quality_settings = self.config.get('quality', {
            'image_dpi': 300,
            'compression': 'lossless',
            'pdf_version': '1.7'
        })
        self.registered_fonts = {}
        
    def reconstruct_pdf(self,
                        original_pdf: str,
                        translated_content: Dict,
                        output_path: str) -> bool:
        """
        Reconstruct PDF with translated content.
        
        Args:
            original_pdf: Path to original PDF
            translated_content: XLIFF parsed content with translations
            output_path: Output PDF path
            
        Returns:
            Success status
        """
        try:
            # Open original for reference
            original_doc = fitz.open(original_pdf)
            
            # Create new document
            new_doc = fitz.open()
            
            # Get skeleton data
            skeleton = translated_content.get('skeleton', {})
            
            # Register fonts
            if self.preserve_fonts:
                self._register_fonts(skeleton.get('fonts', []))
            
            # Process each page
            for page_num in range(original_doc.page_count):
                # Create new page with original dimensions
                original_page = original_doc[page_num]
                page_rect = original_page.rect
                new_page = new_doc.new_page(
                    width=page_rect.width,
                    height=page_rect.height
                )
                
                # Copy non-text elements
                self._copy_non_text_elements(original_page, new_page)
                
                # Add translated text
                self._add_translated_text(
                    new_page,
                    translated_content,
                    page_num,
                    skeleton
                )
                
                # Handle special elements
                self._handle_special_elements(
                    original_page,
                    new_page,
                    translated_content,
                    page_num
                )
            
            # Save document
            new_doc.save(output_path, 
                        garbage=4,
                        deflate=True,
                        clean=True)
            
            # Cleanup
            original_doc.close()
            new_doc.close()
            
            return True
            
        except Exception as e:
            print(f"Reconstruction failed: {e}")
            return False
    
    def _register_fonts(self, fonts: List[Dict]):
        """Register embedded fonts for use"""
        
        for font_data in fonts:
            if font_data.get('embedded') and font_data.get('data'):
                try:
                    # Decode font data
                    font_bytes = base64.b64decode(font_data['data'])
                    
                    # Save to temporary file
                    font_path = f"/tmp/{font_data['name']}.ttf"
                    with open(font_path, 'wb') as f:
                        f.write(font_bytes)
                    
                    # Register with PyMuPDF
                    font_name = font_data['name']
                    self.registered_fonts[font_name] = font_path
                    
                except Exception as e:
                    print(f"Failed to register font {font_data.get('name')}: {e}")
    
    def _copy_non_text_elements(self, original_page, new_page):
        """Copy images, graphics, etc. from original page"""
        
        # Get all images
        image_list = original_page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(original_page.parent, xref)
                
                if pix.n - pix.alpha > 3:  # Convert to RGB if needed
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Get image position
                img_rect = original_page.get_image_bbox(img[7])
                
                # Insert into new page
                img_data = pix.tobytes("png")
                new_page.insert_image(
                    img_rect,
                    stream=img_data,
                    keep_proportion=True
                )
                
            except Exception as e:
                print(f"Failed to copy image {img_index}: {e}")
        
        # Copy vector graphics
        self._copy_vector_graphics(original_page, new_page)
    
    def _copy_vector_graphics(self, original_page, new_page):
        """Copy vector graphics and shapes"""
        
        # Get drawing commands
        drawings = original_page.get_drawings()
        
        for drawing in drawings:
            try:
                if drawing['type'] == 'line':
                    new_page.draw_line(
                        drawing['start'],
                        drawing['end'],
                        color=drawing.get('color', (0, 0, 0)),
                        width=drawing.get('width', 1)
                    )
                elif drawing['type'] == 'rect':
                    new_page.draw_rect(
                        drawing['rect'],
                        color=drawing.get('stroke_color', (0, 0, 0)),
                        fill=drawing.get('fill_color'),
                        width=drawing.get('width', 1)
                    )
                elif drawing['type'] == 'curve':
                    new_page.draw_bezier(
                        drawing['points'],
                        color=drawing.get('color', (0, 0, 0)),
                        width=drawing.get('width', 1)
                    )
            except Exception as e:
                print(f"Failed to copy drawing: {e}")
    
    def _add_translated_text(self, page, translated_content: Dict, 
                           page_num: int, skeleton: Dict):
        """Add translated text to page"""
        
        # Get units for this page
        page_units = self._get_page_units(translated_content, page_num)
        
        for unit in page_units:
            if not unit.get('target'):  # Skip if no translation
                continue
            
            # Get metadata
            metadata = unit.get('metadata', {})
            position = metadata.get('position', {})
            style = metadata.get('style', {})
            
            # Prepare text insertion
            text = unit['target']
            point = fitz.Point(
                position.get('x', 0),
                position.get('y', 0) + position.get('height', 0)  # Bottom-left
            )
            
            # Prepare font
            fontname = self._get_font_name(style.get('font', 'helv'))
            fontsize = style.get('size', 12)
            color = self._parse_color(style.get('color', '#000000'))
            
            # Check text length constraint
            if metadata.get('constraint'):
                text = self._fit_text_to_constraint(
                    text, 
                    metadata['constraint'],
                    fontname,
                    fontsize,
                    position
                )
            
            # Handle special formatting
            if style.get('weight') == 'bold':
                fontname = self._get_bold_variant(fontname)
            if style.get('italic') == 'true':
                fontname = self._get_italic_variant(fontname)
            
            # Insert text
            try:
                rc = page.insert_text(
                    point,
                    text,
                    fontname=fontname,
                    fontsize=fontsize,
                    color=color,
                    render_mode=0,  # Fill text
                    rotate=metadata.get('rotation', 0)
                )
                
                # Check if text overflowed
                if rc < 0:
                    print(f"Text overflow for unit {unit['id']}")
                    # Try to fit with smaller font
                    self._handle_text_overflow(
                        page, point, text, fontname, 
                        fontsize, color, position
                    )
                    
            except Exception as e:
                print(f"Failed to insert text for unit {unit['id']}: {e}")
    
    def _handle_special_elements(self, original_page, new_page, 
                                translated_content: Dict, page_num: int):
        """Handle special elements like tables, formulas, etc."""
        
        skeleton = translated_content.get('skeleton', {})
        
        # Handle tables
        tables = self._get_page_tables(skeleton, page_num)
        for table in tables:
            self._reconstruct_table(new_page, table, translated_content)
        
        # Handle formulas (preserve as-is)
        formulas = self._get_page_formulas(skeleton, page_num)
        for formula in formulas:
            self._preserve_formula(original_page, new_page, formula)
        
        # Handle watermarks
        watermarks = skeleton.get('watermarks', [])
        for watermark in watermarks:
            self._apply_watermark(new_page, watermark)
        
        # Handle form fields
        form_fields = self._get_page_form_fields(skeleton, page_num)
        for field in form_fields:
            self._reconstruct_form_field(new_page, field, translated_content)
        
        # Handle annotations
        annotations = original_page.annots()
        for annot in annotations:
            self._copy_annotation(annot, new_page)
    
    def _reconstruct_table(self, page, table: Dict, translated_content: Dict):
        """Reconstruct table with translated content"""
        
        bbox = table.get('bbox', {})
        rows = table.get('rows', [])
        
        if not rows:
            return
        
        # Calculate cell dimensions
        num_rows = len(rows)
        num_cols = len(rows[0]) if rows else 0
        
        if num_cols == 0:
            return
        
        cell_width = bbox.get('width', 100) / num_cols
        cell_height = bbox.get('height', 100) / num_rows
        
        # Draw table structure
        x_start = bbox.get('x', 0)
        y_start = bbox.get('y', 0)
        
        # Draw grid
        for i in range(num_rows + 1):
            y = y_start + i * cell_height
            page.draw_line(
                fitz.Point(x_start, y),
                fitz.Point(x_start + bbox['width'], y),
                color=(0, 0, 0),
                width=0.5
            )
        
        for j in range(num_cols + 1):
            x = x_start + j * cell_width
            page.draw_line(
                fitz.Point(x, y_start),
                fitz.Point(x, y_start + bbox['height']),
                color=(0, 0, 0),
                width=0.5
            )
        
        # Add cell content
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                # Find translated text for this cell
                cell_text = self._get_table_cell_translation(
                    translated_content, table['id'], row_idx, col_idx
                )
                
                if cell_text:
                    # Calculate cell position
                    cell_x = x_start + col_idx * cell_width + 5  # Padding
                    cell_y = y_start + row_idx * cell_height + cell_height / 2
                    
                    # Insert text
                    page.insert_text(
                        fitz.Point(cell_x, cell_y),
                        cell_text,
                        fontsize=9,
                        color=(0, 0, 0)
                    )
    
    def _preserve_formula(self, original_page, new_page, formula: Dict):
        """Copy formula as image"""
        
        bbox = formula.get('bbox', {})
        
        # Extract formula region from original
        rect = fitz.Rect(
            bbox['x'],
            bbox['y'],
            bbox['x'] + bbox['width'],
            bbox['y'] + bbox['height']
        )
        
        # Get pixmap of region
        mat = fitz.Matrix(2, 2)  # 2x zoom for quality
        pix = original_page.get_pixmap(matrix=mat, clip=rect)
        
        # Insert into new page
        new_page.insert_image(rect, pixmap=pix)
    
    def _apply_watermark(self, page, watermark: Dict):
        """Apply watermark to page"""
        
        if watermark['type'] == 'visible_text':
            # Add transparent text watermark
            page_rect = page.rect
            
            # Calculate diagonal position
            x = page_rect.width / 2
            y = page_rect.height / 2
            
            # Insert rotated transparent text
            page.insert_text(
                fitz.Point(x, y),
                watermark.get('text', 'WATERMARK'),
                fontsize=watermark.get('size', 48),
                color=(0.8, 0.8, 0.8),  # Light gray
                rotate=45,
                render_mode=1  # Stroke text
            )
        
        elif watermark['type'] == 'image':
            # Add image watermark
            if watermark.get('data'):
                img_data = base64.b64decode(watermark['data'])
                page.insert_image(
                    page.rect,
                    stream=img_data,
                    overlay=True,
                    keep_proportion=True
                )
    
    def _reconstruct_form_field(self, page, field: Dict, translated_content: Dict):
        """Reconstruct form field with translated label"""
        
        # Find translated label
        translated_label = self._get_form_field_translation(
            translated_content, field['id']
        )
        
        if translated_label:
            # Add label
            label_pos = field.get('label_position', {})
            page.insert_text(
                fitz.Point(label_pos.get('x', 0), label_pos.get('y', 0)),
                translated_label,
                fontsize=10,
                color=(0, 0, 0)
            )
        
        # Recreate form field
        field_rect = fitz.Rect(
            field['bbox']['x'],
            field['bbox']['y'],
            field['bbox']['x'] + field['bbox']['width'],
            field['bbox']['y'] + field['bbox']['height']
        )
        
        # Draw field border
        page.draw_rect(field_rect, color=(0, 0, 0), width=0.5)
    
    def _copy_annotation(self, annot, new_page):
        """Copy annotation to new page"""
        
        try:
            # Get annotation info
            info = annot.info
            
            # Create new annotation
            new_annot = new_page.add_text_annot(
                annot.rect.top_left,
                info.get('content', ''),
            )
            
            # Copy properties
            new_annot.set_info(info)
            new_annot.update()
            
        except Exception as e:
            print(f"Failed to copy annotation: {e}")
    
    def _fit_text_to_constraint(self, text: str, constraint: Dict,
                               fontname: str, fontsize: float,
                               position: Dict) -> str:
        """Fit text to length constraint"""
        
        max_length = constraint.get('maxLength')
        if not max_length:
            return text
        
        # Measure text
        text_width = self._measure_text(text, fontname, fontsize)
        box_width = position.get('width', 100)
        
        if text_width > box_width:
            # Try to shorten
            while text_width > box_width and len(text) > 3:
                text = text[:-1]
                text_width = self._measure_text(text + '...', fontname, fontsize)
            
            if len(text) < len(constraint.get('original', '')):
                text += '...'
        
        return text
    
    def _handle_text_overflow(self, page, point: fitz.Point, text: str,
                            fontname: str, fontsize: float, color: Tuple,
                            position: Dict):
        """Handle text that doesn't fit"""
        
        # Try progressively smaller font sizes
        min_size = 6
        current_size = fontsize
        
        while current_size > min_size:
            current_size -= 0.5
            
            # Try again with smaller font
            rc = page.insert_text(
                point,
                text,
                fontname=fontname,
                fontsize=current_size,
                color=color
            )
            
            if rc >= 0:  # Success
                break
        
        if current_size <= min_size:
            # Last resort: truncate text
            truncated = self._truncate_to_fit(
                text, position.get('width', 100),
                fontname, min_size
            )
            
            page.insert_text(
                point,
                truncated,
                fontname=fontname,
                fontsize=min_size,
                color=color
            )
    
    def _measure_text(self, text: str, fontname: str, fontsize: float) -> float:
        """Estimate text width"""
        
        # Rough estimation (would be better with actual font metrics)
        avg_char_width = fontsize * 0.5
        return len(text) * avg_char_width
    
    def _truncate_to_fit(self, text: str, max_width: float,
                        fontname: str, fontsize: float) -> str:
        """Truncate text to fit width"""
        
        char_width = fontsize * 0.5
        max_chars = int(max_width / char_width)
        
        if len(text) > max_chars:
            return text[:max_chars-3] + '...'
        
        return text
    
    def _get_page_units(self, translated_content: Dict, page_num: int) -> List[Dict]:
        """Get translation units for specific page"""
        
        units = []
        
        for file_data in translated_content.get('files', []):
            for unit in file_data.get('units', []):
                # Check if unit belongs to this page
                if unit['id'].startswith(f"p{page_num}_"):
                    units.append(unit)
        
        return units
    
    def _get_page_tables(self, skeleton: Dict, page_num: int) -> List[Dict]:
        """Get tables for specific page"""
        
        tables = []
        for table in skeleton.get('tables', []):
            if table.get('page') == page_num:
                tables.append(table)
        
        return tables
    
    def _get_page_formulas(self, skeleton: Dict, page_num: int) -> List[Dict]:
        """Get formulas for specific page"""
        
        formulas = []
        for formula in skeleton.get('formulas', []):
            if formula.get('page') == page_num:
                formulas.append(formula)
        
        return formulas
    
    def _get_page_form_fields(self, skeleton: Dict, page_num: int) -> List[Dict]:
        """Get form fields for specific page"""
        
        fields = []
        for field in skeleton.get('form_fields', []):
            if field.get('page') == page_num:
                fields.append(field)
        
        return fields
    
    def _get_table_cell_translation(self, translated_content: Dict,
                                   table_id: str, row: int, col: int) -> Optional[str]:
        """Get translation for specific table cell"""
        
        unit_id = f"{table_id}_r{row}_c{col}"
        
        for file_data in translated_content.get('files', []):
            for unit in file_data.get('units', []):
                if unit['id'] == unit_id:
                    return unit.get('target')
        
        return None
    
    def _get_form_field_translation(self, translated_content: Dict,
                                   field_id: str) -> Optional[str]:
        """Get translation for form field label"""
        
        for file_data in translated_content.get('files', []):
            for unit in file_data.get('units', []):
                if unit['id'] == field_id:
                    return unit.get('target')
        
        return None
    
    def _get_font_name(self, font: str) -> str:
        """Map font name to PyMuPDF font"""
        
        # Check registered fonts
        if font in self.registered_fonts:
            return self.registered_fonts[font]
        
        # Map to standard fonts
        font_mapping = {
            'Arial': 'helv',
            'Helvetica': 'helv',
            'Times': 'times',
            'Times New Roman': 'times',
            'Courier': 'cour'
        }
        
        return font_mapping.get(font, 'helv')
    
    def _get_bold_variant(self, fontname: str) -> str:
        """Get bold variant of font"""
        
        if fontname == 'helv':
            return 'helvb'
        elif fontname == 'times':
            return 'timesb'
        elif fontname == 'cour':
            return 'courb'
        
        return fontname
    
    def _get_italic_variant(self, fontname: str) -> str:
        """Get italic variant of font"""
        
        if fontname == 'helv':
            return 'helvi'
        elif fontname == 'times':
            return 'timesi'
        elif fontname == 'cour':
            return 'couri'
        
        return fontname
    
    def _parse_color(self, color_str: str) -> Tuple[float, float, float]:
        """Parse color string to RGB tuple"""
        
        if color_str.startswith('#'):
            # Hex color
            hex_color = color_str[1:]
            r = int(hex_color[0:2], 16) / 255
            g = int(hex_color[2:4], 16) / 255
            b = int(hex_color[4:6], 16) / 255
            return (r, g, b)
        
        return (0, 0, 0)  # Default black
```

## Usage Example

```python
# Initialize reconstructor
reconstructor = PDFReconstructor()

# Reconstruct PDF
success = reconstructor.reconstruct_pdf(
    original_pdf='input.pdf',
    translated_content={
        'files': [{
            'units': [
                {
                    'id': 'p0_u1',
                    'source': 'Hello',
                    'target': '你好',
                    'metadata': {
                        'position': {'x': 100, 'y': 100, 'width': 50, 'height': 20},
                        'style': {'font': 'Arial', 'size': 12}
                    }
                }
            ]
        }],
        'skeleton': {
            'fonts': [...],
            'margins': {...}
        }
    },
    output_path='output.pdf'
)

if success:
    print("PDF reconstructed successfully")
```
