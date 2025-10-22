"""
PDF Reconstructor - Complete Implementation per Instruction 11
Reconstruct PDF with translations while preserving exact layout, fonts, and formatting.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
import base64
import logging
from src.utils.config_loader import load_translation_config

logger = logging.getLogger(__name__)


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

        # Load translation config for rendering settings
        self.translation_config = load_translation_config()

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
            logger.info(f"Reconstructing PDF: {original_pdf} → {output_path}")

            # Open original for reference
            original_doc = fitz.open(original_pdf)

            # Create new document
            new_doc = fitz.open()

            # Get skeleton data from first file
            skeleton = {}
            if translated_content.get('files'):
                skeleton = translated_content['files'][0].get('skeleton', {})

            # Register fonts
            if self.preserve_fonts:
                self._register_fonts(skeleton.get('fonts', []))

            # Process each page
            for page_num in range(original_doc.page_count):
                logger.info(f"Processing page {page_num + 1}/{original_doc.page_count}")

                # Create new page with original dimensions
                original_page = original_doc[page_num]
                page_rect = original_page.rect
                new_page = new_doc.new_page(
                    width=page_rect.width,
                    height=page_rect.height
                )

                # Add translated text (primary content)
                self._add_translated_text(
                    new_page,
                    translated_content,
                    page_num,
                    skeleton
                )

                # Copy images and graphics (but not text)
                self._copy_non_text_elements(original_page, new_page)

                # Handle special elements (formulas, watermarks, etc.)
                self._handle_special_elements(
                    original_page,
                    new_page,
                    translated_content,
                    page_num,
                    skeleton
                )

            # Save document
            new_doc.save(output_path,
                        garbage=4,
                        deflate=True,
                        clean=True)

            # Cleanup
            original_doc.close()
            new_doc.close()

            logger.info(f"PDF reconstruction completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _register_fonts(self, fonts):
        """Register embedded fonts for use"""

        # Handle different font data formats
        font_list = []
        if isinstance(fonts, list):
            font_list = fonts
        elif isinstance(fonts, dict):
            # Handle format with 'embedded_fonts' key
            if 'embedded_fonts' in fonts:
                embedded = fonts['embedded_fonts']
                if isinstance(embedded, dict):
                    font_list = list(embedded.values())
                else:
                    font_list = embedded
            else:
                font_list = list(fonts.values()) if fonts else []

        for font_data in font_list:
            # Skip if not a dict (might be a string after base64 encoding)
            if not isinstance(font_data, dict):
                continue

            if font_data.get('embedded') and font_data.get('data'):
                try:
                    # Decode font data
                    font_bytes = base64.b64decode(font_data['data'])

                    # Save to temporary file
                    import tempfile
                    import os
                    temp_dir = tempfile.gettempdir()
                    font_path = os.path.join(temp_dir, f"{font_data['name']}.ttf")

                    with open(font_path, 'wb') as f:
                        f.write(font_bytes)

                    # Register with PyMuPDF
                    font_name = font_data['name']
                    self.registered_fonts[font_name] = font_path

                    logger.info(f"Registered font: {font_name}")

                except Exception as e:
                    logger.warning(f"Failed to register font {font_data.get('name', 'unknown')}: {e}")

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
                img_rects = original_page.get_image_rects(xref)
                if img_rects:
                    img_rect = img_rects[0]

                    # Insert into new page
                    img_data = pix.tobytes("png")
                    new_page.insert_image(
                        img_rect,
                        stream=img_data,
                        keep_proportion=True
                    )

            except Exception as e:
                logger.warning(f"Failed to copy image {img_index}: {e}")

        # Copy vector graphics
        self._copy_vector_graphics(original_page, new_page)

    def _copy_vector_graphics(self, original_page, new_page):
        """Copy vector graphics and shapes"""

        # Get drawing commands
        try:
            drawings = original_page.get_drawings()

            for drawing in drawings:
                try:
                    draw_type = drawing.get('type', '')

                    if draw_type == 'l':  # Line
                        items = drawing.get('items', [])
                        if len(items) >= 2:
                            new_page.draw_line(
                                fitz.Point(items[0][1], items[0][2]),
                                fitz.Point(items[1][1], items[1][2]),
                                color=drawing.get('color', (0, 0, 0)),
                                width=drawing.get('width', 1)
                            )
                    elif draw_type == 're':  # Rectangle
                        rect_data = drawing.get('rect')
                        if rect_data:
                            new_page.draw_rect(
                                rect_data,
                                color=drawing.get('color', (0, 0, 0)),
                                fill=drawing.get('fill'),
                                width=drawing.get('width', 1)
                            )

                except Exception as e:
                    logger.warning(f"Failed to copy drawing: {e}")

        except Exception as e:
            logger.warning(f"Failed to get drawings: {e}")

    def _add_translated_text(self, page, translated_content: Dict,
                           page_num: int, skeleton: Dict):
        """Add translated text to page"""

        # Get units for this page
        page_units = self._get_page_units(translated_content, page_num)

        # Get table bboxes to exclude overlapping text blocks
        tables = self._get_page_tables(skeleton, page_num)
        table_rects = []
        for table in tables:
            bbox = table.get('bbox', {})
            if bbox:
                table_rect = fitz.Rect(
                    bbox.get('x', 0),
                    bbox.get('y', 0),
                    bbox.get('x', 0) + bbox.get('width', 0),
                    bbox.get('y', 0) + bbox.get('height', 0)
                )
                table_rects.append(table_rect)

        logger.info(f"Adding {len(page_units)} translated text units to page {page_num + 1} (excluding {len(table_rects)} table areas)")

        for unit in page_units:
            if not unit.get('target'):  # Skip if no translation
                continue

            # Get metadata
            metadata = unit.get('metadata', {})
            position = metadata.get('position', {})
            style = metadata.get('style', {})

            # Check if this text block overlaps with any table
            x = position.get('x', 0)
            y = position.get('y', 0)
            width = position.get('width', 100)
            height = position.get('height', 20)
            text_rect = fitz.Rect(x, y, x + width, y + height)

            # Skip if overlaps with table
            skip_block = False
            for table_rect in table_rects:
                if table_rect.intersects(text_rect):
                    logger.debug(f"Skipping text block {unit['id']} (overlaps with table)")
                    skip_block = True
                    break

            if skip_block:
                continue

            # Prepare text insertion
            text = unit['target']

            # Calculate position - create a rectangle for the text
            x = position.get('x', 0)
            y = position.get('y', 0)
            width = position.get('width', 100)
            height = position.get('height', 20)

            # Expand bbox significantly to ensure text fits
            # Tight bboxes from extraction don't leave room for re-rendering
            # Load padding values from config
            bbox_padding = self.translation_config['rendering']['bbox_padding']
            padding_x = width * bbox_padding['horizontal']
            padding_y = height * bbox_padding['vertical']
            left_offset = bbox_padding['left_offset']
            right_offset = bbox_padding['right_offset']

            # Create text box rectangle with generous padding
            rect = fitz.Rect(
                max(0, x - padding_x * left_offset),
                max(0, y - padding_y/2),
                x + width + padding_x * right_offset,
                y + height + padding_y/2
            )

            # Prepare font
            fontname = self._get_font_name(style.get('font', 'helv'))
            fontsize = style.get('size', 12)
            color = self._parse_color(style.get('color', '#000000'))

            # Check if text contains CJK characters
            has_cjk = self._has_cjk_characters(text)

            # Handle special formatting
            weight = style.get('weight', 'normal')
            italic = style.get('italic', 'false')

            if weight == 'bold':
                fontname = self._get_bold_variant(fontname)
            if italic == 'true':
                fontname = self._get_italic_variant(fontname)

            # Insert text using appropriate method based on content
            try:
                # For CJK text, use htmlbox which supports full Unicode including math symbols
                if has_cjk:
                    # Convert color tuple to RGB values
                    r = int(color[0] * 255)
                    g = int(color[1] * 255)
                    b = int(color[2] * 255)

                    # Create HTML with proper font size
                    # Use a larger rect to prevent auto-scaling
                    html_rect = fitz.Rect(x, y, x + width * 2, y + height * 2)

                    html_content = f'<p style="font-size:{fontsize}px;color:rgb({r},{g},{b});margin:0;padding:0;line-height:1.2">{text}</p>'

                    rc = page.insert_htmlbox(
                        html_rect,
                        html_content,
                        css='body { margin: 0; padding: 0; }',
                        archive=None
                    )
                else:
                    # For non-CJK text, use standard textbox
                    rc = page.insert_textbox(
                        rect,
                        text,
                        fontname=fontname,
                        fontsize=fontsize,
                        color=color,
                        align=fitz.TEXT_ALIGN_LEFT,
                        render_mode=0,
                        rotate=metadata.get('rotation', 0)
                    )

                # Note: rc > 0 indicates some text may not have fit, but it's been inserted
                # Only handle overflow if rc is very large (significant overflow)
                if rc > len(text) * 0.5:  # More than 50% of text didn't fit
                    logger.warning(f"Significant text overflow for unit {unit['id']}: {rc:.1f} chars didn't fit")
                    # Note: The text has already been inserted (partially), don't re-insert

            except Exception as e:
                logger.error(f"Failed to insert text for unit {unit['id']}: {e}")

    def _handle_special_elements(self, original_page, new_page,
                                translated_content: Dict, page_num: int,
                                skeleton: Dict):
        """Handle special elements like tables, formulas, etc."""

        # Handle tables with translated content
        tables = self._get_page_tables(skeleton, page_num)
        for table in tables:
            self._render_translated_table(new_page, table)

        # Handle formulas (preserve as-is from original)
        formulas = self._get_page_formulas(skeleton, page_num)
        for formula in formulas:
            self._preserve_formula(original_page, new_page, formula)

        # Handle watermarks
        watermarks = skeleton.get('watermarks', [])
        for watermark in watermarks:
            if watermark.get('page') == page_num:
                self._apply_watermark(new_page, watermark)

    def _preserve_formula(self, original_page, new_page, formula: Dict):
        """Insert formula image into reconstructed PDF

        Formulas are preserved as images extracted by the formula extractor,
        which captures the visual representation of the formula from the original PDF.
        """

        try:
            # Get formula bbox and image data
            bbox = formula.get('bbox', {})
            image_data_b64 = formula.get('image_data')

            if not image_data_b64:
                logger.warning(f"Formula on page {formula.get('page')} has no image data")
                return

            # Decode base64 image
            image_bytes = base64.b64decode(image_data_b64)

            # Create rect for formula placement
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)

            rect = fitz.Rect(x, y, x + width, y + height)

            # Insert formula image
            new_page.insert_image(
                rect,
                stream=image_bytes,
                keep_proportion=True,
                overlay=True
            )

            logger.debug(f"Inserted formula image at {rect}")

        except Exception as e:
            logger.warning(f"Failed to preserve formula: {e}")

    def _apply_watermark(self, page, watermark: Dict):
        """Apply watermark to page"""

        try:
            watermark_type = watermark.get('type', 'visible_text')

            if watermark_type == 'visible_text':
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

                logger.info(f"Applied text watermark: {watermark.get('text')}")

            elif watermark_type == 'image' and watermark.get('data'):
                # Add image watermark
                img_data = base64.b64decode(watermark['data'])
                page.insert_image(
                    page.rect,
                    stream=img_data,
                    overlay=True,
                    keep_proportion=True
                )

                logger.info("Applied image watermark")

        except Exception as e:
            logger.warning(f"Failed to apply watermark: {e}")

    def _handle_text_overflow_textbox(self, page, rect: fitz.Rect, text: str,
                                      fontname: str, fontsize: float, color: Tuple):
        """Handle text that doesn't fit in textbox"""

        # Try progressively smaller font sizes
        min_size = 6
        current_size = fontsize

        while current_size > min_size:
            current_size -= 0.5

            # Try again with smaller font
            try:
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontname=fontname,
                    fontsize=current_size,
                    color=color,
                    align=fitz.TEXT_ALIGN_LEFT
                )

                if rc <= 0:  # Success (no overflow)
                    logger.info(f"Text fit with reduced font size: {current_size}")
                    break
            except:
                pass

        if current_size <= min_size:
            # Last resort: truncate text
            max_chars = int(len(text) * 0.7)  # Keep 70% of text
            truncated = text[:max_chars] + "..." if len(text) > max_chars else text

            try:
                page.insert_textbox(
                    rect,
                    truncated,
                    fontname=fontname,
                    fontsize=min_size,
                    color=color,
                    align=fitz.TEXT_ALIGN_LEFT
                )
                logger.info(f"Text truncated to fit: {len(truncated)}/{len(text)} chars")
            except Exception as e:
                logger.error(f"Failed to insert truncated text: {e}")

    def _handle_text_overflow(self, page, point: fitz.Point, text: str,
                            fontname: str, fontsize: float, color: Tuple,
                            position: Dict):
        """Handle text that doesn't fit (legacy method)"""

        # Try progressively smaller font sizes
        min_size = 6
        current_size = fontsize

        while current_size > min_size:
            current_size -= 0.5

            # Try again with smaller font
            try:
                rc = page.insert_text(
                    point,
                    text,
                    fontname=fontname,
                    fontsize=current_size,
                    color=color
                )

                if rc >= 0:  # Success
                    logger.info(f"Text fit with reduced font size: {current_size}")
                    break
            except:
                pass

        if current_size <= min_size:
            # Last resort: truncate text
            truncated = self._truncate_to_fit(
                text, position.get('width', 100),
                fontname, min_size
            )

            try:
                page.insert_text(
                    point,
                    truncated,
                    fontname=fontname,
                    fontsize=min_size,
                    color=color
                )
                logger.info(f"Text truncated to fit: {len(truncated)}/{len(text)} chars")
            except Exception as e:
                logger.error(f"Failed to insert truncated text: {e}")

    def _truncate_to_fit(self, text: str, max_width: float,
                        fontname: str, fontsize: float) -> str:
        """Truncate text to fit width"""

        char_width = fontsize * 0.5
        max_chars = int(max_width / char_width)

        if len(text) > max_chars:
            return text[:max_chars-3] + '...'

        return text

    def _has_cjk_characters(self, text: str) -> bool:
        """Check if text contains CJK (Chinese, Japanese, Korean) characters"""

        for char in text:
            code_point = ord(char)
            # Check CJK Unicode ranges
            if (0x4E00 <= code_point <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= code_point <= 0x4DBF or    # CJK Extension A
                0x20000 <= code_point <= 0x2A6DF or  # CJK Extension B
                0x2A700 <= code_point <= 0x2B73F or  # CJK Extension C
                0x2B740 <= code_point <= 0x2B81F or  # CJK Extension D
                0x2B820 <= code_point <= 0x2CEAF or  # CJK Extension E
                0x3000 <= code_point <= 0x303F or    # CJK Symbols and Punctuation
                0xFF00 <= code_point <= 0xFFEF or    # Halfwidth and Fullwidth Forms
                0x3040 <= code_point <= 0x309F or    # Hiragana
                0x30A0 <= code_point <= 0x30FF or    # Katakana
                0xAC00 <= code_point <= 0xD7AF):     # Hangul Syllables
                return True

        return False

    def _get_page_units(self, translated_content: Dict, page_num: int) -> List[Dict]:
        """Get translation units for specific page"""

        units = []

        for file_data in translated_content.get('files', []):
            for unit in file_data.get('units', []):
                # Check if unit belongs to this page
                unit_id = unit.get('id', '')
                if unit_id.startswith(f"p{page_num}_"):
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

    def _render_translated_table(self, page, table: Dict):
        """Render table with translated cell contents"""

        try:
            bbox = table.get('bbox', {})
            table_x = bbox.get('x', 0)
            table_y = bbox.get('y', 0)
            table_width = bbox.get('width', 0)
            table_height = bbox.get('height', 0)

            # Get translated cells
            translated_cells = table.get('translated_cells', table.get('cells', []))

            if not translated_cells:
                logger.warning(f"Table has no cells to render")
                return

            # Draw table border
            table_rect = fitz.Rect(table_x, table_y, table_x + table_width, table_y + table_height)
            page.draw_rect(table_rect, color=(0, 0, 0), width=1)

            # Calculate grid dimensions
            rows = max(cell.get('row', 0) for cell in translated_cells) + 1
            cols = max(cell.get('col', 0) for cell in translated_cells) + 1

            cell_width = table_width / cols if cols > 0 else 0
            cell_height = table_height / rows if rows > 0 else 0

            # Draw grid lines
            for i in range(rows + 1):
                y = table_y + (i * cell_height)
                page.draw_line(
                    fitz.Point(table_x, y),
                    fitz.Point(table_x + table_width, y),
                    color=(0, 0, 0),
                    width=0.5
                )

            for j in range(cols + 1):
                x = table_x + (j * cell_width)
                page.draw_line(
                    fitz.Point(x, table_y),
                    fitz.Point(x, table_y + table_height),
                    color=(0, 0, 0),
                    width=0.5
                )

            # Render cell contents
            for cell in translated_cells:
                cell_bbox = cell.get('bbox', {})
                row = cell.get('row', 0)
                col = cell.get('col', 0)
                is_header = cell.get('is_header', False)

                # Calculate cell rect
                cell_x = table_x + (col * cell_width)
                cell_y = table_y + (row * cell_height)
                cell_rect = fitz.Rect(cell_x, cell_y, cell_x + cell_width, cell_y + cell_height)

                # Get translated text
                text = cell.get('translated_text', cell.get('text', ''))

                if text:
                    # Use smaller font for table cells
                    fontsize = 9 if not is_header else 10
                    fontname = 'helvb' if is_header else 'helv'

                    # Check if text contains CJK
                    has_cjk = self._has_cjk_characters(text)

                    # Add padding to cell rect
                    text_rect = fitz.Rect(
                        cell_x + 2,
                        cell_y + 2,
                        cell_x + cell_width - 2,
                        cell_y + cell_height - 2
                    )

                    try:
                        if has_cjk:
                            # Use HTML for CJK text
                            html_content = f'<p style="font-size:{fontsize}px;margin:0;padding:2px;line-height:1.1">{text}</p>'
                            page.insert_htmlbox(
                                text_rect,
                                html_content,
                                css='body { margin: 0; padding: 0; }'
                            )
                        else:
                            # Use textbox for non-CJK
                            page.insert_textbox(
                                text_rect,
                                text,
                                fontname=fontname,
                                fontsize=fontsize,
                                color=(0, 0, 0),
                                align=fitz.TEXT_ALIGN_LEFT
                            )
                    except Exception as e:
                        logger.warning(f"Failed to insert table cell text: {e}")

            logger.info(f"Rendered table with {len(translated_cells)} cells")

        except Exception as e:
            logger.error(f"Failed to render table: {e}")

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
            'Courier': 'cour',
            'default': 'helv'
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

        if isinstance(color_str, tuple):
            return color_str

        if isinstance(color_str, str) and color_str.startswith('#'):
            # Hex color
            hex_color = color_str[1:]
            try:
                r = int(hex_color[0:2], 16) / 255
                g = int(hex_color[2:4], 16) / 255
                b = int(hex_color[4:6], 16) / 255
                return (r, g, b)
            except:
                pass

        return (0, 0, 0)  # Default black
