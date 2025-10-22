"""
XLIFF Generator - Complete Implementation per Instruction 10
Creates XLIFF 2.1 documents with complete PDF metadata preservation.
"""

from lxml import etree
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import base64
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class XLIFFUnit:
    """XLIFF translation unit"""
    id: str
    source: str
    target: Optional[str] = None
    metadata: Optional[Dict] = None
    translate: bool = True
    preserve_space: bool = False
    max_length: Optional[int] = None
    notes: Optional[List[str]] = None


class XLIFFGenerator:
    """
    Generate XLIFF 2.1 documents for PDF translation.
    Preserves complete layout and formatting metadata.
    """

    XLIFF_NS = "urn:oasis:names:tc:xliff:document:2.1"
    PDF_NS = "urn:custom:pdf:metadata:1.0"

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.nsmap = {
            None: self.XLIFF_NS,
            'pdf': self.PDF_NS,
            'its': 'http://www.w3.org/2005/11/its',
            'xliff': self.XLIFF_NS
        }

    def create_xliff(self,
                    content: Dict,
                    source_lang: str,
                    target_lang: str,
                    document_metadata: Optional[Dict] = None) -> str:
        """
        Create complete XLIFF document.

        Args:
            content: Extracted PDF content (with 'pages' or 'translation_units')
            source_lang: Source language code
            target_lang: Target language code
            document_metadata: Additional metadata

        Returns:
            XLIFF XML string
        """
        logger.info(f"Creating XLIFF document: {source_lang} → {target_lang}")

        # Create root element
        xliff = self._create_root(source_lang, target_lang)

        # Create file element
        file_elem = self._create_file_element(
            content.get('source_file', 'document.pdf'),
            document_metadata
        )

        # Add skeleton with complete PDF structure
        skeleton = self._create_skeleton(content, document_metadata)
        file_elem.append(skeleton)

        # Process pages or direct translation units
        if 'pages' in content:
            # Page-based structure
            for page_num, page_content in enumerate(content['pages']):
                page_group = self._create_page_group(page_num, page_content)
                units = self._create_units_from_page(page_content, page_num)
                for unit in units:
                    page_group.append(unit)
                file_elem.append(page_group)
        elif 'translation_units' in content:
            # Direct translation units (used by demonstration script)
            for unit_data in content['translation_units']:
                unit = self._create_unit_from_data(unit_data)
                file_elem.append(unit)

        xliff.append(file_elem)

        # Convert to string
        return self._serialize(xliff)

    def _create_root(self, source_lang: str, target_lang: str) -> etree.Element:
        """Create XLIFF root element"""

        xliff = etree.Element(
            f"{{{self.XLIFF_NS}}}xliff",
            version="2.1",
            srcLang=source_lang,
            trgLang=target_lang,
            nsmap=self.nsmap
        )

        return xliff

    def _create_file_element(self, filename: str, metadata: Optional[Dict]) -> etree.Element:
        """Create file element"""

        file_elem = etree.Element(f"{{{self.XLIFF_NS}}}file")
        file_elem.set("id", str(uuid.uuid4()))
        file_elem.set("original", filename)

        # Add metadata
        if metadata:
            file_elem.set("datatype", metadata.get('datatype', 'pdf'))

            # Add notes
            if metadata.get('notes'):
                notes = etree.SubElement(file_elem, f"{{{self.XLIFF_NS}}}notes")
                for note in metadata['notes']:
                    note_elem = etree.SubElement(notes, f"{{{self.XLIFF_NS}}}note")
                    note_elem.text = note

        return file_elem

    def _create_skeleton(self, content: Dict, document_metadata: Optional[Dict]) -> etree.Element:
        """Create skeleton with PDF structure"""

        skeleton = etree.Element(f"{{{self.XLIFF_NS}}}skeleton")

        # Build comprehensive PDF skeleton
        pdf_structure = {
            'document': {
                'pages': len(content.get('pages', [])),
                'source_file': content.get('source_file', 'unknown')
            },
            'fonts': content.get('fonts', {}),
            'formulas': content.get('formulas', []),
            'tables': content.get('tables', []),
            'watermarks': content.get('watermarks', []),
            'edge_cases': content.get('edge_cases', []),
            'metadata': document_metadata or {}
        }

        # Encode as CDATA
        skeleton_data = json.dumps(pdf_structure, indent=2, ensure_ascii=False)
        skeleton.text = etree.CDATA(skeleton_data)

        return skeleton

    def _create_page_group(self, page_num: int, page_content: Dict) -> etree.Element:
        """Create group element for page"""

        group = etree.Element(f"{{{self.XLIFF_NS}}}group")
        group.set("id", f"page_{page_num + 1}")
        group.set("name", f"Page {page_num + 1}")

        # Add page metadata
        if page_content.get('dimensions'):
            metadata = etree.SubElement(group, f"{{{self.PDF_NS}}}metadata")
            dims = page_content['dimensions']
            metadata.set("width", str(dims.get('width', 0)))
            metadata.set("height", str(dims.get('height', 0)))

        # Page rotation
        if page_content.get('rotation'):
            group.set("rotation", str(page_content['rotation']))

        return group

    def _create_units_from_page(self, page_content: Dict, page_num: int) -> List[etree.Element]:
        """Create translation units from page content"""

        units = []
        unit_counter = 1

        # Process text blocks
        for block in page_content.get('text_blocks', []):
            unit = self._create_text_unit(block, f"p{page_num}_u{unit_counter}")
            units.append(unit)
            unit_counter += 1

        # Process tables
        for table in page_content.get('tables', []):
            table_units = self._create_table_units(table, f"p{page_num}_t{unit_counter}")
            units.extend(table_units)
            unit_counter += len(table_units)

        # Process form fields
        for field in page_content.get('form_fields', []):
            if field.get('label'):
                unit = self._create_form_field_unit(field, f"p{page_num}_f{unit_counter}")
                units.append(unit)
                unit_counter += 1

        # Process headers/footers
        for header in page_content.get('headers', []):
            unit = self._create_header_footer_unit(header, f"p{page_num}_h{unit_counter}", 'header')
            units.append(unit)
            unit_counter += 1

        return units

    def _create_unit_from_data(self, unit_data: Dict) -> etree.Element:
        """Create unit from direct translation unit data"""

        unit = etree.Element(f"{{{self.XLIFF_NS}}}unit")
        unit.set("id", unit_data.get('id', str(uuid.uuid4())))

        # Add metadata if present
        if unit_data.get('metadata'):
            metadata = etree.SubElement(unit, f"{{{self.PDF_NS}}}metadata")
            meta_dict = unit_data['metadata']

            # Store metadata as attributes and child elements
            for key, value in meta_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata.set(key, str(value))
                elif isinstance(value, dict):
                    # For complex metadata like position, style
                    meta_child = etree.SubElement(metadata, f"{{{self.PDF_NS}}}{key}")
                    for k, v in value.items():
                        meta_child.set(k, str(v))
                elif isinstance(value, list):
                    # For list metadata like bbox [x0, y0, x1, y1]
                    meta_child = etree.SubElement(metadata, f"{{{self.PDF_NS}}}{key}")
                    meta_child.text = ','.join(str(v) for v in value)

        # Create segment
        segment = etree.SubElement(unit, f"{{{self.XLIFF_NS}}}segment")

        # Source text
        source = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}source")
        source.text = unit_data.get('source', '')

        # Target text
        target = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}target")
        if unit_data.get('target'):
            target.text = unit_data['target']

        return unit

    def _create_text_unit(self, block: Dict, unit_id: str) -> etree.Element:
        """Create unit for text block"""

        unit = etree.Element(f"{{{self.XLIFF_NS}}}unit")
        unit.set("id", unit_id)

        # Check if should translate
        if not block.get('translate', True):
            unit.set("translate", "no")

        # Add metadata
        metadata = etree.SubElement(unit, f"{{{self.PDF_NS}}}metadata")

        # Position information
        bbox = block.get('bbox', {})
        if bbox:
            pos_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}position")
            if isinstance(bbox, dict):
                pos_elem.set("x", str(bbox.get('x', 0)))
                pos_elem.set("y", str(bbox.get('y', 0)))
                pos_elem.set("width", str(bbox.get('width', 0)))
                pos_elem.set("height", str(bbox.get('height', 0)))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                pos_elem.set("x", str(bbox[0]))
                pos_elem.set("y", str(bbox[1]))
                pos_elem.set("width", str(bbox[2] - bbox[0]))
                pos_elem.set("height", str(bbox[3] - bbox[1]))

        # Style information
        style = block.get('style', {})
        if style:
            style_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}style")
            style_elem.set("font", style.get('font', 'default'))
            style_elem.set("size", str(style.get('size', 12)))
            style_elem.set("weight", style.get('weight', 'normal'))
            style_elem.set("italic", str(style.get('italic', False)).lower())
            style_elem.set("color", style.get('color', '#000000'))

        # Length constraint
        if block.get('max_length'):
            constraint_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}constraint")
            constraint_elem.set("maxLength", str(block['max_length']))

        # Create segment
        segment = etree.SubElement(unit, f"{{{self.XLIFF_NS}}}segment")

        # Source text
        source = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}source")
        source.text = block.get('text', '')

        # Preserve spaces if needed
        if block.get('preserve_space', False):
            source.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

        # Target placeholder
        target = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}target")
        if block.get('translation'):
            target.text = block['translation']

        return unit

    def _create_table_units(self, table: Dict, base_id: str) -> List[etree.Element]:
        """Create units for table cells"""

        units = []

        for row_idx, row in enumerate(table.get('rows', [])):
            for col_idx, cell in enumerate(row):
                if not cell.get('text'):
                    continue

                unit_id = f"{base_id}_r{row_idx}_c{col_idx}"
                unit = etree.Element(f"{{{self.XLIFF_NS}}}unit")
                unit.set("id", unit_id)

                # Don't translate headers
                if row_idx == 0 or col_idx == 0:
                    unit.set("translate", "no")

                # Add table metadata
                metadata = etree.SubElement(unit, f"{{{self.PDF_NS}}}metadata")
                table_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}table")
                table_elem.set("row", str(row_idx))
                table_elem.set("col", str(col_idx))
                table_elem.set("isHeader", str(row_idx == 0).lower())

                # Add cell content
                segment = etree.SubElement(unit, f"{{{self.XLIFF_NS}}}segment")
                source = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}source")
                source.text = cell['text']
                target = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}target")

                units.append(unit)

        return units

    def _create_form_field_unit(self, field: Dict, unit_id: str) -> etree.Element:
        """Create unit for form field label"""

        unit = etree.Element(f"{{{self.XLIFF_NS}}}unit")
        unit.set("id", unit_id)

        # Add form field metadata
        metadata = etree.SubElement(unit, f"{{{self.PDF_NS}}}metadata")
        form_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}formField")
        form_elem.set("type", field.get('type', 'text'))
        form_elem.set("name", field.get('name', ''))
        form_elem.set("required", str(field.get('required', False)).lower())

        # Add label for translation
        segment = etree.SubElement(unit, f"{{{self.XLIFF_NS}}}segment")
        source = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}source")
        source.text = field.get('label', '')
        target = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}target")

        return unit

    def _create_header_footer_unit(self, element: Dict, unit_id: str, type: str) -> etree.Element:
        """Create unit for header or footer"""

        unit = etree.Element(f"{{{self.XLIFF_NS}}}unit")
        unit.set("id", unit_id)

        # Add metadata
        metadata = etree.SubElement(unit, f"{{{self.PDF_NS}}}metadata")
        hf_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}{type}")
        hf_elem.set("position", element.get('position', 'center'))

        # Add content
        segment = etree.SubElement(unit, f"{{{self.XLIFF_NS}}}segment")
        source = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}source")
        source.text = element.get('text', '')
        target = etree.SubElement(segment, f"{{{self.XLIFF_NS}}}target")

        return unit

    def _serialize(self, xliff_element: etree.Element) -> str:
        """Serialize XLIFF to string"""

        return etree.tostring(
            xliff_element,
            pretty_print=True,
            xml_declaration=True,
            encoding='UTF-8'
        ).decode('utf-8')

    def parse_xliff(self, xliff_content: str) -> Dict:
        """
        Parse XLIFF back to structure.

        Args:
            xliff_content: XLIFF XML string

        Returns:
            Parsed structure
        """
        try:
            root = etree.fromstring(xliff_content.encode('utf-8'))

            # Extract namespace map
            nsmap = root.nsmap
            xliff_ns = nsmap.get(None, self.XLIFF_NS)
            pdf_ns = nsmap.get('pdf', self.PDF_NS)

            result = {
                'source_lang': root.get('srcLang'),
                'target_lang': root.get('trgLang'),
                'files': []
            }

            # Process files
            for file_elem in root.findall(f'.//{{{xliff_ns}}}file'):
                file_data = {
                    'id': file_elem.get('id'),
                    'original': file_elem.get('original'),
                    'units': [],
                    'skeleton': None
                }

                # Extract skeleton
                skeleton_elem = file_elem.find(f'.//{{{xliff_ns}}}skeleton')
                if skeleton_elem is not None and skeleton_elem.text:
                    try:
                        file_data['skeleton'] = json.loads(skeleton_elem.text)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse skeleton JSON")
                        file_data['skeleton'] = {}

                # Extract units
                for unit_elem in file_elem.findall(f'.//{{{xliff_ns}}}unit'):
                    unit = self._parse_unit(unit_elem, xliff_ns, pdf_ns)
                    file_data['units'].append(unit)

                result['files'].append(file_data)

            return result

        except Exception as e:
            logger.error(f"Failed to parse XLIFF: {e}")
            return {'source_lang': 'en', 'target_lang': 'zh', 'files': []}

    def _parse_unit(self, unit_elem: etree.Element, xliff_ns: str, pdf_ns: str) -> Dict:
        """Parse single unit"""

        unit = {
            'id': unit_elem.get('id'),
            'translate': unit_elem.get('translate', 'yes') == 'yes',
            'metadata': {}
        }

        # Extract metadata
        metadata_elem = unit_elem.find(f'.//{{{pdf_ns}}}metadata')
        if metadata_elem is not None:
            # Extract simple attributes
            for key, value in metadata_elem.attrib.items():
                try:
                    # Try to convert to numeric types if possible
                    if '.' in value:
                        unit['metadata'][key] = float(value)
                    else:
                        unit['metadata'][key] = int(value)
                except (ValueError, TypeError):
                    unit['metadata'][key] = value

            # Position
            pos_elem = metadata_elem.find(f'.//{{{pdf_ns}}}position')
            if pos_elem is not None:
                unit['metadata']['position'] = {
                    'x': float(pos_elem.get('x', 0)),
                    'y': float(pos_elem.get('y', 0)),
                    'width': float(pos_elem.get('width', 0)),
                    'height': float(pos_elem.get('height', 0))
                }

            # Style
            style_elem = metadata_elem.find(f'.//{{{pdf_ns}}}style')
            if style_elem is not None:
                unit['metadata']['style'] = {
                    'font': style_elem.get('font'),
                    'size': float(style_elem.get('size', 12)),
                    'weight': style_elem.get('weight'),
                    'color': style_elem.get('color')
                }

            # Bbox (list format)
            bbox_elem = metadata_elem.find(f'.//{{{pdf_ns}}}bbox')
            if bbox_elem is not None and bbox_elem.text:
                bbox_values = [float(v.strip()) for v in bbox_elem.text.split(',')]
                unit['metadata']['bbox'] = bbox_values

        # Extract text
        segment = unit_elem.find(f'.//{{{xliff_ns}}}segment')
        if segment is not None:
            source = segment.find(f'.//{{{xliff_ns}}}source')
            target = segment.find(f'.//{{{xliff_ns}}}target')

            unit['source'] = source.text if source is not None and source.text else ''
            unit['target'] = target.text if target is not None and target.text else ''

        return unit


class XLIFFValidator:
    """Validate XLIFF documents"""

    def validate(self, xliff_content: str) -> Tuple[bool, List[str]]:
        """
        Validate XLIFF document.

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            # Parse XML
            root = etree.fromstring(xliff_content.encode('utf-8'))

            # Check version
            version = root.get('version')
            if version != '2.1':
                errors.append(f"Invalid XLIFF version: {version}")

            # Check required attributes
            if not root.get('srcLang'):
                errors.append("Missing source language")
            if not root.get('trgLang'):
                errors.append("Missing target language")

            # Validate structure
            nsmap = root.nsmap
            xliff_ns = nsmap.get(None, 'urn:oasis:names:tc:xliff:document:2.1')
            if not root.findall(f'.//{{{xliff_ns}}}file'):
                errors.append("No file elements found")

        except etree.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {e}")

        return len(errors) == 0, errors
