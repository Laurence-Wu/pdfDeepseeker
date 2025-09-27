# XLIFF Generator - Complete Implementation

## Overview
Generate XLIFF 2.1 documents with complete PDF metadata preservation.

## Implementation

```python
from lxml import etree
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import base64
import uuid

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
                    extracted_content: Dict,
                    source_lang: str,
                    target_lang: str,
                    document_metadata: Optional[Dict] = None) -> str:
        """
        Create complete XLIFF document.
        
        Args:
            extracted_content: Extracted PDF content
            source_lang: Source language code
            target_lang: Target language code  
            document_metadata: Additional metadata
            
        Returns:
            XLIFF XML string
        """
        # Create root element
        xliff = self._create_root(source_lang, target_lang)
        
        # Create file element
        file_elem = self._create_file_element(
            extracted_content.get('source_file', 'document.pdf'),
            document_metadata
        )
        
        # Add skeleton with complete PDF structure
        skeleton = self._create_skeleton(extracted_content)
        file_elem.append(skeleton)
        
        # Process pages
        for page_num, page_content in enumerate(extracted_content.get('pages', [])):
            # Create group for page
            page_group = self._create_page_group(page_num, page_content)
            
            # Add units for each text element
            units = self._create_units_from_page(page_content, page_num)
            for unit in units:
                page_group.append(unit)
            
            file_elem.append(page_group)
        
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
    
    def _create_skeleton(self, extracted_content: Dict) -> etree.Element:
        """Create skeleton with PDF structure"""
        
        skeleton = etree.Element(f"{{{self.XLIFF_NS}}}skeleton")
        
        # Build comprehensive PDF skeleton
        pdf_structure = {
            'document': {
                'pages': extracted_content.get('total_pages', 1),
                'size': extracted_content.get('page_size'),
                'orientation': extracted_content.get('orientation', 'portrait')
            },
            'margins': extracted_content.get('margins'),
            'fonts': self._prepare_font_data(extracted_content.get('fonts', {})),
            'layout': extracted_content.get('layout'),
            'edge_cases': extracted_content.get('edge_cases', []),
            'watermarks': extracted_content.get('watermarks', []),
            'metadata': extracted_content.get('metadata', {})
        }
        
        # Encode as CDATA
        skeleton_data = json.dumps(pdf_structure, indent=2)
        skeleton.text = etree.CDATA(skeleton_data)
        
        return skeleton
    
    def _create_page_group(self, page_num: int, page_content: Dict) -> etree.Element:
        """Create group element for page"""
        
        group = etree.Element(f"{{{self.XLIFF_NS}}}group")
        group.set("id", f"page_{page_num + 1}")
        group.set("name", f"Page {page_num + 1}")
        
        # Add page metadata
        metadata = etree.SubElement(group, f"{{{self.PDF_NS}}}metadata")
        
        # Page dimensions
        if page_content.get('dimensions'):
            dims = page_content['dimensions']
            metadata.set("width", str(dims.get('width', 0)))
            metadata.set("height", str(dims.get('height', 0)))
        
        # Page rotation
        if page_content.get('rotation'):
            metadata.set("rotation", str(page_content['rotation']))
        
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
        pos_elem = etree.SubElement(metadata, f"{{{self.PDF_NS}}}position")
        pos_elem.set("x", str(bbox.get('x', 0)))
        pos_elem.set("y", str(bbox.get('y', 0)))
        pos_elem.set("width", str(bbox.get('width', 0)))
        pos_elem.set("height", str(bbox.get('height', 0)))
        
        # Style information
        style = block.get('style', {})
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
    
    def _prepare_font_data(self, fonts: Dict) -> List[Dict]:
        """Prepare font data for skeleton"""
        
        font_list = []
        
        for font_id, font_info in fonts.get('embedded_fonts', {}).items():
            font_data = {
                'id': font_id,
                'name': font_info.get('name'),
                'type': font_info.get('type'),
                'embedded': True
            }
            
            # Include base64 encoded font data if needed
            if self.config.get('embed_fonts', False) and font_info.get('data'):
                font_data['data'] = base64.b64encode(font_info['data']).decode('utf-8')
            
            font_list.append(font_data)
        
        return font_list
    
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
                file_data['skeleton'] = json.loads(skeleton_elem.text)
            
            # Extract units
            for unit_elem in file_elem.findall(f'.//{{{xliff_ns}}}unit'):
                unit = self._parse_unit(unit_elem, xliff_ns, pdf_ns)
                file_data['units'].append(unit)
            
            result['files'].append(file_data)
        
        return result
    
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
        
        # Extract text
        segment = unit_elem.find(f'.//{{{xliff_ns}}}segment')
        if segment is not None:
            source = segment.find(f'.//{{{xliff_ns}}}source')
            target = segment.find(f'.//{{{xliff_ns}}}target')
            
            unit['source'] = source.text if source is not None else ''
            unit['target'] = target.text if target is not None else ''
        
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
            if not root.findall('.//{urn:oasis:names:tc:xliff:document:2.1}file'):
                errors.append("No file elements found")
            
        except etree.XMLSyntaxError as e:
            errors.append(f"XML syntax error: {e}")
        
        return len(errors) == 0, errors
```

## Usage Example

```python
# Initialize generator
generator = XLIFFGenerator()

# Create XLIFF from extraction
xliff_content = generator.create_xliff(
    extracted_content={
        'source_file': 'document.pdf',
        'pages': [
            {
                'text_blocks': [
                    {
                        'text': 'Hello World',
                        'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 20},
                        'style': {'font': 'Arial', 'size': 12}
                    }
                ],
                'dimensions': {'width': 612, 'height': 792}
            }
        ],
        'fonts': {...},
        'margins': {...}
    },
    source_lang='en',
    target_lang='zh'
)

# Parse XLIFF back
parsed = generator.parse_xliff(xliff_content)

# Validate
validator = XLIFFValidator()
is_valid, errors = validator.validate(xliff_content)
```
