"""
XLIFF Generator - Creates XLIFF 2.1 documents for translation.
Handles document structure and metadata preservation.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class XLIFFGenerator:
    """
    Generates XLIFF 2.1 documents for PDF translation.
    Preserves layout, formatting, and metadata.
    """

    def __init__(self):
        self.namespaces = {
            'xliff': 'urn:oasis:names:tc:xliff:document:2.0',
            'fs': 'urn:oasis:names:tc:xliff:fs:2.0'
        }

    async def generate_xliff(self, pdf_path: str, request: Dict[str, Any],
                           metadata: Dict[str, Any]) -> str:
        """
        Generate XLIFF document from PDF extraction data.

        Args:
            pdf_path: Path to source PDF
            request: Translation request parameters
            metadata: Extracted document metadata

        Returns:
            XLIFF document as string
        """
        logger.info(f"Generating XLIFF for {pdf_path}")

        # Create XLIFF root element
        root = ET.Element('xliff')
        root.set('version', '2.1')
        root.set('xmlns', self.namespaces['xliff'])
        root.set('xmlns:fs', self.namespaces['fs'])
        root.set('srcLang', request.get('source_lang', 'en'))
        root.set('trgLang', request.get('target_lang', 'zh'))

        # Add file element
        file_elem = ET.SubElement(root, 'file')
        file_elem.set('id', f"pdf_{uuid.uuid4().hex[:8]}")
        file_elem.set('original', pdf_path)

        # Add skeleton (layout information)
        skeleton = ET.SubElement(file_elem, 'skeleton')
        skeleton.text = self._generate_skeleton_data(metadata)

        # Add translation units
        units = self._extract_translation_units(metadata)
        for unit in units:
            unit_elem = ET.SubElement(file_elem, 'unit')
            unit_elem.set('id', unit['id'])

            # Add source content
            source_elem = ET.SubElement(unit_elem, 'segment')
            source_elem.text = unit['source']

            # Add target (empty for now)
            target_elem = ET.SubElement(unit_elem, 'segment')
            target_elem.set('state', 'initial')

            # Add notes for constraints
            if unit.get('constraints'):
                notes_elem = ET.SubElement(unit_elem, 'notes')
                for constraint in unit['constraints']:
                    note_elem = ET.SubElement(notes_elem, 'note')
                    note_elem.text = str(constraint)

        # Convert to string with proper formatting
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _generate_skeleton_data(self, metadata: Dict[str, Any]) -> str:
        """
        Generate skeleton data containing layout and formatting information.

        Args:
            metadata: Document metadata from extraction

        Returns:
            JSON string containing skeleton data
        """
        import json

        skeleton_data = {
            'layout': metadata.get('layout', {}),
            'fonts': metadata.get('fonts', {}),
            'margins': metadata.get('margins', {}),
            'formulas': metadata.get('formulas', []),
            'tables': metadata.get('tables', []),
            'watermarks': metadata.get('watermarks', []),
            'edge_cases': metadata.get('edge_cases', []),
            'document_type': metadata.get('document_type', 'general'),
            'preservation_settings': {
                'preserve_formatting': True,
                'preserve_formulas': True,
                'preserve_tables': True,
                'preserve_layout': True
            }
        }

        return json.dumps(skeleton_data, ensure_ascii=False, indent=2)

    def _extract_translation_units(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract translatable units from document metadata.

        Args:
            metadata: Document metadata

        Returns:
            List of translation units with constraints
        """
        units = []

        # Extract from layout analysis
        layout_info = metadata.get('layout', {})
        text_blocks = layout_info.get('text_blocks', [])

        for i, block in enumerate(text_blocks):
            if block.get('text', '').strip():
                # Calculate constraints based on block positioning
                constraints = self._calculate_text_constraints(block)

                unit = {
                    'id': f"unit_{i}",
                    'source': block['text'],
                    'position': block['bbox'],
                    'font_info': {
                        'name': block.get('font_name', 'Unknown'),
                        'size': block.get('font_size', 12)
                    },
                    'constraints': constraints
                }
                units.append(unit)

        return units

    def _calculate_text_constraints(self, text_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate translation constraints for a text block.

        Args:
            text_block: Text block with positioning information

        Returns:
            List of constraint dictionaries
        """
        constraints = []

        # Length constraint based on bounding box
        bbox = text_block.get('bbox', {})
        if bbox:
            width = bbox[2] - bbox[0]  # x1 - x0
            height = bbox[3] - bbox[1]  # y1 - y0

            # Estimate maximum characters based on width and font size
            font_size = text_block.get('font_size', 12)
            max_chars = int((width / font_size) * 2)  # Rough estimate

            constraints.append({
                'type': 'max_length',
                'value': max_chars,
                'description': f'Maximum {max_chars} characters to fit in bounding box'
            })

        # Font preservation constraint
        font_name = text_block.get('font_name', '')
        if font_name and font_name != 'Unknown':
            constraints.append({
                'type': 'font_preservation',
                'value': font_name,
                'description': f'Use font: {font_name}'
            })

        return constraints

    def parse_xliff(self, xliff_content: str) -> Dict[str, Any]:
        """
        Parse XLIFF document to extract translation data.

        Args:
            xliff_content: XLIFF document content

        Returns:
            Parsed XLIFF data structure
        """
        try:
            root = ET.fromstring(xliff_content)

            # Extract basic information
            src_lang = root.get('srcLang', 'en')
            trg_lang = root.get('trgLang', 'zh')

            # Extract files
            files = []
            for file_elem in root.findall('file'):
                file_data = {
                    'id': file_elem.get('id'),
                    'original': file_elem.get('original'),
                    'skeleton': self._parse_skeleton(file_elem.find('skeleton')),
                    'units': []
                }

                # Extract translation units
                for unit_elem in file_elem.findall('unit'):
                    unit_data = {
                        'id': unit_elem.get('id'),
                        'source': unit_elem.find('segment').text or '',
                        'target': unit_elem.find('segment[1]').text or '',  # Second segment is target
                        'notes': [note.text for note in unit_elem.findall('notes/note')]
                    }
                    file_data['units'].append(unit_data)

                files.append(file_data)

            return {
                'srcLang': src_lang,
                'trgLang': trg_lang,
                'files': files
            }

        except Exception as e:
            logger.error(f"Failed to parse XLIFF: {str(e)}")
            return {
                'srcLang': 'en',
                'trgLang': 'zh',
                'files': []
            }

    def _parse_skeleton(self, skeleton_elem: Optional[ET.Element]) -> Dict[str, Any]:
        """
        Parse skeleton data from XLIFF.

        Args:
            skeleton_elem: Skeleton XML element

        Returns:
            Parsed skeleton data
        """
        if not skeleton_elem or not skeleton_elem.text:
            return {}

        try:
            import json
            return json.loads(skeleton_elem.text)
        except Exception:
            return {}

    def create_xliff(self, content: Dict[str, Any], source_lang: str,
                    target_lang: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create XLIFF document from structured content.

        Args:
            content: Structured content data
            source_lang: Source language
            target_lang: Target language
            metadata: Additional metadata

        Returns:
            XLIFF document as string
        """
        # Create XLIFF root
        root = ET.Element('xliff')
        root.set('version', '2.1')
        root.set('xmlns', self.namespaces['xliff'])
        root.set('xmlns:fs', self.namespaces['fs'])
        root.set('srcLang', source_lang)
        root.set('trgLang', target_lang)

        # Add file element
        file_elem = ET.SubElement(root, 'file')
        file_elem.set('id', f"generated_{uuid.uuid4().hex[:8]}")

        if metadata:
            file_elem.set('original', metadata.get('original_file', ''))

        # Add skeleton
        skeleton = ET.SubElement(file_elem, 'skeleton')
        if metadata:
            skeleton.text = self._generate_skeleton_data(metadata)

        # Add translation units
        pages = content.get('pages', [])
        for page_idx, page in enumerate(pages):
            for block_idx, block in enumerate(page.get('text_blocks', [])):
                unit_elem = ET.SubElement(file_elem, 'unit')
                unit_elem.set('id', f"p{page_idx}_b{block_idx}")

                # Source segment
                source_elem = ET.SubElement(unit_elem, 'segment')
                source_elem.text = block.get('text', '')

                # Target segment (empty initially)
                target_elem = ET.SubElement(unit_elem, 'segment')
                target_elem.set('state', 'initial')

                # Add notes for constraints
                if block.get('constraints'):
                    notes_elem = ET.SubElement(unit_elem, 'notes')
                    for constraint in block.get('constraints', []):
                        note_elem = ET.SubElement(notes_elem, 'note')
                        note_elem.text = str(constraint)

        # Convert to formatted string
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def validate_xliff(self, xliff_content: str) -> tuple[bool, List[str]]:
        """
        Validate XLIFF document structure.

        Args:
            xliff_content: XLIFF document content

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            root = ET.fromstring(xliff_content)

            # Check required attributes
            if not root.get('srcLang'):
                errors.append("Missing srcLang attribute")
            if not root.get('trgLang'):
                errors.append("Missing trgLang attribute")

            # Check file elements
            files = root.findall('file')
            if not files:
                errors.append("No file elements found")

            for file_elem in files:
                # Check for units
                units = file_elem.findall('unit')
                if not units:
                    errors.append(f"File {file_elem.get('id')} has no translation units")

                for unit in units:
                    # Check for source content
                    source_segments = unit.findall('segment')
                    if not source_segments:
                        errors.append(f"Unit {unit.get('id')} has no source segments")
                    elif not source_segments[0].text:
                        errors.append(f"Unit {unit.get('id')} has empty source content")

        except ET.ParseError as e:
            errors.append(f"XML parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors
