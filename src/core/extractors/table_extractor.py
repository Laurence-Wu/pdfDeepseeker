"""
Table Extractor - Extract tables using pdfplumber and Table Transformer
Preserves structure and formatting.
"""

import pdfplumber
import pandas as pd
from typing import Dict, List, Optional, Tuple


class TableExtractor:
    """
    Extract tables using pdfplumber and Table Transformer.
    Preserves structure and formatting.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        self.extract_structure = self.config.get('extract_structure', True)

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract all tables from PDF with cell-level granularity for translation.

        Args:
            pdf_path: Path to PDF

        Returns:
            List of table dictionaries with translatable cells
        """
        tables = []

        # Use pdfplumber for table extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Find tables
                page_tables = page.find_tables()

                for table_idx, table in enumerate(page_tables):
                    # Extract table data
                    table_data = table.extract()

                    if table_data and len(table_data) > 0:
                        # Parse and structure table with cell-level detail
                        structured = self._structure_table_with_cells(table_data, table.bbox, page)

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
                            'cells': structured['cells'],  # Cell-by-cell data for translation
                            'headers': structured['headers'],
                            'data': structured['data'],
                            'style': self._analyze_table_style(table),
                            'raw_data': table_data,
                            'translatable': True  # Mark as translatable
                        })

        return tables

    def _structure_table_with_cells(self, table_data: List[List], bbox: Tuple, page) -> Dict:
        """
        Structure table data with individual cell extraction for translation.
        Expands multi-line cells into separate rows for proper table structure.
        """

        structured = {
            'rows': [],
            'columns': 0,
            'headers': [],
            'data': [],
            'cells': []  # Individual cells with positions for translation
        }

        if not table_data:
            return structured

        # Determine number of columns
        max_cols = max(len(row) for row in table_data) if table_data else 0
        structured['columns'] = max_cols

        # Expand multi-line cells into separate rows
        expanded_data = self._expand_multiline_rows(table_data)

        # Calculate cell dimensions based on expanded rows
        table_width = bbox[2] - bbox[0]
        table_height = bbox[3] - bbox[1]
        cell_width = table_width / max_cols if max_cols > 0 else 0
        cell_height = table_height / len(expanded_data) if len(expanded_data) > 0 else 0

        # Process all cells
        for row_idx, row in enumerate(expanded_data):
            is_header = row_idx == 0
            row_data = []

            for col_idx, cell_content in enumerate(row):
                cell_text = str(cell_content) if cell_content else ''

                # Calculate cell position
                cell_x = bbox[0] + (col_idx * cell_width)
                cell_y = bbox[1] + (row_idx * cell_height)

                # Create cell entry
                cell_dict = {
                    'text': cell_text,
                    'row': row_idx,
                    'col': col_idx,
                    'is_header': is_header,
                    'bbox': {
                        'x': cell_x,
                        'y': cell_y,
                        'width': cell_width,
                        'height': cell_height
                    },
                    'translatable': len(cell_text) > 1 and not self._is_numeric(cell_text)
                }

                row_data.append(cell_dict)
                structured['cells'].append(cell_dict)

            structured['rows'].append(row_data)

        # Extract headers from first row
        if len(expanded_data) > 0:
            structured['headers'] = [str(cell) if cell else '' for cell in expanded_data[0]]

        # Convert to pandas for data representation
        try:
            if len(expanded_data) > 1:
                df = pd.DataFrame(expanded_data[1:], columns=structured['headers'])
                structured['data'] = df.to_dict('records')
            else:
                structured['data'] = []
        except Exception as e:
            print(f"Warning: Could not convert table to dataframe: {e}")
            structured['data'] = expanded_data

        return structured

    def _expand_multiline_rows(self, table_data: List[List]) -> List[List]:
        """
        Expand rows that contain multi-line cells into separate rows.

        Example:
        [['A', 'B'], ['C\nD', 'E\nF']] -> [['A', 'B'], ['C', 'E'], ['D', 'F']]
        """
        if not table_data:
            return table_data

        # Check if any cells have newlines (multi-line)
        has_multiline = False
        for row in table_data:
            for cell in row:
                if cell and '\n' in str(cell):
                    has_multiline = True
                    break
            if has_multiline:
                break

        if not has_multiline:
            return table_data

        expanded = []

        for row_idx, row in enumerate(table_data):
            # Skip header row (first row) - keep as-is
            if row_idx == 0:
                expanded.append(row)
                continue

            # Split each cell by newlines
            cell_lines = []
            max_lines = 0

            for cell in row:
                cell_text = str(cell) if cell else ''
                lines = [line.strip() for line in cell_text.split('\n') if line.strip()]
                if not lines:
                    lines = ['']
                cell_lines.append(lines)
                max_lines = max(max_lines, len(lines))

            # Pad all cells to have the same number of lines
            for lines in cell_lines:
                while len(lines) < max_lines:
                    lines.append('')

            # Create separate rows for each line
            for line_idx in range(max_lines):
                new_row = [cell_lines[col_idx][line_idx] for col_idx in range(len(row))]
                expanded.append(new_row)

        return expanded

    def _is_numeric(self, text: str) -> bool:
        """Check if text is primarily numeric (values, not labels)"""
        # Remove common units and symbols
        cleaned = text.replace('%', '').replace('·s', '').replace(',', '').strip()
        try:
            # Check if multiple lines are all numbers
            for line in cleaned.split('\n'):
                line = line.strip()
                if line and not line.replace('.', '').replace('-', '').isdigit():
                    return False
            return True
        except:
            return False

    def _structure_table(self, table_data: List[List], bbox: Tuple) -> Dict:
        """Structure table data (legacy method)"""

        structured = {
            'rows': [],
            'columns': 0,
            'headers': [],
            'data': []
        }

        if not table_data:
            return structured

        # Assume first row is header
        if len(table_data) > 0:
            structured['headers'] = [str(cell) if cell else '' for cell in table_data[0]]
            structured['columns'] = len(table_data[0])

        # Process rows
        for row_idx, row in enumerate(table_data):
            row_data = []
            for col_idx, cell in enumerate(row):
                cell_text = str(cell) if cell else ''
                row_data.append({
                    'text': cell_text,
                    'row': row_idx,
                    'col': col_idx,
                    'is_header': row_idx == 0
                })
            structured['rows'].append(row_data)

        # Convert to pandas for easier manipulation
        try:
            if len(table_data) > 1:
                df = pd.DataFrame(table_data[1:], columns=structured['headers'])
                structured['data'] = df.to_dict('records')
            else:
                structured['data'] = []
        except Exception as e:
            print(f"Warning: Could not convert table to dataframe: {e}")
            structured['data'] = table_data

        return structured

    def _analyze_table_style(self, table) -> Dict:
        """Analyze table styling"""

        return {
            'has_header': True,  # Assumption
            'border_style': 'solid',
            'alignment': 'left',
            'cell_count': len(table.cells) if hasattr(table, 'cells') else 0
        }

    def table_to_markdown(self, table_dict: Dict) -> str:
        """Convert table to markdown format"""

        if not table_dict['rows']:
            return ""

        markdown_lines = []

        # Header
        if table_dict['headers']:
            header_line = "| " + " | ".join(table_dict['headers']) + " |"
            separator_line = "|" + "|".join([" --- " for _ in table_dict['headers']]) + "|"
            markdown_lines.append(header_line)
            markdown_lines.append(separator_line)

        # Data rows (skip header row)
        for row in table_dict['rows'][1:]:
            row_text = "| " + " | ".join([cell['text'] for cell in row]) + " |"
            markdown_lines.append(row_text)

        return "\n".join(markdown_lines)

    def table_to_html(self, table_dict: Dict) -> str:
        """Convert table to HTML format"""

        html_lines = ['<table>']

        # Header
        if table_dict['headers']:
            html_lines.append('  <thead>')
            html_lines.append('    <tr>')
            for header in table_dict['headers']:
                html_lines.append(f'      <th>{header}</th>')
            html_lines.append('    </tr>')
            html_lines.append('  </thead>')

        # Body
        html_lines.append('  <tbody>')
        for row in table_dict['rows'][1:]:
            html_lines.append('    <tr>')
            for cell in row:
                html_lines.append(f'      <td>{cell["text"]}</td>')
            html_lines.append('    </tr>')
        html_lines.append('  </tbody>')

        html_lines.append('</table>')
        return "\n".join(html_lines)
