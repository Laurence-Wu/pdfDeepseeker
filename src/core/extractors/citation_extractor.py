"""
Citation Extractor - Detect and preserve academic citations
"""
import re
from typing import List, Dict, Tuple
import fitz


class CitationExtractor:
    """
    Extract and preserve academic citations.
    Supports various citation formats.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Citation patterns
        self.patterns = {
            'numbered': r'\[\d+\]',  # [1], [2], etc.
            'author_year': r'\b[A-Z][a-z]+(?:\s+(?:et\s+al\.|&|and)\s+[A-Z][a-z]+)?\s*\(\d{4}[a-z]?\)',  # Smith (2024), Doe & Wang (2023)
            'reference_line': r'^\[\d+\]\s+.+\(\d{4}\)',  # Full reference lines
        }

    def extract_citations(self, pdf_path: str) -> List[Dict]:
        """
        Extract all citations from PDF.

        Returns:
            List of citation dictionaries with bbox and text
        """
        citations = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            page_citations = self._extract_from_page(page, page_num)
            citations.extend(page_citations)

        doc.close()
        return citations

    def _extract_from_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract citations from a single page"""
        citations = []

        # Get text with position information
        blocks = page.get_text("dict")['blocks']

        for block in blocks:
            if block['type'] != 0:  # Skip non-text blocks
                continue

            for line in block.get('lines', []):
                line_text = ""
                line_bbox = None

                # Reconstruct line text and bbox
                for span in line.get('spans', []):
                    line_text += span['text']
                    if line_bbox is None:
                        line_bbox = list(span['bbox'])
                    else:
                        # Extend bbox to include this span
                        line_bbox[2] = max(line_bbox[2], span['bbox'][2])
                        line_bbox[3] = max(line_bbox[3], span['bbox'][3])

                if not line_text.strip():
                    continue

                # Check for citations
                citation_matches = self._find_citations_in_text(line_text)

                for match in citation_matches:
                    citation_text, citation_type, start_pos, end_pos = match

                    # Estimate bbox for this specific citation within the line
                    # (simplified - uses full line bbox for now)
                    citations.append({
                        'page': page_num,
                        'bbox': {
                            'x': line_bbox[0],
                            'y': line_bbox[1],
                            'width': line_bbox[2] - line_bbox[0],
                            'height': line_bbox[3] - line_bbox[1]
                        },
                        'text': citation_text,
                        'type': citation_type,
                        'full_line': line_text,
                        'is_reference_list': citation_type == 'reference_line'
                    })

        return citations

    def _find_citations_in_text(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Find all citations in text.

        Returns:
            List of (citation_text, citation_type, start_pos, end_pos)
        """
        matches = []

        # Check each pattern
        for citation_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                citation_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()

                matches.append((citation_text, citation_type, start_pos, end_pos))

        return matches

    def is_citation_block(self, text: str) -> bool:
        """Check if a text block is primarily citations"""

        # Check for reference list indicators
        if re.search(r'^References\s*$', text, re.IGNORECASE | re.MULTILINE):
            return True

        if re.search(r'^Bibliography\s*$', text, re.IGNORECASE | re.MULTILINE):
            return True

        # Check if text contains multiple numbered citations
        numbered_citations = re.findall(self.patterns['numbered'], text)
        if len(numbered_citations) >= 3:  # At least 3 citations
            return True

        # Check for reference line pattern
        if re.match(self.patterns['reference_line'], text.strip()):
            return True

        return False

    def should_preserve_text(self, text: str) -> bool:
        """Determine if text should be preserved (not translated) due to citations"""

        # If it's a reference block, preserve entirely
        if self.is_citation_block(text):
            return True

        # If text is mostly citation (>50% of length is citation markers)
        citation_chars = 0
        for citation_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                citation_chars += len(match.group(0))

        if len(text.strip()) > 0 and (citation_chars / len(text.strip())) > 0.5:
            return True

        return False
