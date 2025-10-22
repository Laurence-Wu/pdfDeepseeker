"""
Content Classifier - Determine what content should be translated vs preserved
"""
import re
from typing import Dict, List, Tuple
import fitz


class ContentClassifier:
    """
    Classifies text blocks to determine translation behavior.
    Identifies formulas, citations, code, URLs, etc. that should be preserved.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def classify_text(self, text: str, context: Dict = None) -> Dict:
        """
        Classify a text block.

        Returns:
            {
                'should_translate': bool,
                'preserve_reason': str or None,
                'content_type': str,
                'confidence': float
            }
        """
        context = context or {}

        # Check each classification in order of priority
        checks = [
            self._check_formula,
            self._check_citation,
            self._check_reference_list,
            self._check_code,
            self._check_url,
            self._check_email,
            self._check_equation_label,
            self._check_figure_label,
            self._check_table_label,
        ]

        for check_func in checks:
            result = check_func(text, context)
            if not result['should_translate']:
                return result

        # Default: should translate
        return {
            'should_translate': True,
            'preserve_reason': None,
            'content_type': 'text',
            'confidence': 0.9
        }

    def _check_formula(self, text: str, context: Dict) -> Dict:
        """Check if text contains mathematical formula"""

        # LaTeX-style formulas
        if re.search(r'[\\]\w+\{', text):  # \frac{, \sqrt{, etc.
            return {
                'should_translate': False,
                'preserve_reason': 'latex_formula',
                'content_type': 'formula',
                'confidence': 0.95
            }

        # Inline formulas with symbols
        math_symbols = ['∂', '∫', '∑', '∏', '√', '≈', '≠', '≤', '≥', '±', '×', '÷', '∞']
        symbol_count = sum(1 for symbol in math_symbols if symbol in text)

        # Greek letters (often in formulas)
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω', 'Ω']
        greek_count = sum(1 for letter in greek_letters if letter in text)

        # Equation-like patterns: "x = y", "a + b", "f(x)"
        equation_patterns = [
            r'\b[a-zA-Z]\s*[=+\-*/]\s*[a-zA-Z0-9]',  # x = y, a + b
            r'\b[a-zA-Z]\([a-zA-Z0-9,\s]*\)',  # f(x), g(x,y)
            r'\d+\s*[×÷]\s*\d+',  # 5 × 3
        ]
        has_equation = any(re.search(pattern, text) for pattern in equation_patterns)

        # Decision: formula if multiple indicators
        indicators = symbol_count + greek_count + (1 if has_equation else 0)

        if indicators >= 2 or symbol_count >= 3:
            return {
                'should_translate': False,
                'preserve_reason': 'mathematical_formula',
                'content_type': 'formula',
                'confidence': 0.8 + min(indicators * 0.05, 0.15)
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_citation(self, text: str, context: Dict) -> Dict:
        """Check if text is a citation"""

        # Numbered citations: [1], [2, 3]
        if re.match(r'^\[\d+(?:,\s*\d+)*\]$', text.strip()):
            return {
                'should_translate': False,
                'preserve_reason': 'citation_number',
                'content_type': 'citation',
                'confidence': 0.99
            }

        # Author-year citations: (Smith, 2024)
        if re.match(r'^\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,|\s+&|\s+and)\s*\d{4}\)$', text.strip()):
            return {
                'should_translate': False,
                'preserve_reason': 'citation_author_year',
                'content_type': 'citation',
                'confidence': 0.95
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_reference_list(self, text: str, context: Dict) -> Dict:
        """Check if text is part of a reference list"""

        # Reference list entry pattern: [1] Author. (Year). Title.
        if re.match(r'^\[\d+\]\s+[A-Z][a-z]+.*\(\d{4}\)', text.strip()):
            return {
                'should_translate': False,
                'preserve_reason': 'reference_entry',
                'content_type': 'reference',
                'confidence': 0.95
            }

        # Section header
        if re.match(r'^(References|Bibliography)\s*$', text.strip(), re.IGNORECASE):
            return {
                'should_translate': True,  # Translate the header, but mark as reference section
                'preserve_reason': None,
                'content_type': 'reference_header',
                'confidence': 0.99
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_code(self, text: str, context: Dict) -> Dict:
        """Check if text is code"""

        # Programming keywords
        code_keywords = ['def ', 'class ', 'import ', 'function ', 'var ', 'const ', 'return ', 'if ', 'else ', 'for ', 'while ']
        has_code_keyword = any(keyword in text.lower() for keyword in code_keywords)

        # Code-like patterns: semicolons, curly braces, etc.
        code_symbols = sum(text.count(sym) for sym in [';', '{', '}', '->', '=>', '==='])

        if has_code_keyword or code_symbols >= 2:
            return {
                'should_translate': False,
                'preserve_reason': 'code_block',
                'content_type': 'code',
                'confidence': 0.85
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_url(self, text: str, context: Dict) -> Dict:
        """Check if text is a URL"""

        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        if re.search(url_pattern, text.lower()):
            # If URL is substantial part of text, don't translate
            url_chars = sum(len(match.group(0)) for match in re.finditer(url_pattern, text.lower()))
            if url_chars / max(len(text), 1) > 0.3:
                return {
                    'should_translate': False,
                    'preserve_reason': 'contains_url',
                    'content_type': 'url',
                    'confidence': 0.9
                }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_email(self, text: str, context: Dict) -> Dict:
        """Check if text is an email"""

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, text):
            return {
                'should_translate': False,
                'preserve_reason': 'contains_email',
                'content_type': 'email',
                'confidence': 0.95
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_equation_label(self, text: str, context: Dict) -> Dict:
        """Check if text is an equation label/number"""

        # Equation labels: "Eq. 1", "(1)", "Equation 2.3"
        if re.match(r'^(Eq\.|Equation)\s*\d+(\.\d+)?$', text.strip(), re.IGNORECASE):
            return {
                'should_translate': False,
                'preserve_reason': 'equation_label',
                'content_type': 'label',
                'confidence': 0.9
            }

        if re.match(r'^\(\d+(\.\d+)?\)$', text.strip()):
            return {
                'should_translate': False,
                'preserve_reason': 'equation_number',
                'content_type': 'label',
                'confidence': 0.85
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_figure_label(self, text: str, context: Dict) -> Dict:
        """Check if text is a figure label"""

        if re.match(r'^(Fig\.|Figure)\s*\d+', text.strip(), re.IGNORECASE):
            # Translate "Figure" but not the number
            return {
                'should_translate': True,  # Will be handled specially
                'preserve_reason': None,
                'content_type': 'figure_label',
                'confidence': 0.9
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}

    def _check_table_label(self, text: str, context: Dict) -> Dict:
        """Check if text is a table label"""

        if re.match(r'^Table\s*\d+', text.strip(), re.IGNORECASE):
            # Translate "Table" but not the number
            return {
                'should_translate': True,  # Will be handled specially
                'preserve_reason': None,
                'content_type': 'table_label',
                'confidence': 0.9
            }

        return {'should_translate': True, 'preserve_reason': None, 'content_type': 'text', 'confidence': 0.5}
