# Text Length Controller - Complete Implementation

## Overview
Precisely controls translated text length to ensure it fits within original layout boundaries.

## Implementation

```python
from fonttools.ttLib import TTFont
import io
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class TextMeasurement:
    """Text measurement result"""
    width: float
    height: float
    char_count: int
    fits_bbox: bool
    overflow_ratio: float  # >1 means overflow

class TextLengthController:
    """
    Control translated text length to fit original layout.
    Uses precise font metrics and multiple strategies.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_expansion_ratio = self.config.get('max_expansion_ratio', 1.1)
        self.min_compression_ratio = self.config.get('min_compression_ratio', 0.7)
        self.font_cache = {}
        self.abbreviation_dict = self._load_abbreviations()
        
    def measure_text(self, text: str, font_info: Dict) -> TextMeasurement:
        """
        Precisely measure text using font metrics.
        
        Args:
            text: Text to measure
            font_info: Font information including name, size, data
            
        Returns:
            TextMeasurement object
        """
        font_name = font_info.get('name', 'default')
        font_size = font_info.get('size', 12)
        
        # Load font metrics
        font_metrics = self._get_font_metrics(font_info)
        
        if not font_metrics:
            # Fallback to estimation
            return self._estimate_text_size(text, font_size)
        
        # Calculate precise width
        total_width = 0
        for char in text:
            char_width = font_metrics.get_char_width(char)
            total_width += char_width
        
        # Scale to font size
        actual_width = (total_width / font_metrics.units_per_em) * font_size
        
        # Calculate height
        line_height = ((font_metrics.ascent - font_metrics.descent) / 
                      font_metrics.units_per_em) * font_size
        
        # Account for multi-line text
        lines = text.split('\n')
        actual_height = line_height * len(lines)
        
        return TextMeasurement(
            width=actual_width,
            height=actual_height,
            char_count=len(text),
            fits_bbox=True,  # To be determined by caller
            overflow_ratio=1.0
        )
    
    def _get_font_metrics(self, font_info: Dict) -> Optional['FontMetrics']:
        """Get or load font metrics"""
        
        font_name = font_info.get('name', 'default')
        
        # Check cache
        if font_name in self.font_cache:
            return self.font_cache[font_name]
        
        # Load font
        if font_info.get('data'):
            try:
                font_data = font_info['data']
                metrics = FontMetrics(font_data)
                self.font_cache[font_name] = metrics
                return metrics
            except Exception as e:
                print(f"Failed to load font metrics: {e}")
        
        return None
    
    def _estimate_text_size(self, text: str, font_size: float) -> TextMeasurement:
        """Estimate text size without font metrics"""
        
        # Average character width estimation
        avg_char_width = font_size * 0.5
        width = len(text) * avg_char_width
        height = font_size * 1.2  # Line height
        
        return TextMeasurement(
            width=width,
            height=height,
            char_count=len(text),
            fits_bbox=True,
            overflow_ratio=1.0
        )
    
    def fit_translation(self, translation: str, source_bbox: Dict, 
                       font_info: Dict) -> Dict:
        """
        Ensure translation fits in original bounding box.
        
        Args:
            translation: Translated text
            source_bbox: Original text bounding box
            font_info: Font information
            
        Returns:
            Fitting result with adjusted text
        """
        # Measure translation
        measurement = self.measure_text(translation, font_info)
        
        # Check if it fits
        bbox_width = source_bbox.get('width', 100)
        bbox_height = source_bbox.get('height', 20)
        
        overflow_ratio = max(
            measurement.width / bbox_width,
            measurement.height / bbox_height
        )
        
        result = {
            'original_text': translation,
            'fitted_text': translation,
            'method': 'none',
            'success': True,
            'overflow_ratio': overflow_ratio
        }
        
        # If text doesn't fit, apply strategies
        if overflow_ratio > 1.0:
            # Try multiple strategies
            strategies = [
                ('abbreviation', self._apply_abbreviation),
                ('spacing', self._adjust_spacing),
                ('font_size', self._reduce_font_size),
                ('truncation', self._truncate_text)
            ]
            
            for strategy_name, strategy_func in strategies:
                fitted_text = strategy_func(
                    translation, source_bbox, font_info, overflow_ratio
                )
                
                # Measure fitted text
                new_measurement = self.measure_text(fitted_text, font_info)
                new_overflow = max(
                    new_measurement.width / bbox_width,
                    new_measurement.height / bbox_height
                )
                
                if new_overflow <= 1.0:
                    result['fitted_text'] = fitted_text
                    result['method'] = strategy_name
                    result['overflow_ratio'] = new_overflow
                    break
            else:
                # No strategy worked completely
                result['success'] = False
                result['fitted_text'] = fitted_text  # Use last attempt
        
        return result
    
    def _apply_abbreviation(self, text: str, bbox: Dict, 
                           font_info: Dict, overflow_ratio: float) -> str:
        """Apply abbreviation strategy"""
        
        abbreviated = text
        
        # Common abbreviations
        replacements = [
            (r'\bInternational\b', 'Int\'l'),
            (r'\bCorporation\b', 'Corp.'),
            (r'\bCompany\b', 'Co.'),
            (r'\bLimited\b', 'Ltd.'),
            (r'\bDepartment\b', 'Dept.'),
            (r'\bManagement\b', 'Mgmt.'),
            (r'\bInformation\b', 'Info'),
            (r'\bTechnology\b', 'Tech'),
            (r'\bDevelopment\b', 'Dev'),
            (r'\bApplication\b', 'App'),
            (r'\bDocument\b', 'Doc'),
            (r'\bNumber\b', 'No.'),
            (r'\bVersion\b', 'Ver.'),
            (r'\bMaximum\b', 'Max'),
            (r'\bMinimum\b', 'Min'),
            (r'\bAverage\b', 'Avg'),
        ]
        
        for pattern, replacement in replacements:
            abbreviated = re.sub(pattern, replacement, abbreviated, flags=re.IGNORECASE)
        
        # Remove articles if still too long
        if overflow_ratio > 1.2:
            abbreviated = re.sub(r'\b(the|a|an)\b', '', abbreviated, flags=re.IGNORECASE)
            abbreviated = ' '.join(abbreviated.split())  # Clean up spaces
        
        return abbreviated
    
    def _adjust_spacing(self, text: str, bbox: Dict, 
                       font_info: Dict, overflow_ratio: float) -> str:
        """Adjust character/word spacing"""
        
        # Remove extra spaces
        adjusted = ' '.join(text.split())
        
        # Remove spaces around punctuation
        adjusted = re.sub(r'\s+([.,;!?])', r'\1', adjusted)
        adjusted = re.sub(r'([.,;!?])\s+', r'\1', adjusted)
        
        # Use narrow spaces if supported
        if overflow_ratio < 1.1:
            # Replace normal spaces with thin spaces
            adjusted = adjusted.replace(' ', '\u2009')  # Thin space
        
        return adjusted
    
    def _reduce_font_size(self, text: str, bbox: Dict, 
                         font_info: Dict, overflow_ratio: float) -> str:
        """Recommend font size reduction"""
        
        # This method returns metadata for font adjustment
        # Actual font size change happens during reconstruction
        
        # Add metadata marker
        if overflow_ratio > 1.0:
            # Calculate required font size
            current_size = font_info.get('size', 12)
            required_size = current_size / overflow_ratio
            
            # Add invisible marker for reconstruction
            marker = f"\u200B[FONTSIZE:{required_size:.1f}]\u200B"
            return marker + text
        
        return text
    
    def _truncate_text(self, text: str, bbox: Dict, 
                      font_info: Dict, overflow_ratio: float) -> str:
        """Truncate text as last resort"""
        
        if overflow_ratio <= 1.0:
            return text
        
        # Calculate how many characters to keep
        target_length = int(len(text) / overflow_ratio)
        
        if target_length < len(text):
            # Try to truncate at word boundary
            truncated = text[:target_length]
            last_space = truncated.rfind(' ')
            
            if last_space > target_length * 0.8:
                truncated = text[:last_space]
            
            # Add ellipsis
            truncated = truncated.rstrip() + '...'
            
            return truncated
        
        return text
    
    def calculate_max_length(self, source_text: str, source_bbox: Dict, 
                           font_info: Dict) -> int:
        """
        Calculate maximum allowed length for translation.
        
        Args:
            source_text: Original text
            source_bbox: Original bounding box
            font_info: Font information
            
        Returns:
            Maximum character count
        """
        # Measure source text
        source_measurement = self.measure_text(source_text, font_info)
        
        # Calculate average character width
        if len(source_text) > 0:
            avg_char_width = source_measurement.width / len(source_text)
        else:
            avg_char_width = font_info.get('size', 12) * 0.5
        
        # Calculate maximum characters based on bbox
        max_width = source_bbox.get('width', 100) * self.max_expansion_ratio
        max_chars = int(max_width / avg_char_width)
        
        return max_chars
    
    def validate_translation_fit(self, translation: str, source_bbox: Dict,
                                font_info: Dict) -> Dict:
        """
        Validate if translation fits in source bbox.
        
        Returns:
            Validation result with details
        """
        measurement = self.measure_text(translation, font_info)
        
        bbox_width = source_bbox.get('width', 100)
        bbox_height = source_bbox.get('height', 20)
        
        width_ratio = measurement.width / bbox_width
        height_ratio = measurement.height / bbox_height
        
        return {
            'fits': width_ratio <= 1.0 and height_ratio <= 1.0,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'overflow_direction': 'width' if width_ratio > height_ratio else 'height',
            'adjustment_needed': max(width_ratio, height_ratio) - 1.0
        }
    
    def generate_length_constraint(self, source_text: str, source_bbox: Dict,
                                  font_info: Dict) -> Dict:
        """
        Generate constraint for translation request.
        
        Returns:
            Constraint dictionary for prompt
        """
        max_chars = self.calculate_max_length(source_text, source_bbox, font_info)
        
        return {
            'max_length': max_chars,
            'current_length': len(source_text),
            'expansion_allowed': self.max_expansion_ratio,
            'compression_allowed': self.min_compression_ratio,
            'bbox': source_bbox,
            'strategy_preference': ['abbreviation', 'spacing', 'font_size']
        }
    
    def _load_abbreviations(self) -> Dict:
        """Load abbreviation dictionary"""
        
        return {
            # English abbreviations
            'International': 'Int\'l',
            'Corporation': 'Corp.',
            'Company': 'Co.',
            'Limited': 'Ltd.',
            'Incorporated': 'Inc.',
            'Department': 'Dept.',
            'Management': 'Mgmt.',
            'Information': 'Info',
            'Technology': 'Tech',
            'Development': 'Dev',
            'Application': 'App',
            'Document': 'Doc',
            'Administrator': 'Admin',
            'Configuration': 'Config',
            'Specification': 'Spec',
            # Add more as needed
        }


class FontMetrics:
    """Helper class for font metrics"""
    
    def __init__(self, font_data: bytes):
        """Load font metrics from font data"""
        
        self.font = TTFont(io.BytesIO(font_data))
        self.units_per_em = self.font['head'].unitsPerEm
        
        # Get metrics
        if 'hhea' in self.font:
            self.ascent = self.font['hhea'].ascent
            self.descent = self.font['hhea'].descent
            self.line_gap = self.font['hhea'].lineGap
        else:
            self.ascent = self.units_per_em * 0.8
            self.descent = -self.units_per_em * 0.2
            self.line_gap = 0
        
        # Load character widths
        self.char_widths = {}
        if 'hmtx' in self.font:
            hmtx = self.font['hmtx']
            cmap = self.font.getBestCmap()
            
            if cmap:
                for char_code, glyph_name in cmap.items():
                    if glyph_name in hmtx.metrics:
                        width, lsb = hmtx.metrics[glyph_name]
                        self.char_widths[chr(char_code)] = width
        
        # Calculate average width
        if self.char_widths:
            self.avg_width = sum(self.char_widths.values()) / len(self.char_widths)
        else:
            self.avg_width = self.units_per_em * 0.5
    
    def get_char_width(self, char: str) -> float:
        """Get width of specific character"""
        
        return self.char_widths.get(char, self.avg_width)


class TextFittingStrategy:
    """Advanced text fitting strategies"""
    
    @staticmethod
    def apply_hyphenation(text: str, max_width: float, 
                         char_width: float) -> str:
        """Apply hyphenation to fit text"""
        
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = len(word) * char_width
            
            if current_width + word_width > max_width:
                # Try to hyphenate
                if len(word) > 8 and word_width > max_width * 0.3:
                    # Simple hyphenation at syllable boundary
                    hyphen_point = len(word) // 2
                    part1 = word[:hyphen_point] + '-'
                    part2 = word[hyphen_point:]
                    
                    current_line.append(part1)
                    lines.append(' '.join(current_line))
                    current_line = [part2]
                    current_width = len(part2) * char_width
                else:
                    # Start new line
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width + char_width  # Space
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    @staticmethod
    def apply_condensed_style(text: str) -> str:
        """Apply condensed writing style"""
        
        # Remove redundant words
        condensed = text
        redundant_patterns = [
            (r'\bvery\s+', ''),
            (r'\bquite\s+', ''),
            (r'\bthat\s+is\b', 'i.e.'),
            (r'\bfor\s+example\b', 'e.g.'),
            (r'\bin\s+order\s+to\b', 'to'),
            (r'\bas\s+well\s+as\b', 'and'),
            (r'\bin\s+addition\s+to\b', 'plus'),
        ]
        
        for pattern, replacement in redundant_patterns:
            condensed = re.sub(pattern, replacement, condensed, flags=re.IGNORECASE)
        
        return condensed
```

## Usage Example

```python
# Initialize controller
controller = TextLengthController()

# Measure text
font_info = {
    'name': 'Arial',
    'size': 12,
    'data': font_data_bytes
}

measurement = controller.measure_text("Hello World", font_info)
print(f"Text width: {measurement.width}px")
print(f"Text height: {measurement.height}px")

# Fit translation
source_bbox = {'x': 100, 'y': 100, 'width': 200, 'height': 20}
translation = "This is a very long translation that might not fit"

result = controller.fit_translation(translation, source_bbox, font_info)
print(f"Fitted text: {result['fitted_text']}")
print(f"Method used: {result['method']}")
print(f"Success: {result['success']}")

# Generate constraint for translation
constraint = controller.generate_length_constraint(
    "Original text",
    source_bbox,
    font_info
)
print(f"Max length allowed: {constraint['max_length']} characters")
```
