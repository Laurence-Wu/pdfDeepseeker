# Margin Manager - Complete Implementation

## Overview
Detects document margins and ensures translated content respects original boundaries.

## Implementation

```python
import pdfplumber
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import statistics

@dataclass
class Margin:
    """Margin data structure"""
    top: float
    bottom: float
    left: float
    right: float
    page_num: int
    confidence: float = 1.0

class MarginManager:
    """
    Detect and enforce document margins.
    Uses multiple methods for accurate margin detection.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize MarginManager.
        
        Args:
            config: Configuration dict with thresholds
        """
        self.config = config or {}
        self.threshold = self.config.get('threshold', 10)  # pixels
        self.enforce_strict = self.config.get('enforce_strict', True)
        self.detection_method = self.config.get('method', 'pdfplumber')
        self.min_margin = self.config.get('min_margin', 36)  # 0.5 inch
        
    def extract_margins(self, pdf_path: str) -> List[Margin]:
        """
        Extract margins from PDF using content boundaries.
        
        Algorithm:
        1. For each page, find bounding box of all content
        2. Calculate distance from content to page edges
        3. Use statistical mode for consistent margins
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Margin objects, one per page
        """
        margins = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract margin for this page
                margin = self._extract_page_margin(page, page_num)
                margins.append(margin)
        
        # Apply consistency analysis
        consistent_margins = self._find_consistent_margins(margins)
        
        # Optionally enforce consistent margins across all pages
        if self.config.get('enforce_consistent', True):
            margins = self._apply_consistent_margins(margins, consistent_margins)
        
        return margins
    
    def _extract_page_margin(self, page, page_num: int) -> Margin:
        """
        Extract margin for a single page.
        
        Args:
            page: pdfplumber page object
            page_num: Page number
            
        Returns:
            Margin object
        """
        # Get page dimensions
        page_width = page.width
        page_height = page.height
        
        # Find content bounding box
        content_bbox = self.detect_content_boundaries(page)
        
        if content_bbox:
            x0, y0, x1, y1 = content_bbox
            
            # Calculate margins
            margin = Margin(
                top=y0,
                bottom=page_height - y1,
                left=x0,
                right=page_width - x1,
                page_num=page_num,
                confidence=self._calculate_confidence(page, content_bbox)
            )
        else:
            # Default margins if no content detected
            margin = Margin(
                top=self.min_margin,
                bottom=self.min_margin,
                left=self.min_margin,
                right=self.min_margin,
                page_num=page_num,
                confidence=0.5
            )
        
        # Ensure minimum margins
        margin = self._enforce_minimum_margins(margin)
        
        return margin
    
    def detect_content_boundaries(self, page) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect actual content boundaries on a page.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            Tuple of (x0, y0, x1, y1) representing content bbox
        """
        # Collect all content elements
        all_bboxes = []
        
        # Get text characters
        chars = page.chars
        for char in chars:
            if char.get('text', '').strip():  # Skip whitespace
                all_bboxes.append((
                    char['x0'],
                    char['top'],
                    char['x1'],
                    char['bottom']
                ))
        
        # Get lines
        lines = page.lines
        for line in lines:
            all_bboxes.append((
                min(line['x0'], line['x1']),
                min(line['top'], line['bottom']),
                max(line['x0'], line['x1']),
                max(line['top'], line['bottom'])
            ))
        
        # Get rectangles
        rects = page.rects
        for rect in rects:
            all_bboxes.append((
                rect['x0'],
                rect['top'],
                rect['x1'],
                rect['bottom']
            ))
        
        # Get tables
        tables = page.find_tables()
        for table in tables:
            if table.bbox:
                all_bboxes.append(table.bbox)
        
        # Get images
        images = page.images
        for image in images:
            all_bboxes.append((
                image['x0'],
                image['top'],
                image['x1'],
                image['bottom']
            ))
        
        if not all_bboxes:
            return None
        
        # Find overall bounding box
        x0 = min(bbox[0] for bbox in all_bboxes)
        y0 = min(bbox[1] for bbox in all_bboxes)
        x1 = max(bbox[2] for bbox in all_bboxes)
        y1 = max(bbox[3] for bbox in all_bboxes)
        
        # Apply threshold to ignore outliers
        x0, y0, x1, y1 = self._remove_outliers(all_bboxes, (x0, y0, x1, y1))
        
        return (x0, y0, x1, y1)
    
    def _remove_outliers(self, bboxes: List[Tuple], 
                        initial_bbox: Tuple) -> Tuple[float, float, float, float]:
        """
        Remove outlier elements that might skew margin detection.
        
        Args:
            bboxes: List of all bounding boxes
            initial_bbox: Initial combined bbox
            
        Returns:
            Adjusted bbox without outliers
        """
        if len(bboxes) < 10:
            return initial_bbox
        
        # Calculate statistics for each edge
        x0_values = [bbox[0] for bbox in bboxes]
        y0_values = [bbox[1] for bbox in bboxes]
        x1_values = [bbox[2] for bbox in bboxes]
        y1_values = [bbox[3] for bbox in bboxes]
        
        # Use percentiles to remove outliers
        x0 = np.percentile(x0_values, 5)  # 5th percentile for left
        y0 = np.percentile(y0_values, 5)  # 5th percentile for top
        x1 = np.percentile(x1_values, 95)  # 95th percentile for right
        y1 = np.percentile(y1_values, 95)  # 95th percentile for bottom
        
        return (x0, y0, x1, y1)
    
    def _calculate_confidence(self, page, content_bbox: Tuple) -> float:
        """
        Calculate confidence score for margin detection.
        
        Args:
            page: Page object
            content_bbox: Detected content bbox
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Check if content is too close to edges
        x0, y0, x1, y1 = content_bbox
        
        if x0 < self.threshold:
            confidence -= 0.2
        if y0 < self.threshold:
            confidence -= 0.2
        if page.width - x1 < self.threshold:
            confidence -= 0.2
        if page.height - y1 < self.threshold:
            confidence -= 0.2
        
        # Check content density
        chars = page.chars
        if len(chars) < 10:
            confidence -= 0.3
        
        return max(0.1, confidence)
    
    def _find_consistent_margins(self, all_margins: List[Margin]) -> Margin:
        """
        Find most common margins across pages.
        
        Uses statistical mode to find consistent margins,
        important for documents with varying content per page.
        
        Args:
            all_margins: List of margins from all pages
            
        Returns:
            Most common margin values
        """
        if not all_margins:
            return Margin(
                top=self.min_margin,
                bottom=self.min_margin,
                left=self.min_margin,
                right=self.min_margin,
                page_num=-1
            )
        
        # Collect all values
        tops = [m.top for m in all_margins if m.confidence > 0.5]
        bottoms = [m.bottom for m in all_margins if m.confidence > 0.5]
        lefts = [m.left for m in all_margins if m.confidence > 0.5]
        rights = [m.right for m in all_margins if m.confidence > 0.5]
        
        # Find mode (most common value) with rounding
        def find_mode_with_tolerance(values: List[float], tolerance: float = 5) -> float:
            """Find mode with tolerance for slight variations"""
            if not values:
                return self.min_margin
            
            # Round to nearest tolerance
            rounded = [round(v / tolerance) * tolerance for v in values]
            
            try:
                return statistics.mode(rounded)
            except statistics.StatisticsError:
                # No unique mode, use median
                return statistics.median(values)
        
        return Margin(
            top=find_mode_with_tolerance(tops),
            bottom=find_mode_with_tolerance(bottoms),
            left=find_mode_with_tolerance(lefts),
            right=find_mode_with_tolerance(rights),
            page_num=-1,  # Indicates global margin
            confidence=1.0
        )
    
    def _apply_consistent_margins(self, margins: List[Margin], 
                                 consistent: Margin) -> List[Margin]:
        """
        Apply consistent margins to all pages.
        
        Args:
            margins: Original margins
            consistent: Consistent margin to apply
            
        Returns:
            Updated margins
        """
        updated = []
        
        for margin in margins:
            # Only update if confidence is low
            if margin.confidence < 0.7:
                updated.append(Margin(
                    top=consistent.top,
                    bottom=consistent.bottom,
                    left=consistent.left,
                    right=consistent.right,
                    page_num=margin.page_num,
                    confidence=margin.confidence
                ))
            else:
                # Keep original if high confidence
                updated.append(margin)
        
        return updated
    
    def _enforce_minimum_margins(self, margin: Margin) -> Margin:
        """Ensure margins meet minimum requirements"""
        
        return Margin(
            top=max(margin.top, self.min_margin),
            bottom=max(margin.bottom, self.min_margin),
            left=max(margin.left, self.min_margin),
            right=max(margin.right, self.min_margin),
            page_num=margin.page_num,
            confidence=margin.confidence
        )
    
    def enforce_margins(self, content: Dict, margins: Margin) -> Dict:
        """
        Ensure translated content respects margins.
        
        Args:
            content: Content with position data
            margins: Margins to enforce
            
        Returns:
            Adjusted content
        """
        adjusted = content.copy()
        violations = []
        
        for element in adjusted.get('elements', []):
            bbox = element.get('bbox', {})
            
            # Check violations
            if bbox.get('x', 0) < margins.left:
                violations.append(f"Element {element.get('id')} violates left margin")
                bbox['x'] = margins.left
            
            if bbox.get('y', 0) < margins.top:
                violations.append(f"Element {element.get('id')} violates top margin")
                bbox['y'] = margins.top
            
            # Check right margin (need page width)
            page_width = content.get('page_width', 612)  # Letter size default
            if bbox.get('x', 0) + bbox.get('width', 0) > page_width - margins.right:
                violations.append(f"Element {element.get('id')} violates right margin")
                # Adjust width or position
                max_width = page_width - margins.right - bbox['x']
                bbox['width'] = min(bbox.get('width', 0), max_width)
            
            # Check bottom margin
            page_height = content.get('page_height', 792)  # Letter size default
            if bbox.get('y', 0) + bbox.get('height', 0) > page_height - margins.bottom:
                violations.append(f"Element {element.get('id')} violates bottom margin")
                # Adjust height or position
                max_height = page_height - margins.bottom - bbox['y']
                bbox['height'] = min(bbox.get('height', 0), max_height)
        
        if violations:
            adjusted['margin_violations'] = violations
            print(f"Margin violations detected: {len(violations)}")
        
        return adjusted
    
    def get_safe_area(self, page_size: Tuple[float, float], 
                     margins: Margin) -> Dict:
        """
        Get safe content area within margins.
        
        Args:
            page_size: (width, height) tuple
            margins: Margin object
            
        Returns:
            Safe area dictionary
        """
        width, height = page_size
        
        return {
            'x': margins.left,
            'y': margins.top,
            'width': width - margins.left - margins.right,
            'height': height - margins.top - margins.bottom,
            'total_area': (width - margins.left - margins.right) * 
                         (height - margins.top - margins.bottom)
        }
    
    def calculate_margin_ratio(self, page_size: Tuple[float, float], 
                              margins: Margin) -> Dict:
        """
        Calculate margin ratios for analysis.
        
        Args:
            page_size: (width, height) tuple
            margins: Margin object
            
        Returns:
            Ratio dictionary
        """
        width, height = page_size
        
        return {
            'top_ratio': margins.top / height,
            'bottom_ratio': margins.bottom / height,
            'left_ratio': margins.left / width,
            'right_ratio': margins.right / width,
            'content_area_ratio': ((width - margins.left - margins.right) * 
                                 (height - margins.top - margins.bottom)) / 
                                (width * height)
        }
    
    def suggest_margin_adjustments(self, margins: List[Margin]) -> Dict:
        """
        Suggest margin adjustments for consistency.
        
        Args:
            margins: List of margins from all pages
            
        Returns:
            Suggestions dictionary
        """
        suggestions = {
            'inconsistent_pages': [],
            'recommended_margins': None,
            'adjustments': []
        }
        
        # Find consistent margins
        consistent = self._find_consistent_margins(margins)
        suggestions['recommended_margins'] = {
            'top': consistent.top,
            'bottom': consistent.bottom,
            'left': consistent.left,
            'right': consistent.right
        }
        
        # Find pages that deviate
        for margin in margins:
            deviation = max(
                abs(margin.top - consistent.top),
                abs(margin.bottom - consistent.bottom),
                abs(margin.left - consistent.left),
                abs(margin.right - consistent.right)
            )
            
            if deviation > self.threshold:
                suggestions['inconsistent_pages'].append({
                    'page': margin.page_num,
                    'deviation': deviation,
                    'current': {
                        'top': margin.top,
                        'bottom': margin.bottom,
                        'left': margin.left,
                        'right': margin.right
                    }
                })
        
        return suggestions
```

## Usage Example

```python
# Initialize manager
margin_manager = MarginManager(config={
    'threshold': 10,
    'enforce_strict': True,
    'min_margin': 36
})

# Extract margins
margins = margin_manager.extract_margins('document.pdf')

# Print margins for each page
for margin in margins:
    print(f"Page {margin.page_num}: "
          f"Top={margin.top:.1f}, Bottom={margin.bottom:.1f}, "
          f"Left={margin.left:.1f}, Right={margin.right:.1f}")

# Enforce margins on content
adjusted_content = margin_manager.enforce_margins(content, margins[0])

# Get safe area
safe_area = margin_manager.get_safe_area((612, 792), margins[0])
print(f"Safe content area: {safe_area['width']:.1f} x {safe_area['height']:.1f}")
```
