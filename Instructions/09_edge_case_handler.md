# Edge Case Handler - Complete Implementation

## Overview
Comprehensive handler for document formatting edge cases.

## Implementation

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

class EdgeCaseType(Enum):
    """Edge case types"""
    ROTATED_TEXT = "rotated_text"
    VERTICAL_TEXT = "vertical_text"
    MULTI_COLUMN = "multi_column"
    FOOTNOTES = "footnotes"
    DROP_CAPS = "drop_caps"
    TEXT_IN_SHAPES = "text_in_shapes"
    FORM_FIELDS = "form_fields"
    ANNOTATIONS = "annotations"
    HYPERLINKS = "hyperlinks"
    PAGE_NUMBERS = "page_numbers"
    HEADERS_FOOTERS = "headers_footers"
    BOOKMARKS = "bookmarks"
    TABLE_OF_CONTENTS = "toc"
    CROSSED_OUT = "crossed_out"
    HIGHLIGHTED = "highlighted"

@dataclass
class EdgeCase:
    """Edge case detection result"""
    type: str
    element_id: str
    confidence: float
    metadata: Dict
    handling_strategy: str

class EdgeCaseHandler:
    """
    Detect and handle document formatting edge cases.
    Ensures accurate preservation of special formatting.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.detection_threshold = self.config.get('detection_threshold', 0.7)
        
        # Handler mapping
        self.handlers = {
            EdgeCaseType.ROTATED_TEXT: self.handle_rotated_text,
            EdgeCaseType.VERTICAL_TEXT: self.handle_vertical_text,
            EdgeCaseType.MULTI_COLUMN: self.handle_multi_column,
            EdgeCaseType.FOOTNOTES: self.handle_footnotes,
            EdgeCaseType.DROP_CAPS: self.handle_drop_caps,
            EdgeCaseType.FORM_FIELDS: self.handle_form_fields,
            EdgeCaseType.ANNOTATIONS: self.handle_annotations,
            EdgeCaseType.HYPERLINKS: self.handle_hyperlinks,
            EdgeCaseType.PAGE_NUMBERS: self.handle_page_numbers,
            EdgeCaseType.TEXT_IN_SHAPES: self.handle_text_in_shapes
        }
    
    def detect_edge_cases(self, page_data: Dict) -> List[EdgeCase]:
        """
        Detect all edge cases in page.
        
        Args:
            page_data: Page extraction data
            
        Returns:
            List of detected edge cases
        """
        edge_cases = []
        
        # Run all detectors
        edge_cases.extend(self._detect_rotated_text(page_data))
        edge_cases.extend(self._detect_vertical_text(page_data))
        edge_cases.extend(self._detect_multi_column(page_data))
        edge_cases.extend(self._detect_footnotes(page_data))
        edge_cases.extend(self._detect_drop_caps(page_data))
        edge_cases.extend(self._detect_form_fields(page_data))
        edge_cases.extend(self._detect_page_numbers(page_data))
        edge_cases.extend(self._detect_hyperlinks(page_data))
        
        return edge_cases
    
    def handle_rotated_text(self, element: Dict) -> Dict:
        """Handle rotated text elements"""
        
        rotation = element.get('rotation', 0)
        transform_matrix = element.get('transform')
        
        # Extract rotation angle from transformation matrix
        if transform_matrix:
            angle = np.arctan2(transform_matrix[1], transform_matrix[0]) * 180 / np.pi
        else:
            angle = rotation
        
        return {
            'action': 'preserve_rotation',
            'angle': angle,
            'transform': transform_matrix,
            'translate': True,
            'reconstruction': 'apply_rotation'
        }
    
    def handle_vertical_text(self, element: Dict) -> Dict:
        """Handle vertical text (CJK, etc.)"""
        
        writing_mode = element.get('writing_mode', 'horizontal-tb')
        
        return {
            'action': 'preserve_vertical',
            'writing_mode': writing_mode,
            'direction': element.get('direction', 'tb-rl'),
            'translate': True,
            'special_handling': 'vertical_layout'
        }
    
    def handle_multi_column(self, page_data: Dict) -> Dict:
        """Handle multi-column layouts"""
        
        columns = self._detect_columns(page_data)
        
        return {
            'action': 'preserve_columns',
            'column_count': len(columns),
            'column_boundaries': columns,
            'reading_order': self._determine_reading_order(columns),
            'gutter_width': self._calculate_gutter_width(columns)
        }
    
    def handle_footnotes(self, element: Dict) -> Dict:
        """Handle footnotes with references"""
        
        return {
            'action': 'preserve_footnote',
            'marker': element.get('marker', '*'),
            'reference_id': element.get('ref_id'),
            'note_id': element.get('note_id'),
            'translate': True,
            'preserve_numbering': True,
            'link_to_reference': True
        }
    
    def handle_drop_caps(self, element: Dict) -> Dict:
        """Handle drop capitals"""
        
        return {
            'action': 'preserve_drop_cap',
            'initial_char': element.get('char', ''),
            'size_ratio': element.get('size_ratio', 3),
            'lines_occupied': element.get('lines', 3),
            'translate': False,  # Usually don't translate single letters
            'style_preservation': 'exact'
        }
    
    def handle_form_fields(self, element: Dict) -> Dict:
        """Handle interactive form fields"""
        
        field_type = element.get('field_type', 'text')
        
        return {
            'action': 'preserve_form_field',
            'field_type': field_type,
            'field_name': element.get('name'),
            'field_value': element.get('value'),
            'required': element.get('required', False),
            'translate_label': True,
            'preserve_interactivity': True,
            'validation': element.get('validation')
        }
    
    def handle_annotations(self, element: Dict) -> Dict:
        """Handle PDF annotations"""
        
        annotation_type = element.get('annotation_type', 'note')
        
        return {
            'action': 'preserve_annotation',
            'type': annotation_type,
            'content': element.get('content'),
            'author': element.get('author'),
            'timestamp': element.get('timestamp'),
            'translate_content': annotation_type in ['note', 'comment'],
            'preserve_position': True,
            'color': element.get('color'),
            'icon': element.get('icon')
        }
    
    def handle_hyperlinks(self, element: Dict) -> Dict:
        """Handle hyperlinks"""
        
        return {
            'action': 'preserve_hyperlink',
            'url': element.get('url'),
            'display_text': element.get('text'),
            'translate_display': True,
            'preserve_url': True,
            'link_type': element.get('link_type', 'external')
        }
    
    def handle_page_numbers(self, element: Dict) -> Dict:
        """Handle page numbering"""
        
        return {
            'action': 'update_page_number',
            'format': element.get('format', 'numeric'),
            'position': element.get('position', 'bottom-center'),
            'prefix': element.get('prefix', ''),
            'suffix': element.get('suffix', ''),
            'translate': False,
            'update_dynamically': True
        }
    
    def handle_text_in_shapes(self, element: Dict) -> Dict:
        """Handle text within shapes"""
        
        shape_type = element.get('shape_type', 'rectangle')
        
        return {
            'action': 'preserve_shape_text',
            'shape_type': shape_type,
            'shape_path': element.get('path'),
            'text_alignment': element.get('alignment', 'center'),
            'translate': True,
            'fit_strategy': 'resize_text' if shape_type != 'path' else 'follow_path'
        }
    
    def _detect_rotated_text(self, page_data: Dict) -> List[EdgeCase]:
        """Detect rotated text elements"""
        
        edge_cases = []
        
        for element in page_data.get('text_elements', []):
            transform = element.get('transform')
            if transform:
                # Check for rotation in transformation matrix
                angle = np.arctan2(transform[1], transform[0]) * 180 / np.pi
                if abs(angle) > 5:  # More than 5 degrees
                    edge_cases.append(EdgeCase(
                        type=EdgeCaseType.ROTATED_TEXT.value,
                        element_id=element['id'],
                        confidence=0.95,
                        metadata={'angle': angle, 'transform': transform},
                        handling_strategy='preserve_rotation'
                    ))
        
        return edge_cases
    
    def _detect_vertical_text(self, page_data: Dict) -> List[EdgeCase]:
        """Detect vertical text (CJK)"""
        
        edge_cases = []
        
        for element in page_data.get('text_elements', []):
            chars = element.get('chars', [])
            if len(chars) > 1:
                # Check if characters progress vertically
                y_positions = [c['y'] for c in chars]
                x_positions = [c['x'] for c in chars]
                
                y_variance = np.var(y_positions)
                x_variance = np.var(x_positions)
                
                if y_variance > x_variance * 3:  # Vertical progression
                    edge_cases.append(EdgeCase(
                        type=EdgeCaseType.VERTICAL_TEXT.value,
                        element_id=element['id'],
                        confidence=0.9,
                        metadata={'direction': 'vertical'},
                        handling_strategy='preserve_vertical'
                    ))
        
        return edge_cases
    
    def _detect_multi_column(self, page_data: Dict) -> List[EdgeCase]:
        """Detect multi-column layout"""
        
        edge_cases = []
        text_blocks = page_data.get('text_blocks', [])
        
        if len(text_blocks) > 5:
            # Cluster x-positions
            x_positions = [block['bbox']['x'] for block in text_blocks]
            columns = self._cluster_positions(x_positions, threshold=50)
            
            if len(columns) > 1:
                edge_cases.append(EdgeCase(
                    type=EdgeCaseType.MULTI_COLUMN.value,
                    element_id='page',
                    confidence=0.85,
                    metadata={'column_count': len(columns), 'columns': columns},
                    handling_strategy='preserve_columns'
                ))
        
        return edge_cases
    
    def _detect_footnotes(self, page_data: Dict) -> List[EdgeCase]:
        """Detect footnotes"""
        
        edge_cases = []
        page_height = page_data.get('height', 800)
        
        for element in page_data.get('text_elements', []):
            y = element['bbox']['y']
            font_size = element.get('font_size', 12)
            text = element.get('text', '')
            
            # Bottom 20% of page, small font, starts with marker
            if (y > page_height * 0.8 and 
                font_size < 10 and 
                any(text.startswith(m) for m in ['*', '†', '‡', '§', '¶', '1', '2', '3'])):
                
                edge_cases.append(EdgeCase(
                    type=EdgeCaseType.FOOTNOTES.value,
                    element_id=element['id'],
                    confidence=0.8,
                    metadata={'marker': text[0], 'font_size': font_size},
                    handling_strategy='preserve_footnote'
                ))
        
        return edge_cases
    
    def _detect_drop_caps(self, page_data: Dict) -> List[EdgeCase]:
        """Detect drop capitals"""
        
        edge_cases = []
        
        for block in page_data.get('text_blocks', []):
            if block.get('is_paragraph'):
                first_char = block.get('first_char')
                if first_char:
                    char_size = first_char.get('font_size', 12)
                    avg_size = block.get('avg_font_size', 12)
                    
                    if char_size > avg_size * 2.5:
                        edge_cases.append(EdgeCase(
                            type=EdgeCaseType.DROP_CAPS.value,
                            element_id=block['id'],
                            confidence=0.9,
                            metadata={'char': first_char['text'], 'size_ratio': char_size / avg_size},
                            handling_strategy='preserve_drop_cap'
                        ))
        
        return edge_cases
    
    def _detect_form_fields(self, page_data: Dict) -> List[EdgeCase]:
        """Detect form fields"""
        
        edge_cases = []
        
        for field in page_data.get('form_fields', []):
            edge_cases.append(EdgeCase(
                type=EdgeCaseType.FORM_FIELDS.value,
                element_id=field['id'],
                confidence=1.0,
                metadata=field,
                handling_strategy='preserve_form_field'
            ))
        
        return edge_cases
    
    def _detect_page_numbers(self, page_data: Dict) -> List[EdgeCase]:
        """Detect page numbers"""
        
        edge_cases = []
        page_height = page_data.get('height', 800)
        
        for element in page_data.get('text_elements', []):
            y = element['bbox']['y']
            text = element.get('text', '').strip()
            
            # Bottom or top of page, short numeric text
            if ((y > page_height * 0.9 or y < page_height * 0.1) and 
                len(text) < 10 and 
                any(char.isdigit() for char in text)):
                
                edge_cases.append(EdgeCase(
                    type=EdgeCaseType.PAGE_NUMBERS.value,
                    element_id=element['id'],
                    confidence=0.7,
                    metadata={'text': text, 'position': 'bottom' if y > page_height * 0.5 else 'top'},
                    handling_strategy='update_page_number'
                ))
        
        return edge_cases
    
    def _detect_hyperlinks(self, page_data: Dict) -> List[EdgeCase]:
        """Detect hyperlinks"""
        
        edge_cases = []
        
        for link in page_data.get('links', []):
            edge_cases.append(EdgeCase(
                type=EdgeCaseType.HYPERLINKS.value,
                element_id=link['id'],
                confidence=1.0,
                metadata=link,
                handling_strategy='preserve_hyperlink'
            ))
        
        return edge_cases
    
    def _detect_columns(self, page_data: Dict) -> List[Tuple[float, float]]:
        """Detect column boundaries"""
        
        text_blocks = page_data.get('text_blocks', [])
        if not text_blocks:
            return []
        
        x_positions = [block['bbox']['x'] for block in text_blocks]
        return self._cluster_positions(x_positions, threshold=50)
    
    def _cluster_positions(self, positions: List[float], threshold: float) -> List[Tuple[float, float]]:
        """Cluster 1D positions"""
        
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = []
        current_cluster = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] > threshold:
                clusters.append((min(current_cluster), max(current_cluster)))
                current_cluster = [pos]
            else:
                current_cluster.append(pos)
        
        clusters.append((min(current_cluster), max(current_cluster)))
        return clusters
    
    def _determine_reading_order(self, columns: List[Tuple[float, float]]) -> List[int]:
        """Determine column reading order"""
        
        # Default: left to right
        return list(range(len(columns)))
    
    def _calculate_gutter_width(self, columns: List[Tuple[float, float]]) -> float:
        """Calculate gutter width between columns"""
        
        if len(columns) < 2:
            return 0
        
        gutters = []
        for i in range(len(columns) - 1):
            gutter = columns[i+1][0] - columns[i][1]
            gutters.append(gutter)
        
        return np.mean(gutters) if gutters else 0
    
    def apply_strategies(self, page_data: Dict, edge_cases: List[EdgeCase]) -> Dict:
        """
        Apply handling strategies to page data.
        
        Args:
            page_data: Original page data
            edge_cases: Detected edge cases
            
        Returns:
            Enhanced page data with edge case handling
        """
        enhanced = page_data.copy()
        enhanced['edge_cases'] = []
        
        for edge_case in edge_cases:
            # Get handler
            handler = self.handlers.get(EdgeCaseType[edge_case.type.upper()])
            if handler:
                # Find element
                element = self._find_element(page_data, edge_case.element_id)
                if element:
                    # Apply handler
                    handling = handler(element)
                    
                    # Store handling instructions
                    enhanced['edge_cases'].append({
                        'type': edge_case.type,
                        'element_id': edge_case.element_id,
                        'handling': handling
                    })
        
        return enhanced
    
    def _find_element(self, page_data: Dict, element_id: str) -> Optional[Dict]:
        """Find element by ID"""
        
        # Search in different element types
        for element_list in ['text_elements', 'text_blocks', 'form_fields', 'links']:
            for element in page_data.get(element_list, []):
                if element.get('id') == element_id:
                    return element
        
        return None
```

## Usage Example

```python
# Initialize handler
handler = EdgeCaseHandler()

# Detect edge cases
edge_cases = handler.detect_edge_cases(page_data)

# Apply handling strategies
enhanced_page = handler.apply_strategies(page_data, edge_cases)

# Process each edge case
for edge_case in edge_cases:
    print(f"Detected: {edge_case.type}")
    print(f"Confidence: {edge_case.confidence}")
    print(f"Strategy: {edge_case.handling_strategy}")
```
