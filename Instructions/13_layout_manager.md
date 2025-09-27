# Layout Manager - Complete Implementation

## Overview
Analyzes and preserves document layout relationships, including text wrapping, overlays, and spatial arrangements.

## Implementation

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import cv2
from sklearn.cluster import DBSCAN

class LayoutRelationship(Enum):
    """Types of layout relationships"""
    TEXT_WRAP = "text_wrap"
    OVERLAY = "overlay"
    CAPTION = "caption"
    SIDE_BY_SIDE = "side_by_side"
    COLUMN = "column"
    HEADER_FOOTER = "header_footer"
    MARGIN_NOTE = "margin_note"
    FLOATING = "floating"

@dataclass
class LayoutElement:
    """Layout element structure"""
    id: str
    type: str  # text, image, table, etc.
    bbox: Dict[str, float]  # x, y, width, height
    content: Optional[str] = None
    relationships: List[Dict] = None
    z_index: int = 0  # Layer order

class LayoutManager:
    """
    Manage document layout and spatial relationships.
    Ensures preservation of complex layouts during translation.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.column_threshold = self.config.get('column_threshold', 0.8)
        self.wrap_distance = self.config.get('wrap_distance', 20)
        self.overlay_opacity_threshold = self.config.get('overlay_opacity', 0.5)
        
    def analyze_layout(self, page_content: Dict) -> Dict:
        """
        Analyze complete page layout.
        
        Args:
            page_content: Extracted page content
            
        Returns:
            Layout analysis dictionary
        """
        # Convert to layout elements
        elements = self._create_layout_elements(page_content)
        
        # Detect layout structure
        layout = {
            'elements': elements,
            'columns': self._detect_columns(elements),
            'relationships': self._analyze_relationships(elements),
            'reading_order': self._determine_reading_order(elements),
            'layers': self._detect_layers(elements),
            'flow_type': self._determine_flow_type(elements)
        }
        
        # Detect special layouts
        layout['special_layouts'] = self._detect_special_layouts(elements)
        
        return layout
    
    def _create_layout_elements(self, page_content: Dict) -> List[LayoutElement]:
        """Convert page content to layout elements"""
        
        elements = []
        element_id = 0
        
        # Text blocks
        for block in page_content.get('text_blocks', []):
            elements.append(LayoutElement(
                id=f"text_{element_id}",
                type='text',
                bbox=block.get('bbox', {}),
                content=block.get('text', ''),
                z_index=0
            ))
            element_id += 1
        
        # Images
        for img in page_content.get('images', []):
            elements.append(LayoutElement(
                id=f"img_{element_id}",
                type='image',
                bbox=img.get('bbox', {}),
                z_index=img.get('z_index', 0)
            ))
            element_id += 1
        
        # Tables
        for table in page_content.get('tables', []):
            elements.append(LayoutElement(
                id=f"table_{element_id}",
                type='table',
                bbox=table.get('bbox', {}),
                z_index=0
            ))
            element_id += 1
        
        return elements
    
    def _detect_columns(self, elements: List[LayoutElement]) -> List[Dict]:
        """Detect column layout"""
        
        text_elements = [e for e in elements if e.type == 'text']
        
        if len(text_elements) < 3:
            return []
        
        # Extract x-coordinates
        x_coords = np.array([[e.bbox.get('x', 0)] for e in text_elements])
        
        # Cluster x-coordinates to find columns
        if len(x_coords) > 1:
            clustering = DBSCAN(eps=50, min_samples=2).fit(x_coords)
            labels = clustering.labels_
            
            # Group by cluster
            columns = []
            for label in set(labels):
                if label != -1:  # Ignore noise
                    column_elements = [
                        text_elements[i] for i, l in enumerate(labels) if l == label
                    ]
                    
                    # Calculate column bounds
                    x_values = [e.bbox['x'] for e in column_elements]
                    y_values = [e.bbox['y'] for e in column_elements]
                    widths = [e.bbox.get('width', 0) for e in column_elements]
                    heights = [e.bbox.get('height', 0) for e in column_elements]
                    
                    columns.append({
                        'id': f"column_{label}",
                        'bounds': {
                            'x': min(x_values),
                            'y': min(y_values),
                            'width': max(x_values[i] + widths[i] for i in range(len(x_values))) - min(x_values),
                            'height': max(y_values[i] + heights[i] for i in range(len(y_values))) - min(y_values)
                        },
                        'elements': [e.id for e in column_elements],
                        'alignment': self._detect_alignment(column_elements)
                    })
            
            return columns
        
        return []
    
    def _analyze_relationships(self, elements: List[LayoutElement]) -> List[Dict]:
        """Analyze relationships between elements"""
        
        relationships = []
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                # Check for text wrapping
                if self._is_text_wrapping(elem1, elem2):
                    relationships.append({
                        'type': LayoutRelationship.TEXT_WRAP.value,
                        'element1': elem1.id,
                        'element2': elem2.id,
                        'wrap_side': self._get_wrap_side(elem1, elem2)
                    })
                
                # Check for overlays
                if self._is_overlay(elem1, elem2):
                    relationships.append({
                        'type': LayoutRelationship.OVERLAY.value,
                        'top': elem1.id if elem1.z_index > elem2.z_index else elem2.id,
                        'bottom': elem2.id if elem1.z_index > elem2.z_index else elem1.id,
                        'opacity': self._calculate_overlay_opacity(elem1, elem2)
                    })
                
                # Check for captions
                if self._is_caption(elem1, elem2):
                    relationships.append({
                        'type': LayoutRelationship.CAPTION.value,
                        'image': elem1.id if elem1.type == 'image' else elem2.id,
                        'caption': elem2.id if elem1.type == 'image' else elem1.id
                    })
                
                # Check for side-by-side
                if self._is_side_by_side(elem1, elem2):
                    relationships.append({
                        'type': LayoutRelationship.SIDE_BY_SIDE.value,
                        'left': elem1.id if elem1.bbox['x'] < elem2.bbox['x'] else elem2.id,
                        'right': elem2.id if elem1.bbox['x'] < elem2.bbox['x'] else elem1.id
                    })
        
        return relationships
    
    def _is_text_wrapping(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """Check if text wraps around element"""
        
        # Text wrapping typically involves text and image
        if not ((elem1.type == 'text' and elem2.type == 'image') or 
                (elem1.type == 'image' and elem2.type == 'text')):
            return False
        
        text_elem = elem1 if elem1.type == 'text' else elem2
        img_elem = elem2 if elem1.type == 'text' else elem1
        
        # Check if text is close to image
        distance = self._calculate_distance(text_elem.bbox, img_elem.bbox)
        
        # Check if text flows around image
        if distance < self.wrap_distance:
            # Check if text bbox partially overlaps with image horizontally
            text_x_range = (text_elem.bbox['x'], text_elem.bbox['x'] + text_elem.bbox.get('width', 0))
            img_x_range = (img_elem.bbox['x'], img_elem.bbox['x'] + img_elem.bbox.get('width', 0))
            
            # Check for horizontal overlap
            if (text_x_range[0] <= img_x_range[0] <= text_x_range[1] or
                text_x_range[0] <= img_x_range[1] <= text_x_range[1]):
                return True
        
        return False
    
    def _is_overlay(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """Check if elements overlay"""
        
        # Calculate intersection
        intersection = self._calculate_intersection(elem1.bbox, elem2.bbox)
        
        if intersection > 0:
            # Check if significant overlay
            area1 = elem1.bbox.get('width', 0) * elem1.bbox.get('height', 0)
            area2 = elem2.bbox.get('width', 0) * elem2.bbox.get('height', 0)
            
            overlap_ratio = intersection / min(area1, area2)
            return overlap_ratio > 0.1
        
        return False
    
    def _is_caption(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """Check if one element is caption for another"""
        
        # Caption typically involves image/table and text
        if not ((elem1.type in ['image', 'table'] and elem2.type == 'text') or
                (elem2.type in ['image', 'table'] and elem1.type == 'text')):
            return False
        
        text_elem = elem1 if elem1.type == 'text' else elem2
        visual_elem = elem2 if elem1.type == 'text' else elem1
        
        # Caption is usually below or above visual element
        text_center_x = text_elem.bbox['x'] + text_elem.bbox.get('width', 0) / 2
        visual_center_x = visual_elem.bbox['x'] + visual_elem.bbox.get('width', 0) / 2
        
        # Check horizontal alignment
        if abs(text_center_x - visual_center_x) < visual_elem.bbox.get('width', 0) / 2:
            # Check vertical proximity
            vertical_distance = abs(text_elem.bbox['y'] - 
                                  (visual_elem.bbox['y'] + visual_elem.bbox.get('height', 0)))
            
            if vertical_distance < 30:  # Within 30 pixels
                # Check for caption keywords
                if text_elem.content:
                    caption_keywords = ['figure', 'fig.', 'table', 'image', '图', '表']
                    if any(keyword in text_elem.content.lower() for keyword in caption_keywords):
                        return True
        
        return False
    
    def _is_side_by_side(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """Check if elements are side by side"""
        
        # Check vertical overlap
        y1_range = (elem1.bbox['y'], elem1.bbox['y'] + elem1.bbox.get('height', 0))
        y2_range = (elem2.bbox['y'], elem2.bbox['y'] + elem2.bbox.get('height', 0))
        
        # Calculate vertical overlap
        overlap_start = max(y1_range[0], y2_range[0])
        overlap_end = min(y1_range[1], y2_range[1])
        
        if overlap_end > overlap_start:
            vertical_overlap = overlap_end - overlap_start
            min_height = min(elem1.bbox.get('height', 1), elem2.bbox.get('height', 1))
            
            # If significant vertical overlap
            if vertical_overlap / min_height > 0.5:
                # Check horizontal separation
                x1_end = elem1.bbox['x'] + elem1.bbox.get('width', 0)
                x2_start = elem2.bbox['x']
                x2_end = elem2.bbox['x'] + elem2.bbox.get('width', 0)
                x1_start = elem1.bbox['x']
                
                horizontal_gap = min(abs(x2_start - x1_end), abs(x1_start - x2_end))
                
                # Side by side if small horizontal gap
                return horizontal_gap < 50
        
        return False
    
    def _determine_reading_order(self, elements: List[LayoutElement]) -> List[str]:
        """Determine reading order of elements"""
        
        if not elements:
            return []
        
        # Sort by position (top to bottom, left to right)
        sorted_elements = sorted(
            elements,
            key=lambda e: (
                e.bbox.get('y', 0) // 50,  # Group by rows (50px tolerance)
                e.bbox.get('x', 0)  # Then by x position
            )
        )
        
        return [e.id for e in sorted_elements]
    
    def _detect_layers(self, elements: List[LayoutElement]) -> List[List[str]]:
        """Detect element layers (z-order)"""
        
        # Group by z-index
        layers = {}
        for elem in elements:
            z = elem.z_index
            if z not in layers:
                layers[z] = []
            layers[z].append(elem.id)
        
        # Return sorted layers
        return [layers[z] for z in sorted(layers.keys())]
    
    def _determine_flow_type(self, elements: List[LayoutElement]) -> str:
        """Determine document flow type"""
        
        columns = self._detect_columns(elements)
        
        if len(columns) > 1:
            return 'multi-column'
        elif self._has_complex_wrapping(elements):
            return 'magazine'
        elif self._is_form_layout(elements):
            return 'form'
        else:
            return 'single-column'
    
    def _detect_special_layouts(self, elements: List[LayoutElement]) -> Dict:
        """Detect special layout patterns"""
        
        special = {
            'has_sidebar': self._detect_sidebar(elements),
            'has_pullquotes': self._detect_pullquotes(elements),
            'has_margin_notes': self._detect_margin_notes(elements),
            'has_floating_elements': self._detect_floating(elements)
        }
        
        return special
    
    def handle_overlays(self, text_elements: List[Dict], 
                       image_elements: List[Dict]) -> Dict:
        """
        Handle text-image overlays.
        
        Args:
            text_elements: Text elements with positions
            image_elements: Image elements with positions
            
        Returns:
            Overlay handling instructions
        """
        overlays = {
            'collisions': [],
            'watermarks': [],
            'adjustments': []
        }
        
        for text in text_elements:
            for img in image_elements:
                intersection = self._calculate_intersection(
                    text.get('bbox', {}),
                    img.get('bbox', {})
                )
                
                if intersection > 0:
                    # Check if watermark (low opacity)
                    if img.get('opacity', 1.0) < self.overlay_opacity_threshold:
                        overlays['watermarks'].append({
                            'text_id': text.get('id'),
                            'image_id': img.get('id'),
                            'action': 'preserve_overlay'
                        })
                    else:
                        # Collision that needs resolution
                        overlays['collisions'].append({
                            'text_id': text.get('id'),
                            'image_id': img.get('id'),
                            'intersection': intersection
                        })
                        
                        # Calculate adjustment
                        adjustment = self._calculate_overlay_adjustment(text, img)
                        overlays['adjustments'].append(adjustment)
        
        return overlays
    
    def maintain_relative_positions(self, source_layout: Dict, 
                                   translated_elements: List[Dict]) -> List[Dict]:
        """
        Maintain relative positions after translation.
        
        Args:
            source_layout: Original layout
            translated_elements: Elements with translations
            
        Returns:
            Adjusted elements
        """
        adjusted = []
        
        # Get original relationships
        relationships = source_layout.get('relationships', [])
        
        for element in translated_elements:
            # Find related elements
            related = self._find_related_elements(element['id'], relationships)
            
            # Adjust position based on relationships
            if related:
                adjusted_element = self._adjust_for_relationships(
                    element, related, translated_elements
                )
                adjusted.append(adjusted_element)
            else:
                adjusted.append(element)
        
        return adjusted
    
    # Helper methods
    
    def _calculate_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate distance between two bboxes"""
        
        center1 = (bbox1['x'] + bbox1.get('width', 0)/2,
                  bbox1['y'] + bbox1.get('height', 0)/2)
        center2 = (bbox2['x'] + bbox2.get('width', 0)/2,
                  bbox2['y'] + bbox2.get('height', 0)/2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_intersection(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate intersection area of two bboxes"""
        
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1.get('width', 0),
                bbox2['x'] + bbox2.get('width', 0))
        y2 = min(bbox1['y'] + bbox1.get('height', 0),
                bbox2['y'] + bbox2.get('height', 0))
        
        if x2 > x1 and y2 > y1:
            return (x2 - x1) * (y2 - y1)
        
        return 0
    
    def _get_wrap_side(self, text: LayoutElement, img: LayoutElement) -> str:
        """Determine which side text wraps around image"""
        
        if text.bbox['x'] < img.bbox['x']:
            return 'left'
        elif text.bbox['x'] > img.bbox['x'] + img.bbox.get('width', 0):
            return 'right'
        elif text.bbox['y'] < img.bbox['y']:
            return 'top'
        else:
            return 'bottom'
    
    def _calculate_overlay_opacity(self, elem1: LayoutElement, 
                                  elem2: LayoutElement) -> float:
        """Calculate effective overlay opacity"""
        
        # This would need actual opacity values from PDF
        # Using placeholder calculation
        return 0.5
    
    def _detect_alignment(self, elements: List[LayoutElement]) -> str:
        """Detect text alignment in elements"""
        
        if not elements:
            return 'left'
        
        x_positions = [e.bbox['x'] for e in elements]
        x_variance = np.var(x_positions) if len(x_positions) > 1 else 0
        
        if x_variance < 10:
            return 'left'
        
        # Check for center alignment
        widths = [e.bbox.get('width', 0) for e in elements]
        centers = [x_positions[i] + widths[i]/2 for i in range(len(elements))]
        center_variance = np.var(centers) if len(centers) > 1 else 0
        
        if center_variance < 10:
            return 'center'
        
        # Check for right alignment
        rights = [x_positions[i] + widths[i] for i in range(len(elements))]
        right_variance = np.var(rights) if len(rights) > 1 else 0
        
        if right_variance < 10:
            return 'right'
        
        return 'justified'
    
    def _has_complex_wrapping(self, elements: List[LayoutElement]) -> bool:
        """Check if layout has complex text wrapping"""
        
        text_elements = [e for e in elements if e.type == 'text']
        img_elements = [e for e in elements if e.type == 'image']
        
        wrap_count = 0
        for text in text_elements:
            for img in img_elements:
                if self._is_text_wrapping(text, img):
                    wrap_count += 1
        
        return wrap_count >= 2
    
    def _is_form_layout(self, elements: List[LayoutElement]) -> bool:
        """Check if layout resembles a form"""
        
        # Forms typically have many small text elements in grid pattern
        text_elements = [e for e in elements if e.type == 'text']
        
        if len(text_elements) < 10:
            return False
        
        # Check for grid-like arrangement
        x_positions = [e.bbox['x'] for e in text_elements]
        y_positions = [e.bbox['y'] for e in text_elements]
        
        # Check for regular spacing
        x_unique = len(set(x_positions))
        y_unique = len(set(y_positions))
        
        return x_unique < len(text_elements) / 2 and y_unique < len(text_elements) / 2
    
    def _detect_sidebar(self, elements: List[LayoutElement]) -> bool:
        """Detect sidebar layout"""
        
        # Sidebar is typically a narrow column on the side
        columns = self._detect_columns(elements)
        
        if len(columns) >= 2:
            widths = [col['bounds']['width'] for col in columns]
            min_width = min(widths)
            max_width = max(widths)
            
            # Sidebar if one column is much narrower
            return min_width < max_width * 0.4
        
        return False
    
    def _detect_pullquotes(self, elements: List[LayoutElement]) -> bool:
        """Detect pullquotes"""
        
        # Pullquotes are typically larger text elements surrounded by smaller text
        text_elements = [e for e in elements if e.type == 'text']
        
        for elem in text_elements:
            # Check for larger font size (would need actual font info)
            # Using bbox height as proxy
            if elem.bbox.get('height', 0) > 50:
                # Check if surrounded by text
                surrounding = self._count_surrounding_elements(elem, text_elements)
                if surrounding >= 2:
                    return True
        
        return False
    
    def _detect_margin_notes(self, elements: List[LayoutElement]) -> bool:
        """Detect margin notes"""
        
        # Margin notes are small text elements in margins
        text_elements = [e for e in elements if e.type == 'text']
        
        for elem in text_elements:
            # Check if in margin (far left or right)
            if elem.bbox['x'] < 50 or elem.bbox['x'] > 500:  # Adjust based on page size
                if elem.bbox.get('width', 0) < 100:  # Narrow
                    return True
        
        return False
    
    def _detect_floating(self, elements: List[LayoutElement]) -> bool:
        """Detect floating elements"""
        
        # Floating elements have z-index > 0
        return any(e.z_index > 0 for e in elements)
    
    def _count_surrounding_elements(self, element: LayoutElement, 
                                   all_elements: List[LayoutElement]) -> int:
        """Count elements surrounding given element"""
        
        count = 0
        for other in all_elements:
            if other.id != element.id:
                distance = self._calculate_distance(element.bbox, other.bbox)
                if distance < 100:  # Within 100 pixels
                    count += 1
        
        return count
    
    def _calculate_overlay_adjustment(self, text: Dict, img: Dict) -> Dict:
        """Calculate adjustment to resolve overlay"""
        
        # Simple strategy: move text to nearest free space
        return {
            'element_id': text['id'],
            'action': 'reposition',
            'new_position': {
                'x': img['bbox']['x'] + img['bbox'].get('width', 0) + 10,
                'y': text['bbox']['y']
            }
        }
    
    def _find_related_elements(self, element_id: str, 
                              relationships: List[Dict]) -> List[Dict]:
        """Find elements related to given element"""
        
        related = []
        for rel in relationships:
            if (rel.get('element1') == element_id or 
                rel.get('element2') == element_id or
                rel.get('image') == element_id or
                rel.get('caption') == element_id):
                related.append(rel)
        
        return related
    
    def _adjust_for_relationships(self, element: Dict, 
                                 relationships: List[Dict],
                                 all_elements: List[Dict]) -> Dict:
        """Adjust element position based on relationships"""
        
        adjusted = element.copy()
        
        for rel in relationships:
            if rel['type'] == LayoutRelationship.CAPTION.value:
                # Keep caption near image
                if rel.get('caption') == element['id']:
                    img_elem = next((e for e in all_elements if e['id'] == rel['image']), None)
                    if img_elem:
                        # Position below image
                        adjusted['bbox']['y'] = (img_elem['bbox']['y'] + 
                                                img_elem['bbox'].get('height', 0) + 10)
                        # Center horizontally
                        adjusted['bbox']['x'] = img_elem['bbox']['x']
        
        return adjusted
```

## Usage Example

```python
# Initialize manager
layout_manager = LayoutManager()

# Analyze layout
layout = layout_manager.analyze_layout(page_content)

print(f"Layout type: {layout['flow_type']}")
print(f"Columns detected: {len(layout['columns'])}")
print(f"Relationships found: {len(layout['relationships'])}")

# Handle overlays
overlays = layout_manager.handle_overlays(text_elements, image_elements)
print(f"Collisions: {len(overlays['collisions'])}")
print(f"Watermarks: {len(overlays['watermarks'])}")

# Maintain positions after translation
adjusted = layout_manager.maintain_relative_positions(layout, translated_elements)
```
