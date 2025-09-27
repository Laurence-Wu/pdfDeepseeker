"""
Layout Manager - Analyzes and manages document spatial relationships and layout structure.
Determines column structure and text positioning for layout preservation.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
import fitz  # PyMuPDF
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LayoutType(str, Enum):
    """Types of document layouts."""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """Represents a block of text with positioning information."""
    id: str
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    page_number: int
    block_type: str = "text"


@dataclass
class LayoutInfo:
    """Information about document layout."""
    layout_type: LayoutType
    columns: List[Dict[str, Any]]
    text_blocks: List[TextBlock]
    page_count: int
    confidence: float
    spatial_relationships: Dict[str, Any]


class LayoutManager:
    """
    Analyzes and manages document spatial relationships and layout structure.
    Determines column structure and text positioning for layout preservation.
    """

    def __init__(self):
        self.min_column_width = 100  # Minimum column width in points
        self.column_gap_threshold = 50  # Minimum gap between columns
        self.text_density_threshold = 0.1  # Minimum text density for column detection

    async def extract_layout(self, pdf_path: str) -> LayoutInfo:
        """
        Extract layout information from PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            LayoutInfo with detected layout structure
        """
        logger.info(f"Extracting layout from {pdf_path}")

        try:
            loop = asyncio.get_event_loop()
            layout_info = await loop.run_in_executor(None, self._analyze_layout_sync, pdf_path)

            logger.info(f"Detected layout: {layout_info.layout_type}")
            return layout_info

        except Exception as e:
            logger.error(f"Failed to extract layout: {str(e)}")
            # Return default layout on failure
            return LayoutInfo(
                layout_type=LayoutType.UNKNOWN,
                columns=[],
                text_blocks=[],
                page_count=1,
                confidence=0.1,
                spatial_relationships={}
            )

    def _analyze_layout_sync(self, pdf_path: str) -> LayoutInfo:
        """
        Synchronous layout analysis using PyMuPDF.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        all_text_blocks = []
        page_analyses = []

        # Analyze each page
        for page_num in range(min(10, total_pages)):  # Sample first 10 pages for performance
            page = doc[page_num]
            page_analysis = self._analyze_page_layout(page, page_num)
            page_analyses.append(page_analysis)
            all_text_blocks.extend(page_analysis['text_blocks'])

        # Determine overall layout type
        layout_type = self._determine_layout_type(page_analyses)

        # Extract column information
        columns = self._extract_column_structure(page_analyses)

        # Build spatial relationships
        spatial_relationships = self._build_spatial_relationships(all_text_blocks)

        doc.close()

        return LayoutInfo(
            layout_type=layout_type,
            columns=columns,
            text_blocks=all_text_blocks,
            page_count=total_pages,
            confidence=self._calculate_layout_confidence(page_analyses),
            spatial_relationships=spatial_relationships
        )

    def _analyze_page_layout(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """
        Analyze layout of a single page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page layout analysis
        """
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Get text blocks
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [block for block in blocks if block.get("type") == 0]

        processed_blocks = []

        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox")
                    if bbox:
                        text_block = TextBlock(
                            id=f"p{page_num}_b{len(processed_blocks)}",
                            text=span.get("text", ""),
                            bbox=bbox,
                            font_size=span.get("size", 12),
                            font_name=span.get("font", "Unknown"),
                            page_number=page_num
                        )
                        processed_blocks.append(text_block)

        # Analyze column structure
        column_bounds = self._detect_columns(processed_blocks, page_width)

        return {
            'page_number': page_num,
            'text_blocks': processed_blocks,
            'column_bounds': column_bounds,
            'text_density': len(processed_blocks) / (page_width * page_height) if processed_blocks else 0
        }

    def _detect_columns(self, text_blocks: List[TextBlock], page_width: float) -> List[Dict[str, Any]]:
        """
        Detect column boundaries from text blocks.

        Args:
            text_blocks: List of text blocks on the page
            page_width: Page width in points

        Returns:
            List of column boundary dictionaries
        """
        if not text_blocks:
            return []

        # Group text blocks by their x-position
        x_positions = [block.bbox[0] for block in text_blocks]
        x_positions.sort()

        # Find gaps between text blocks to identify columns
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > self.column_gap_threshold:
                gaps.append((x_positions[i-1], x_positions[i], gap))

        if not gaps:
            # Single column
            return [{'left': 0, 'right': page_width, 'width': page_width}]

        # Create column boundaries
        columns = []
        prev_right = 0

        for start_x, end_x, gap_width in gaps:
            # Add column before gap
            if start_x - prev_right > self.min_column_width:
                columns.append({
                    'left': prev_right,
                    'right': start_x,
                    'width': start_x - prev_right
                })
            prev_right = end_x

        # Add final column
        if page_width - prev_right > self.min_column_width:
            columns.append({
                'left': prev_right,
                'right': page_width,
                'width': page_width - prev_right
            })

        return columns

    def _determine_layout_type(self, page_analyses: List[Dict[str, Any]]) -> LayoutType:
        """
        Determine overall document layout type based on page analyses.
        """
        if not page_analyses:
            return LayoutType.UNKNOWN

        column_counts = []
        densities = []

        for analysis in page_analyses:
            columns = analysis['column_bounds']
            column_counts.append(len(columns))
            densities.append(analysis['text_density'])

        avg_columns = np.mean(column_counts)
        avg_density = np.mean(densities)

        if avg_columns <= 1.2:
            return LayoutType.SINGLE_COLUMN
        elif avg_columns <= 2.2:
            return LayoutType.TWO_COLUMN
        elif avg_density > self.text_density_threshold:
            return LayoutType.MULTI_COLUMN
        else:
            return LayoutType.MIXED

    def _extract_column_structure(self, page_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract consistent column structure across pages.
        """
        all_columns = []

        for analysis in page_analyses:
            columns = analysis['column_bounds']
            all_columns.extend(columns)

        if not all_columns:
            return []

        # Group columns by position and find consistent ones
        column_groups = {}
        for col in all_columns:
            key = f"{col['left']".0f"}_{col['right']".0f"}"
            if key not in column_groups:
                column_groups[key] = []
            column_groups[key].append(col)

        # Return most consistent columns
        consistent_columns = []
        for group_cols in column_groups.values():
            if len(group_cols) >= len(page_analyses) * 0.5:  # Appears on at least 50% of pages
                avg_col = {
                    'left': np.mean([c['left'] for c in group_cols]),
                    'right': np.mean([c['right'] for c in group_cols]),
                    'width': np.mean([c['width'] for c in group_cols]),
                    'frequency': len(group_cols)
                }
                consistent_columns.append(avg_col)

        return sorted(consistent_columns, key=lambda x: x['left'])

    def _build_spatial_relationships(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """
        Build spatial relationships between text blocks for layout preservation.
        """
        relationships = {
            'above_below': [],
            'left_right': [],
            'overlapping': []
        }

        # Sort blocks by page and position
        blocks_by_page = {}
        for block in text_blocks:
            if block.page_number not in blocks_by_page:
                blocks_by_page[block.page_number] = []
            blocks_by_page[block.page_number].append(block)

        # Analyze relationships within each page
        for page_num, blocks in blocks_by_page.items():
            # Sort by vertical position (top to bottom)
            blocks.sort(key=lambda b: b.bbox[1])

            for i, block1 in enumerate(blocks):
                for j, block2 in enumerate(blocks[i+1:], i+1):
                    rel = self._analyze_block_relationship(block1, block2)
                    if rel:
                        relationships[rel['type']].append(rel)

        return relationships

    def _analyze_block_relationship(self, block1: TextBlock, block2: TextBlock) -> Optional[Dict[str, Any]]:
        """
        Analyze spatial relationship between two text blocks.
        """
        x1, y1, x1_end, y1_end = block1.bbox
        x2, y2, x2_end, y2_end = block2.bbox

        # Check vertical relationship (above/below)
        vertical_overlap = min(y1_end, y2_end) - max(y1, y2)
        if vertical_overlap > 0:
            return {
                'type': 'overlapping',
                'block1_id': block1.id,
                'block2_id': block2.id,
                'overlap_ratio': vertical_overlap / min(y1_end - y1, y2_end - y2)
            }

        # Check horizontal relationship (left/right)
        if abs(y1 - y2) < 20:  # Roughly same vertical position
            if x1_end < x2:  # block1 is left of block2
                return {
                    'type': 'left_right',
                    'block1_id': block1.id,
                    'block2_id': block2.id,
                    'gap': x2 - x1_end
                }

        # Check vertical relationship (above/below)
        if x1 <= x2 <= x1_end or x2 <= x1 <= x2_end:  # Overlapping horizontally
            if y1_end <= y2:  # block1 is above block2
                return {
                    'type': 'above_below',
                    'block1_id': block1.id,
                    'block2_id': block2.id,
                    'vertical_gap': y2 - y1_end
                }

        return None

    def _calculate_layout_confidence(self, page_analyses: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for layout analysis.
        """
        if not page_analyses:
            return 0.0

        # Based on consistency of layout across pages
        layout_consistency = []
        for analysis in page_analyses:
            # Simple consistency metric based on text density
            density = analysis['text_density']
            # Normalize density (assume reasonable range is 0.001 to 0.1)
            normalized_density = min(max(density / 0.1, 0), 1)
            layout_consistency.append(normalized_density)

        return np.mean(layout_consistency)
