"""
Margin Manager - Detects and manages document margins for layout preservation.
Uses statistical analysis to determine consistent margins across pages.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarginInfo:
    """Information about document margins."""
    left: float
    right: float
    top: float
    bottom: float
    page_count: int
    confidence: float


class MarginManager:
    """
    Detects and manages document margins for layout preservation.
    Uses statistical analysis to determine consistent margins across pages.
    """

    def __init__(self):
        self.min_sample_pages = 3
        self.margin_tolerance = 0.02  # 2% tolerance for margin variations

    async def extract_margins(self, pdf_path: str) -> MarginInfo:
        """
        Extract margin information from PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            MarginInfo with detected margins and confidence
        """
        logger.info(f"Extracting margins from {pdf_path}")

        try:
            # Run margin detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            margin_info = await loop.run_in_executor(None, self._detect_margins_sync, pdf_path)

            logger.info(f"Detected margins: {margin_info}")
            return margin_info

        except Exception as e:
            logger.error(f"Failed to extract margins: {str(e)}")
            # Return default margins on failure
            return MarginInfo(72, 72, 72, 72, 1, 0.5)  # 1 inch default margins

    def _detect_margins_sync(self, pdf_path: str) -> MarginInfo:
        """
        Synchronous margin detection using PyMuPDF.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Sample pages for margin detection (not all pages to improve performance)
        sample_pages = min(self.min_sample_pages, total_pages)
        page_indices = np.linspace(0, total_pages-1, sample_pages, dtype=int)

        all_margins = []

        for page_idx in page_indices:
            page = doc[page_idx]
            margins = self._analyze_page_margins(page)
            if margins:
                all_margins.append(margins)

        if not all_margins:
            doc.close()
            return MarginInfo(72, 72, 72, 72, total_pages, 0.1)

        # Calculate statistical margins
        margins_array = np.array(all_margins)
        left_margins = margins_array[:, 0]
        right_margins = margins_array[:, 1]
        top_margins = margins_array[:, 2]
        bottom_margins = margins_array[:, 3]

        # Calculate mean and standard deviation
        left_mean, left_std = np.mean(left_margins), np.std(left_margins)
        right_mean, right_std = np.mean(right_margins), np.std(right_margins)
        top_mean, top_std = np.mean(top_margins), np.std(top_margins)
        bottom_mean, bottom_std = np.mean(bottom_margins), np.std(bottom_margins)

        # Filter out outliers using standard deviation
        left_filtered = [x for x in left_margins if abs(x - left_mean) <= left_std * 1.5]
        right_filtered = [x for x in right_margins if abs(x - right_mean) <= right_std * 1.5]
        top_filtered = [x for x in top_margins if abs(x - top_mean) <= top_std * 1.5]
        bottom_filtered = [x for x in bottom_margins if abs(x - bottom_mean) <= bottom_std * 1.5]

        # Calculate final margins
        final_left = np.mean(left_filtered) if left_filtered else left_mean
        final_right = np.mean(right_filtered) if right_filtered else right_mean
        final_top = np.mean(top_filtered) if top_filtered else top_mean
        final_bottom = np.mean(bottom_filtered) if bottom_filtered else bottom_mean

        # Calculate confidence based on consistency
        consistency_score = self._calculate_consistency_score(
            left_std, right_std, top_std, bottom_std
        )

        doc.close()

        return MarginInfo(
            left=final_left,
            right=final_right,
            top=final_top,
            bottom=final_bottom,
            page_count=total_pages,
            confidence=consistency_score
        )

    def _analyze_page_margins(self, page: fitz.Page) -> Optional[Tuple[float, float, float, float]]:
        """
        Analyze margins for a single page.

        Args:
            page: PyMuPDF page object

        Returns:
            Tuple of (left, right, top, bottom) margins in points, or None if detection fails
        """
        try:
            # Get page dimensions
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # Get text blocks on the page
            blocks = page.get_text("dict")["blocks"]

            if not blocks:
                return None

            # Find text boundaries
            text_blocks = [block for block in blocks if block.get("type") == 0]  # Text blocks only

            if not text_blocks:
                return None

            # Extract bounding boxes
            bboxes = []
            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox")
                        if bbox:
                            bboxes.append(bbox)

            if not bboxes:
                return None

            # Convert to numpy array
            bbox_array = np.array(bboxes)

            # Find text boundaries
            min_x = np.min(bbox_array[:, 0])
            max_x = np.max(bbox_array[:, 2])
            min_y = np.min(bbox_array[:, 1])
            max_y = np.max(bbox_array[:, 3])

            # Calculate margins (in points)
            left_margin = min_x
            right_margin = page_width - max_x
            top_margin = min_y
            bottom_margin = page_height - max_y

            return (left_margin, right_margin, top_margin, bottom_margin)

        except Exception as e:
            logger.warning(f"Failed to analyze margins for page: {str(e)}")
            return None

    def _calculate_consistency_score(self, left_std: float, right_std: float,
                                   top_std: float, bottom_std: float) -> float:
        """
        Calculate consistency score based on margin standard deviations.
        Lower standard deviation = higher consistency = higher confidence.
        """
        # Average standard deviation
        avg_std = np.mean([left_std, right_std, top_std, bottom_std])

        # Normalize to 0-1 scale (lower is better, so invert)
        # Assume max reasonable std is 50 points
        max_reasonable_std = 50.0
        normalized_std = min(avg_std / max_reasonable_std, 1.0)

        return 1.0 - normalized_std

    def enforce_margins(self, content_bounds: Dict, target_margins: MarginInfo) -> Dict:
        """
        Enforce consistent margins on content.

        Args:
            content_bounds: Current content boundaries
            target_margins: Target margin information

        Returns:
            Adjusted content boundaries
        """
        adjusted_bounds = content_bounds.copy()

        # Ensure content respects target margins
        adjusted_bounds['left'] = max(adjusted_bounds.get('left', 0), target_margins.left)
        adjusted_bounds['right'] = min(adjusted_bounds.get('right', 1000), target_margins.right)
        adjusted_bounds['top'] = max(adjusted_bounds.get('top', 0), target_margins.top)
        adjusted_bounds['bottom'] = min(adjusted_bounds.get('bottom', 1000), target_margins.bottom)

        return adjusted_bounds
