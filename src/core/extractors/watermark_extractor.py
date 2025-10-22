"""
Watermark Extractor - Extract visible and invisible watermarks
Preserves for reconstruction.
"""

import fitz
import cv2
import numpy as np
from typing import Dict, List, Optional
import base64


class WatermarkExtractor:
    """
    Extract visible and invisible watermarks.
    Preserves for reconstruction.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.detect_visible = self.config.get('detect_visible', True)
        self.detect_invisible = self.config.get('detect_invisible', False)
        self.opacity_threshold = self.config.get('opacity_threshold', 0.5)

    def extract_watermarks(self, pdf_path: str) -> List[Dict]:
        """
        Extract all watermarks from PDF.

        Args:
            pdf_path: Path to PDF

        Returns:
            List of watermark dictionaries
        """
        watermarks = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            # Extract visible watermarks
            if self.detect_visible:
                visible = self._extract_visible_watermarks(page, page_num)
                watermarks.extend(visible)

            # Extract invisible watermarks
            if self.detect_invisible:
                invisible = self._extract_invisible_watermarks(page, page_num)
                watermarks.extend(invisible)

        doc.close()
        return watermarks

    def _extract_visible_watermarks(self, page, page_num: int) -> List[Dict]:
        """Extract visible watermarks"""

        watermarks = []

        # Check for transparent text
        text_instances = page.get_text("dict")
        for block in text_instances.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Check opacity - watermarks are typically semi-transparent
                        # Since fitz doesn't directly provide opacity, we check other indicators
                        text = span.get("text", "")
                        font_size = span.get("size", 12)
                        bbox = span.get("bbox", (0, 0, 0, 0))

                        # Heuristics for watermark detection
                        is_large_text = font_size > 30
                        is_diagonal = self._check_if_diagonal(bbox)
                        is_watermark_text = any(word in text.lower() for word in
                                               ['draft', 'confidential', 'copy', 'sample', 'watermark'])

                        if is_large_text or is_diagonal or is_watermark_text:
                            watermarks.append({
                                'page': page_num,
                                'type': 'visible_text',
                                'text': text,
                                'bbox': {
                                    'x': bbox[0],
                                    'y': bbox[1],
                                    'width': bbox[2] - bbox[0],
                                    'height': bbox[3] - bbox[1]
                                },
                                'font': span.get("font", ""),
                                'size': font_size,
                                'color': span.get("color", 0),
                                'detection_reason':
                                    'large_text' if is_large_text else
                                    'diagonal' if is_diagonal else
                                    'watermark_keyword'
                            })

        # Check for image-based watermarks
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                # Check if image might be watermark (typically semi-transparent)
                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_data = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                if img_data is not None and len(img_data.shape) == 3:
                    # Check if has alpha channel or low opacity indicators
                    if img_data.shape[2] == 4:  # Has alpha channel
                        alpha = img_data[:, :, 3]
                        avg_opacity = np.mean(alpha) / 255.0

                        if avg_opacity < self.opacity_threshold:
                            watermarks.append({
                                'page': page_num,
                                'type': 'visible_image',
                                'image_ref': xref,
                                'opacity': avg_opacity,
                                'format': base_image.get("ext", "unknown"),
                                'detection_reason': 'transparent_image'
                            })
            except Exception as e:
                print(f"Could not process image {xref}: {e}")
                continue

        return watermarks

    def _extract_invisible_watermarks(self, page, page_num: int) -> List[Dict]:
        """Extract invisible watermarks"""

        watermarks = []

        try:
            # Render page
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Try LSB steganography detection
            watermark_data = self._detect_lsb_watermark(img)

            if watermark_data:
                watermarks.append({
                    'page': page_num,
                    'type': 'invisible',
                    'data': base64.b64encode(watermark_data).decode(),
                    'method': 'lsb',
                    'detection_reason': 'lsb_pattern_detected'
                })

        except Exception as e:
            print(f"Invisible watermark extraction failed: {e}")

        return watermarks

    def _check_if_diagonal(self, bbox: tuple) -> bool:
        """Check if text box is rotated/diagonal"""
        # Simple check based on bbox dimensions
        # This is a simplified version; real rotation detection needs transformation matrix
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # If width and height are very similar, might be rotated
        if width > 0 and height > 0:
            ratio = max(width, height) / min(width, height)
            return 0.8 < ratio < 1.2 and width > 100
        return False

    def _detect_lsb_watermark(self, image: np.ndarray) -> Optional[bytes]:
        """Detect LSB (Least Significant Bit) watermark"""

        try:
            # Extract LSBs from blue channel (common practice)
            if len(image.shape) == 3:
                blue_channel = image[:, :, 0]
                lsb = blue_channel & 1

                # Convert LSB array to bytes
                bits = lsb.flatten()

                # Check for patterns that might indicate embedded data
                # Look for non-random patterns in first 1000 bits
                if len(bits) > 1000:
                    sample = bits[:1000]
                    ones_count = np.sum(sample)

                    # If significantly skewed from 50/50, might contain data
                    if ones_count < 400 or ones_count > 600:
                        # Extract potential watermark data (first 32 bytes)
                        byte_array = []
                        for i in range(0, min(256, len(bits)), 8):
                            if i + 8 <= len(bits):
                                byte = 0
                                for j in range(8):
                                    byte = (byte << 1) | bits[i + j]
                                byte_array.append(byte)

                        if byte_array:
                            return bytes(byte_array)

        except Exception as e:
            print(f"LSB detection failed: {e}")

        return None

    def has_watermark(self, pdf_path: str) -> bool:
        """Quick check if PDF has any watermarks"""
        watermarks = self.extract_watermarks(pdf_path)
        return len(watermarks) > 0
