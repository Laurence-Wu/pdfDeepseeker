# VLA Trigger - Part 1: Core Detection System

## Overview
Determines when to use Vision-Language Models based on document complexity analysis.

## Implementation

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image

class ComplexityLevel(Enum):
    """Document complexity levels"""
    SIMPLE = 1      # Plain text, single column
    MODERATE = 2    # Some formatting, clear structure
    COMPLEX = 3     # Mixed content, multiple columns
    EXTREME = 4     # Highly visual, artistic layout

@dataclass
class VLADecision:
    """VLA usage decision"""
    use_vla: bool
    confidence: float
    reasons: List[str]
    complexity_level: ComplexityLevel
    recommended_model: str
    fallback_model: Optional[str]

class VLATrigger:
    """
    Intelligent VLA triggering based on document complexity.
    Analyzes multiple factors to determine optimal processing path.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize VLA Trigger.
        
        Args:
            config: Configuration with thresholds
        """
        self.config = config or {}
        
        # Complexity weights
        self.complexity_weights = {
            'layout_complexity': 0.25,
            'mixed_content': 0.20,
            'ocr_confidence': 0.20,
            'visual_elements': 0.15,
            'text_structure': 0.10,
            'quality_issues': 0.10
        }
        
        # Thresholds
        self.thresholds = {
            'simple': 0.3,
            'moderate': 0.5,
            'complex': 0.7,
            'extreme': 0.85
        }
        
        # Model recommendations
        self.model_mapping = {
            ComplexityLevel.SIMPLE: 'paddleocr',
            ComplexityLevel.MODERATE: 'surya',
            ComplexityLevel.COMPLEX: 'mplug-docowl',
            ComplexityLevel.EXTREME: 'internvl-2.0'
        }
    
    def analyze_document(
        self,
        page_image: np.ndarray,
        initial_extraction: Dict
    ) -> VLADecision:
        """
        Analyze document to determine if VLA is needed.
        
        Args:
            page_image: Page as numpy array
            initial_extraction: Results from initial OCR attempt
            
        Returns:
            VLADecision object
        """
        # Calculate all complexity factors
        factors = self.calculate_complexity_factors(
            page_image, initial_extraction
        )
        
        # Make decision based on weighted factors
        decision = self.make_vla_decision(factors)
        
        return decision
    
    def calculate_complexity_factors(
        self,
        page_image: np.ndarray,
        extraction: Dict
    ) -> Dict[str, float]:
        """
        Calculate all factors indicating document complexity.
        
        Returns:
            Dict with factor scores (0-1)
        """
        factors = {}
        
        # 1. Layout Complexity
        factors['layout_complexity'] = self._analyze_layout_complexity(page_image)
        
        # 2. Mixed Content (text + images + tables)
        factors['mixed_content'] = self._analyze_mixed_content(extraction)
        
        # 3. OCR Confidence
        factors['ocr_confidence'] = self._analyze_ocr_confidence(extraction)
        
        # 4. Visual Elements
        factors['visual_elements'] = self._analyze_visual_elements(page_image)
        
        # 5. Text Structure
        factors['text_structure'] = self._analyze_text_structure(extraction)
        
        # 6. Quality Issues
        factors['quality_issues'] = self._analyze_quality_issues(page_image)
        
        return factors
    
    def _analyze_layout_complexity(self, image: np.ndarray) -> float:
        """
        Analyze layout complexity using edge detection and region analysis.
        
        Returns:
            Complexity score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Detect text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours (text blocks)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze contour distribution
        if len(contours) == 0:
            return 0.0
        
        # Calculate complexity metrics
        areas = [cv2.contourArea(c) for c in contours]
        area_variance = np.var(areas) / (np.mean(areas) + 1e-6)
        
        # Check for multi-column layout
        x_positions = [cv2.boundingRect(c)[0] for c in contours]
        x_clusters = self._count_clusters(x_positions, threshold=50)
        
        # Combined complexity score
        complexity = min(1.0, (
            edge_density * 2 +  # Edge density indicator
            min(1.0, len(contours) / 50) +  # Number of regions
            min(1.0, area_variance / 1000) +  # Variance in sizes
            min(1.0, (x_clusters - 1) / 3)  # Column count
        ) / 4)
        
        return complexity
    
    def _analyze_mixed_content(self, extraction: Dict) -> float:
        """
        Analyze presence of mixed content types.
        
        Returns:
            Mixed content score (0-1)
        """
        content_types = {
            'has_text': len(extraction.get('text_blocks', [])) > 0,
            'has_images': len(extraction.get('images', [])) > 0,
            'has_tables': len(extraction.get('tables', [])) > 0,
            'has_formulas': len(extraction.get('formulas', [])) > 0,
            'has_charts': len(extraction.get('charts', [])) > 0
        }
        
        # Count active content types
        active_types = sum(content_types.values())
        
        # Calculate complexity based on diversity
        if active_types <= 1:
            return 0.0
        elif active_types == 2:
            return 0.3
        elif active_types == 3:
            return 0.6
        else:
            return 0.9
    
    def _analyze_ocr_confidence(self, extraction: Dict) -> float:
        """
        Analyze OCR confidence scores.
        
        Returns:
            Inverse confidence score (0-1, higher = worse OCR)
        """
        confidences = []
        
        # Collect all confidence scores
        for block in extraction.get('text_blocks', []):
            if 'confidence' in block:
                confidences.append(block['confidence'])
        
        if not confidences:
            return 0.5  # No confidence data
        
        # Calculate metrics
        avg_confidence = np.mean(confidences)
        low_confidence_ratio = sum(c < 0.8 for c in confidences) / len(confidences)
        
        # Inverse score (low confidence = high complexity)
        complexity = (1 - avg_confidence) * 0.7 + low_confidence_ratio * 0.3
        
        return complexity
    
    def _analyze_visual_elements(self, image: np.ndarray) -> float:
        """
        Analyze visual complexity of the image.
        
        Returns:
            Visual complexity score (0-1)
        """
        # Color variance
        if len(image.shape) == 3:
            color_variance = np.var(image) / 255**2
        else:
            color_variance = 0
        
        # Texture analysis using Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = np.var(laplacian) / 1000
        
        # Detect graphical elements (non-text regions)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = np.sum(binary == 255) / binary.size
        
        # High white ratio might indicate diagrams/charts
        graphic_score = abs(white_ratio - 0.85)  # Deviation from typical text page
        
        # Combined visual complexity
        complexity = min(1.0, (
            color_variance +
            min(1.0, texture_score) +
            graphic_score * 2
        ) / 3)
        
        return complexity
    
    def _analyze_text_structure(self, extraction: Dict) -> float:
        """
        Analyze text structure complexity.
        
        Returns:
            Structure complexity score (0-1)
        """
        text_blocks = extraction.get('text_blocks', [])
        
        if not text_blocks:
            return 0.0
        
        # Analyze font variations
        fonts = set()
        font_sizes = []
        for block in text_blocks:
            if 'font' in block:
                fonts.add(block['font'])
            if 'font_size' in block:
                font_sizes.append(block['font_size'])
        
        font_variety = min(1.0, len(fonts) / 5)
        size_variance = np.var(font_sizes) / 100 if font_sizes else 0
        
        # Analyze text alignment
        alignments = [block.get('alignment', 'left') for block in text_blocks]
        alignment_variety = len(set(alignments)) / 4  # max 4 alignment types
        
        # Check for special formatting
        has_bullets = any('•' in block.get('text', '') for block in text_blocks)
        has_numbering = any(
            any(c.isdigit() for c in block.get('text', '')[:3])
            for block in text_blocks
        )
        
        # Combined structure complexity
        complexity = min(1.0, (
            font_variety * 0.3 +
            min(1.0, size_variance) * 0.3 +
            alignment_variety * 0.2 +
            (0.1 if has_bullets else 0) +
            (0.1 if has_numbering else 0)
        ))
        
        return complexity
    
    def _analyze_quality_issues(self, image: np.ndarray) -> float:
        """
        Detect quality issues that might affect OCR.
        
        Returns:
            Quality issue score (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Check for skew
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        skew_score = 0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 45:
                    angle = 90 - angle
                angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                skew_score = min(1.0, avg_angle / 10)  # Normalize to 0-1
        
        # Check for noise
        noise_score = self._estimate_noise(gray)
        
        # Check for blur
        blur_score = self._estimate_blur(gray)
        
        # Check contrast
        contrast_score = 1 - (np.std(gray) / 128)  # Low contrast = high score
        
        # Combined quality issue score
        quality_issues = (
            skew_score * 0.3 +
            noise_score * 0.3 +
            blur_score * 0.2 +
            contrast_score * 0.2
        )
        
        return min(1.0, quality_issues)
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate image noise level"""
        # Use difference between image and its median filtered version
        median = cv2.medianBlur(gray, 5)
        diff = np.abs(gray.astype(float) - median.astype(float))
        noise_level = np.mean(diff) / 255
        return min(1.0, noise_level * 10)
    
    def _estimate_blur(self, gray: np.ndarray) -> float:
        """Estimate image blur using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Low variance indicates blur
        blur_score = max(0, 1 - variance / 500)
        return blur_score
    
    def _count_clusters(self, values: List[float], threshold: float) -> int:
        """Count number of clusters in 1D data"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        clusters = 1
        
        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] > threshold:
                clusters += 1
        
        return clusters
    
    def make_vla_decision(self, factors: Dict[str, float]) -> VLADecision:
        """
        Make final VLA decision based on factors.
        
        Args:
            factors: Complexity factors
            
        Returns:
            VLADecision object
        """
        # Calculate weighted complexity score
        total_complexity = sum(
            factors.get(factor, 0) * weight
            for factor, weight in self.complexity_weights.items()
        )
        
        # Determine complexity level
        if total_complexity < self.thresholds['simple']:
            level = ComplexityLevel.SIMPLE
        elif total_complexity < self.thresholds['moderate']:
            level = ComplexityLevel.MODERATE
        elif total_complexity < self.thresholds['complex']:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.EXTREME
        
        # Decide whether to use VLA
        use_vla = total_complexity >= self.thresholds['moderate']
        
        # Generate reasoning
        reasons = []
        for factor, score in factors.items():
            if score > 0.5:
                reasons.append(f"High {factor.replace('_', ' ')}: {score:.2f}")
        
        # Select models
        recommended_model = self.model_mapping[level]
        fallback_model = self.model_mapping.get(
            ComplexityLevel(max(1, level.value - 1))
        )
        
        return VLADecision(
            use_vla=use_vla,
            confidence=min(0.95, total_complexity + 0.3),
            reasons=reasons,
            complexity_level=level,
            recommended_model=recommended_model,
            fallback_model=fallback_model
        )
```

## Usage Example

```python
# Initialize trigger
trigger = VLATrigger()

# Analyze document
decision = trigger.analyze_document(page_image, initial_ocr_result)

if decision.use_vla:
    print(f"Using VLA model: {decision.recommended_model}")
    print(f"Complexity: {decision.complexity_level.name}")
    print(f"Reasons: {', '.join(decision.reasons)}")
else:
    print("Standard OCR is sufficient")
```
