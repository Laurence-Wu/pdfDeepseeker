# VLA Trigger - Part 3: Processing Pipeline

## Overview
Complete VLA processing pipeline with error handling, caching, and optimization.

## Implementation

```python
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
import time

@dataclass
class ProcessingResult:
    """Complete processing result"""
    success: bool
    data: Dict
    model_used: str
    processing_time: float
    confidence: float
    errors: List[str]
    cached: bool = False

class VLAProcessingPipeline:
    """
    Complete VLA processing pipeline with optimization.
    Handles decision making, model selection, and processing.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize processing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.trigger = VLATrigger(config)
        self.processor = VLAProcessor(config)
        self.model_selector = ModelSelector()
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_dir = Path(self.config.get('cache_dir', '/tmp/vla_cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # Performance monitoring
        self.metrics = {
            'total_processed': 0,
            'cache_hits': 0,
            'model_usage': {},
            'avg_processing_time': 0
        }
    
    async def process_document(
        self,
        image: np.ndarray,
        document_info: Optional[Dict] = None,
        force_vla: bool = False
    ) -> ProcessingResult:
        """
        Process document with intelligent VLA usage.
        
        Args:
            image: Document image
            document_info: Additional document information
            force_vla: Force VLA usage regardless of complexity
            
        Returns:
            ProcessingResult with extracted content
        """
        start_time = time.time()
        errors = []
        
        # Check cache first
        if self.cache_enabled and not force_vla:
            cached_result = await self._check_cache(image)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return cached_result
        
        try:
            # Step 1: Initial OCR attempt (fast)
            initial_result = await self._initial_ocr(image)
            
            # Step 2: Analyze complexity and decide on VLA
            if not force_vla:
                decision = self.trigger.analyze_document(image, initial_result)
            else:
                # Force VLA with high complexity assumption
                decision = VLADecision(
                    use_vla=True,
                    confidence=0.9,
                    reasons=["Forced VLA processing"],
                    complexity_level=ComplexityLevel.COMPLEX,
                    recommended_model='surya',
                    fallback_model='paddleocr'
                )
            
            # Step 3: Process based on decision
            if decision.use_vla:
                result = await self._process_with_vla(
                    image,
                    decision,
                    document_info
                )
            else:
                # Use initial OCR result
                result = self._structure_ocr_result(initial_result)
                model_used = 'standard_ocr'
            
            # Step 4: Post-processing
            result = await self._post_process(result, document_info)
            
            # Step 5: Quality check
            quality_score = self._assess_quality(result)
            
            # Step 6: Retry with better model if quality is poor
            if quality_score < 0.7 and not force_vla:
                print(f"Quality score {quality_score:.2f} too low, retrying with VLA")
                return await self.process_document(image, document_info, force_vla=True)
            
            # Cache result
            if self.cache_enabled:
                await self._cache_result(image, result)
            
            # Update metrics
            self._update_metrics(decision.recommended_model if decision.use_vla else 'standard_ocr', time.time() - start_time)
            
            return ProcessingResult(
                success=True,
                data=result,
                model_used=decision.recommended_model if decision.use_vla else 'standard_ocr',
                processing_time=time.time() - start_time,
                confidence=quality_score,
                errors=errors,
                cached=False
            )
            
        except Exception as e:
            errors.append(str(e))
            print(f"Processing error: {e}")
            
            # Return minimal result on error
            return ProcessingResult(
                success=False,
                data={'text_blocks': [], 'error': str(e)},
                model_used='error',
                processing_time=time.time() - start_time,
                confidence=0.0,
                errors=errors,
                cached=False
            )
    
    async def _initial_ocr(self, image: np.ndarray) -> Dict:
        """
        Perform fast initial OCR for complexity assessment.
        
        Args:
            image: Document image
            
        Returns:
            Initial OCR results
        """
        # Use PaddleOCR for speed
        if self.processor.models.get('paddleocr'):
            return await self.processor._process_with_paddleocr(image)
        else:
            # Fallback to basic text detection
            return {
                'text_blocks': [],
                'confidence': [],
                'layout': {}
            }
    
    async def _process_with_vla(
        self,
        image: np.ndarray,
        decision: VLADecision,
        document_info: Optional[Dict]
    ) -> Dict:
        """
        Process with VLA model based on decision.
        
        Args:
            image: Document image
            decision: VLA decision with model recommendation
            document_info: Additional context
            
        Returns:
            Extraction results
        """
        # Select model based on additional factors
        if document_info:
            model_name = self.model_selector.select_model(
                complexity_level=decision.complexity_level.name.lower(),
                document_type=document_info.get('type', 'general'),
                language=document_info.get('language', 'en'),
                resolution=(image.shape[1], image.shape[0])
            )
        else:
            model_name = decision.recommended_model
        
        print(f"Processing with {model_name} (complexity: {decision.complexity_level.name})")
        
        # Process with selected model
        try:
            result = await self.processor.process_with_vla(
                image,
                model_name,
                fallback=True
            )
            
            # Add metadata
            result['vla_metadata'] = {
                'model': model_name,
                'complexity': decision.complexity_level.name,
                'reasons': decision.reasons
            }
            
            return result
            
        except Exception as e:
            print(f"VLA processing failed: {e}, using fallback")
            
            # Try fallback model
            if decision.fallback_model:
                return await self.processor.process_with_vla(
                    image,
                    decision.fallback_model,
                    fallback=False
                )
            
            raise e
    
    def _structure_ocr_result(self, ocr_result: Dict) -> Dict:
        """
        Structure standard OCR result.
        
        Args:
            ocr_result: Raw OCR result
            
        Returns:
            Structured result
        """
        # Ensure consistent structure
        structured = {
            'text_blocks': ocr_result.get('text_blocks', []),
            'tables': [],
            'formulas': [],
            'images': [],
            'layout': ocr_result.get('layout', {}),
            'metadata': {
                'processing_type': 'standard_ocr',
                'confidence': np.mean(ocr_result.get('confidence', [0.9]))
            }
        }
        
        return structured
    
    async def _post_process(self, result: Dict, document_info: Optional[Dict]) -> Dict:
        """
        Post-process extraction results.
        
        Args:
            result: Raw extraction result
            document_info: Document context
            
        Returns:
            Enhanced result
        """
        # Sort text blocks by reading order
        result['text_blocks'] = self._sort_reading_order(result['text_blocks'])
        
        # Detect and group related elements
        result['groups'] = self._group_related_elements(result)
        
        # Identify special elements
        result['special_elements'] = self._identify_special_elements(result)
        
        # Add document-level metadata
        if document_info:
            result['document_metadata'] = {
                'type': document_info.get('type'),
                'language': document_info.get('language'),
                'page_number': document_info.get('page_number'),
                'total_pages': document_info.get('total_pages')
            }
        
        return result
    
    def _sort_reading_order(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Sort text blocks in reading order.
        
        Args:
            text_blocks: Unsorted text blocks
            
        Returns:
            Sorted text blocks
        """
        if not text_blocks:
            return text_blocks
        
        # Sort by y-position primarily, then x-position
        def sort_key(block):
            bbox = block.get('bbox', {})
            y = bbox.get('y', 0)
            x = bbox.get('x', 0)
            
            # Group into rows (tolerance of 10 pixels)
            row = y // 10
            return (row, x)
        
        return sorted(text_blocks, key=sort_key)
    
    def _group_related_elements(self, result: Dict) -> List[Dict]:
        """
        Group related elements (e.g., caption with image).
        
        Args:
            result: Extraction result
            
        Returns:
            List of element groups
        """
        groups = []
        
        # Group images with nearby captions
        for img in result.get('images', []):
            group = {'type': 'figure', 'image': img, 'caption': None}
            
            # Find nearby text that might be caption
            img_bbox = img.get('bbox', {})
            for text_block in result.get('text_blocks', []):
                text_bbox = text_block.get('bbox', {})
                
                # Check if text is below image and close
                if (text_bbox.get('y', 0) > img_bbox.get('y', 0) + img_bbox.get('height', 0) and
                    text_bbox.get('y', 0) < img_bbox.get('y', 0) + img_bbox.get('height', 0) + 50):
                    
                    # Check for caption keywords
                    text = text_block.get('text', '').lower()
                    if any(keyword in text for keyword in ['figure', 'fig.', 'image', '图']):
                        group['caption'] = text_block
                        break
            
            groups.append(group)
        
        return groups
    
    def _identify_special_elements(self, result: Dict) -> Dict:
        """
        Identify special elements like headers, footers, page numbers.
        
        Args:
            result: Extraction result
            
        Returns:
            Special elements dictionary
        """
        special = {
            'headers': [],
            'footers': [],
            'page_numbers': [],
            'footnotes': []
        }
        
        text_blocks = result.get('text_blocks', [])
        if not text_blocks:
            return special
        
        # Get page dimensions
        page_height = max(b.get('bbox', {}).get('y', 0) + b.get('bbox', {}).get('height', 0) for b in text_blocks)
        
        for block in text_blocks:
            bbox = block.get('bbox', {})
            y = bbox.get('y', 0)
            text = block.get('text', '')
            
            # Headers (top 10% of page)
            if y < page_height * 0.1:
                special['headers'].append(block)
            
            # Footers (bottom 10% of page)
            elif y > page_height * 0.9:
                special['footers'].append(block)
                
                # Check for page numbers
                if any(char.isdigit() for char in text) and len(text) < 10:
                    special['page_numbers'].append(block)
            
            # Footnotes (small text at bottom)
            elif y > page_height * 0.8:
                font_size = block.get('font_size', 12)
                if font_size < 10 or text.startswith(('*', '†', '‡', '§', '¶', '1', '2', '3')):
                    special['footnotes'].append(block)
        
        return special
    
    def _assess_quality(self, result: Dict) -> float:
        """
        Assess extraction quality.
        
        Args:
            result: Extraction result
            
        Returns:
            Quality score (0-1)
        """
        scores = []
        
        # Check text block confidence
        confidences = []
        for block in result.get('text_blocks', []):
            if 'confidence' in block:
                confidences.append(block['confidence'])
        
        if confidences:
            avg_confidence = np.mean(confidences)
            scores.append(avg_confidence)
        
        # Check completeness (presence of expected elements)
        has_text = len(result.get('text_blocks', [])) > 0
        scores.append(1.0 if has_text else 0.0)
        
        # Check layout detection
        has_layout = bool(result.get('layout'))
        scores.append(0.8 if has_layout else 0.5)
        
        # Check for errors
        has_errors = 'error' in result
        scores.append(0.0 if has_errors else 1.0)
        
        # Return average score
        return np.mean(scores) if scores else 0.0
    
    async def _check_cache(self, image: np.ndarray) -> Optional[ProcessingResult]:
        """Check cache for existing result"""
        
        if not self.cache_enabled:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(image)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check age
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_ttl:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    return ProcessingResult(
                        success=True,
                        data=data,
                        model_used='cached',
                        processing_time=0.0,
                        confidence=1.0,
                        errors=[],
                        cached=True
                    )
                except Exception as e:
                    print(f"Cache read error: {e}")
        
        return None
    
    async def _cache_result(self, image: np.ndarray, result: Dict):
        """Cache processing result"""
        
        if not self.cache_enabled:
            return
        
        try:
            cache_key = self._generate_cache_key(image)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        """Generate unique cache key for image"""
        
        # Use image hash
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _update_metrics(self, model: str, processing_time: float):
        """Update performance metrics"""
        
        self.metrics['total_processed'] += 1
        self.metrics['model_usage'][model] = self.metrics['model_usage'].get(model, 0) + 1
        
        # Update average processing time
        prev_avg = self.metrics['avg_processing_time']
        n = self.metrics['total_processed']
        self.metrics['avg_processing_time'] = (prev_avg * (n - 1) + processing_time) / n
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics.copy()


class VLABatchProcessor:
    """
    Process multiple documents in batch with VLA.
    """
    
    def __init__(self, pipeline: VLAProcessingPipeline):
        self.pipeline = pipeline
        self.max_concurrent = 5
    
    async def process_batch(
        self,
        images: List[np.ndarray],
        document_infos: Optional[List[Dict]] = None
    ) -> List[ProcessingResult]:
        """
        Process batch of documents.
        
        Args:
            images: List of document images
            document_infos: Optional document information
            
        Returns:
            List of processing results
        """
        if document_infos is None:
            document_infos = [None] * len(images)
        
        # Create tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_limit(img, info):
            async with semaphore:
                return await self.pipeline.process_document(img, info)
        
        tasks = [
            process_with_limit(img, info)
            for img, info in zip(images, document_infos)
        ]
        
        # Process concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    data={'error': str(result)},
                    model_used='error',
                    processing_time=0.0,
                    confidence=0.0,
                    errors=[str(result)],
                    cached=False
                ))
            else:
                processed_results.append(result)
        
        return processed_results
```
