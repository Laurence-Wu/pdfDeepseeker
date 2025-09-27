# VLA Trigger - Part 2: Model Integration

## Overview
Integration layer for multiple Vision-Language models based on complexity requirements.

## Implementation

```python
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VLAProcessor:
    """
    Process documents using Vision-Language models.
    Manages multiple models and selects based on complexity.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize VLA Processor with models.
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.get('use_gpu', False) 
            else "cpu"
        )
        self.models = {}
        self.processors = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_models()
    
    def _load_models(self):
        """Load VLA models based on configuration"""
        
        # Primary models
        if self.config.get('enable_surya', True):
            self._load_surya()
        
        if self.config.get('enable_mplug', False):
            self._load_mplug_docowl()
        
        if self.config.get('enable_layoutlm', False):
            self._load_layoutlm()
        
        # Fallback to PaddleOCR
        self._load_paddleocr()
    
    def _load_surya(self):
        """
        Load Surya model for document understanding.
        Best balance of speed and accuracy.
        """
        try:
            from surya.model import load_model, load_processor
            from surya.ocr import OCRPredictor
            
            # Load model and processor
            self.models['surya'] = {
                'model': load_model(),
                'processor': load_processor(),
                'predictor': OCRPredictor()
            }
            
            if self.device.type == 'cuda':
                self.models['surya']['model'].to(self.device)
            
            print("✓ Surya model loaded")
            
        except Exception as e:
            print(f"Failed to load Surya: {e}")
            self.models['surya'] = None
    
    def _load_mplug_docowl(self):
        """
        Load mPLUG-DocOwl for complex documents.
        Best for extreme cases with 4K+ resolution.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_name = "mPLUG/DocOwl1.5"
            self.models['mplug'] = {
                'model': AutoModel.from_pretrained(model_name, trust_remote_code=True),
                'tokenizer': AutoTokenizer.from_pretrained(model_name)
            }
            
            if self.device.type == 'cuda':
                self.models['mplug']['model'].to(self.device)
            
            print("✓ mPLUG-DocOwl loaded")
            
        except Exception as e:
            print(f"Failed to load mPLUG-DocOwl: {e}")
            self.models['mplug'] = None
    
    def _load_layoutlm(self):
        """
        Load LayoutLMv3 for structured document understanding.
        Good for forms and tables.
        """
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
            
            model_name = "microsoft/layoutlmv3-base"
            self.models['layoutlm'] = {
                'model': LayoutLMv3ForTokenClassification.from_pretrained(model_name),
                'processor': LayoutLMv3Processor.from_pretrained(model_name)
            }
            
            if self.device.type == 'cuda':
                self.models['layoutlm']['model'].to(self.device)
            
            print("✓ LayoutLMv3 loaded")
            
        except Exception as e:
            print(f"Failed to load LayoutLMv3: {e}")
            self.models['layoutlm'] = None
    
    def _load_paddleocr(self):
        """
        Load PaddleOCR as fallback.
        Reliable for simple to moderate complexity.
        """
        try:
            from paddleocr import PaddleOCR
            
            self.models['paddleocr'] = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.device.type == 'cuda',
                show_log=False
            )
            
            print("✓ PaddleOCR loaded")
            
        except Exception as e:
            print(f"Failed to load PaddleOCR: {e}")
            self.models['paddleocr'] = None
    
    async def process_with_vla(
        self,
        image: np.ndarray,
        model_name: str,
        fallback: bool = True
    ) -> Dict:
        """
        Process image with specified VLA model.
        
        Args:
            image: Document image
            model_name: Model to use
            fallback: Whether to fallback on failure
            
        Returns:
            Extraction results
        """
        try:
            # Select processing function
            if model_name == 'surya' and self.models.get('surya'):
                result = await self._process_with_surya(image)
            elif model_name == 'mplug' and self.models.get('mplug'):
                result = await self._process_with_mplug(image)
            elif model_name == 'layoutlm' and self.models.get('layoutlm'):
                result = await self._process_with_layoutlm(image)
            elif model_name == 'paddleocr' and self.models.get('paddleocr'):
                result = await self._process_with_paddleocr(image)
            else:
                raise ValueError(f"Model {model_name} not available")
            
            return result
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            
            # Fallback to PaddleOCR
            if fallback and model_name != 'paddleocr':
                print("Falling back to PaddleOCR")
                return await self._process_with_paddleocr(image)
            
            raise e
    
    async def _process_with_surya(self, image: np.ndarray) -> Dict:
        """
        Process with Surya model.
        
        Returns:
            Structured extraction results
        """
        loop = asyncio.get_event_loop()
        
        def _run_surya():
            surya_models = self.models['surya']
            predictor = surya_models['predictor']
            
            # Run OCR
            result = predictor.predict(image)
            
            # Structure results
            structured = {
                'text_blocks': [],
                'layout': {},
                'confidence': []
            }
            
            for block in result.get('blocks', []):
                structured['text_blocks'].append({
                    'text': block['text'],
                    'bbox': block['bbox'],
                    'confidence': block.get('confidence', 0.9),
                    'font_size': block.get('font_size'),
                    'is_bold': block.get('is_bold', False),
                    'is_italic': block.get('is_italic', False)
                })
                
                structured['confidence'].append(block.get('confidence', 0.9))
            
            # Layout analysis
            structured['layout'] = {
                'columns': result.get('num_columns', 1),
                'reading_order': result.get('reading_order', []),
                'has_header': result.get('has_header', False),
                'has_footer': result.get('has_footer', False)
            }
            
            return structured
        
        return await loop.run_in_executor(self.executor, _run_surya)
    
    async def _process_with_mplug(self, image: np.ndarray) -> Dict:
        """
        Process with mPLUG-DocOwl for complex documents.
        
        Returns:
            High-resolution extraction results
        """
        loop = asyncio.get_event_loop()
        
        def _run_mplug():
            model_dict = self.models['mplug']
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']
            
            # Prepare image (support 4K resolution)
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            
            # Generate comprehensive prompt
            prompt = (
                "Extract all text content from this document. "
                "Identify tables, formulas, headers, and maintain layout structure. "
                "Provide bounding boxes for each element."
            )
            
            # Process
            inputs = tokenizer(prompt, return_tensors="pt")
            pixel_values = model.image_processor(pil_image, return_tensors="pt")
            
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                pixel_values = pixel_values.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pixel_values=pixel_values,
                    max_new_tokens=2048,
                    do_sample=False
                )
            
            # Decode
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse structured output
            return self._parse_mplug_output(result_text)
        
        return await loop.run_in_executor(self.executor, _run_mplug)
    
    async def _process_with_layoutlm(self, image: np.ndarray) -> Dict:
        """
        Process with LayoutLMv3 for structured understanding.
        
        Returns:
            Layout-aware extraction results
        """
        loop = asyncio.get_event_loop()
        
        def _run_layoutlm():
            model_dict = self.models['layoutlm']
            model = model_dict['model']
            processor = model_dict['processor']
            
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image)
            
            # Process image
            encoding = processor(
                pil_image,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            
            if self.device.type == 'cuda':
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Extract tokens and boxes
            tokens = processor.tokenizer.convert_ids_to_tokens(
                encoding['input_ids'].squeeze().tolist()
            )
            boxes = encoding.get('bbox', [[0, 0, 0, 0]] * len(tokens))
            
            # Structure results
            structured = {
                'text_blocks': [],
                'layout': {},
                'entities': []
            }
            
            # Group tokens into text blocks
            current_block = {'text': '', 'bbox': None, 'label': None}
            
            for token, box, label in zip(tokens, boxes, predictions):
                if token in ['[PAD]', '[SEP]', '[CLS]']:
                    continue
                
                if token.startswith('##'):
                    current_block['text'] += token[2:]
                else:
                    if current_block['text']:
                        structured['text_blocks'].append(current_block.copy())
                    current_block = {
                        'text': token,
                        'bbox': box,
                        'label': model.config.id2label.get(label, 'text')
                    }
            
            # Add last block
            if current_block['text']:
                structured['text_blocks'].append(current_block)
            
            # Identify layout elements
            structured['layout'] = {
                'headers': [b for b in structured['text_blocks'] if b.get('label') == 'header'],
                'tables': [b for b in structured['text_blocks'] if b.get('label') == 'table'],
                'lists': [b for b in structured['text_blocks'] if b.get('label') == 'list']
            }
            
            return structured
        
        return await loop.run_in_executor(self.executor, _run_layoutlm)
    
    async def _process_with_paddleocr(self, image: np.ndarray) -> Dict:
        """
        Process with PaddleOCR as reliable fallback.
        
        Returns:
            OCR extraction results
        """
        loop = asyncio.get_event_loop()
        
        def _run_paddle():
            ocr = self.models['paddleocr']
            
            # Run OCR
            result = ocr.ocr(image, cls=True)
            
            # Structure results
            structured = {
                'text_blocks': [],
                'confidence': []
            }
            
            for line in result[0] if result and result[0] else []:
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                structured['text_blocks'].append({
                    'text': text,
                    'bbox': {
                        'x': min(p[0] for p in bbox),
                        'y': min(p[1] for p in bbox),
                        'width': max(p[0] for p in bbox) - min(p[0] for p in bbox),
                        'height': max(p[1] for p in bbox) - min(p[1] for p in bbox)
                    },
                    'confidence': confidence
                })
                
                structured['confidence'].append(confidence)
            
            return structured
        
        return await loop.run_in_executor(self.executor, _run_paddle)
    
    def _parse_mplug_output(self, text: str) -> Dict:
        """Parse mPLUG model output into structured format"""
        
        # This would parse the model's text output
        # Implementation depends on mPLUG output format
        structured = {
            'text_blocks': [],
            'tables': [],
            'formulas': [],
            'layout': {}
        }
        
        # Parse logic here...
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                structured['text_blocks'].append({
                    'text': line.strip(),
                    'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 20}
                })
        
        return structured
    
    async def process_batch(
        self,
        images: List[np.ndarray],
        model_name: str
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of images
            model_name: Model to use
            
        Returns:
            List of extraction results
        """
        tasks = [
            self.process_with_vla(img, model_name)
            for img in images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing image {i}: {result}")
                # Fallback to empty result
                processed_results.append({'text_blocks': [], 'error': str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
```

## Model Selection Strategy

```python
class ModelSelector:
    """
    Select optimal VLA model based on document characteristics.
    """
    
    def __init__(self):
        self.model_capabilities = {
            'surya': {
                'speed': 'fast',
                'accuracy': 'high',
                'languages': ['en', 'zh', 'ja', 'ko', 'ar', 'hi'],
                'best_for': ['general', 'multi-column', 'mixed-script'],
                'max_resolution': 2048
            },
            'mplug': {
                'speed': 'slow',
                'accuracy': 'very_high',
                'languages': ['en', 'zh', 'multi'],
                'best_for': ['complex', 'charts', 'diagrams', 'high-res'],
                'max_resolution': 4096
            },
            'layoutlm': {
                'speed': 'medium',
                'accuracy': 'high',
                'languages': ['en'],
                'best_for': ['forms', 'tables', 'structured'],
                'max_resolution': 1024
            },
            'paddleocr': {
                'speed': 'very_fast',
                'accuracy': 'good',
                'languages': ['en', 'ch', 'multi'],
                'best_for': ['simple', 'plain-text'],
                'max_resolution': 2048
            }
        }
    
    def select_model(
        self,
        complexity_level: str,
        document_type: str,
        language: str,
        resolution: Tuple[int, int]
    ) -> str:
        """
        Select best model for document.
        
        Args:
            complexity_level: Document complexity
            document_type: Type of document
            language: Document language
            resolution: Image resolution
            
        Returns:
            Recommended model name
        """
        max_res = max(resolution)
        
        # Filter by resolution capability
        capable_models = [
            name for name, caps in self.model_capabilities.items()
            if caps['max_resolution'] >= max_res
        ]
        
        # Filter by language support
        capable_models = [
            name for name in capable_models
            if language in self.model_capabilities[name]['languages']
            or 'multi' in self.model_capabilities[name]['languages']
        ]
        
        # Select based on document type
        if document_type in ['form', 'table']:
            if 'layoutlm' in capable_models:
                return 'layoutlm'
        
        if document_type in ['chart', 'diagram', 'complex']:
            if 'mplug' in capable_models:
                return 'mplug'
        
        # Default selection based on complexity
        if complexity_level == 'extreme':
            return 'mplug' if 'mplug' in capable_models else 'surya'
        elif complexity_level == 'complex':
            return 'surya' if 'surya' in capable_models else 'mplug'
        elif complexity_level == 'moderate':
            return 'surya' if 'surya' in capable_models else 'paddleocr'
        else:
            return 'paddleocr'
```
