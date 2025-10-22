"""
Model Selector - Intelligent VLA model selection
Selects optimal model based on document characteristics
"""

from typing import Tuple
from .vla_trigger import ComplexityLevel


class ModelSelector:
    """
    Select optimal VLA model based on document characteristics.
    """

    def __init__(self):
        """Initialize model capabilities matrix"""
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
        complexity_level: ComplexityLevel,
        document_type: str = 'general',
        language: str = 'en',
        resolution: Tuple[int, int] = (1024, 1024)
    ) -> str:
        """
        Select best model for document.

        Args:
            complexity_level: Document complexity (ComplexityLevel enum)
            document_type: Type of document (form/table/chart/diagram/general)
            language: Document language code
            resolution: Image resolution (width, height)

        Returns:
            Recommended model name
        """
        max_res = max(resolution)

        # Filter by resolution capability
        capable_models = [
            name for name, caps in self.model_capabilities.items()
            if caps['max_resolution'] >= max_res
        ]

        if not capable_models:
            # If resolution too high, use highest capacity model
            capable_models = ['mplug']

        # Filter by language support
        capable_models = [
            name for name in capable_models
            if language in self.model_capabilities[name]['languages']
            or 'multi' in self.model_capabilities[name]['languages']
        ]

        if not capable_models:
            # Fallback to multi-language models
            capable_models = [
                name for name in self.model_capabilities.keys()
                if 'multi' in self.model_capabilities[name]['languages']
            ]

        # Select based on document type
        if document_type in ['form', 'table', 'structured']:
            if 'layoutlm' in capable_models:
                return 'layoutlm'

        if document_type in ['chart', 'diagram', 'complex', 'high-res']:
            if 'mplug' in capable_models:
                return 'mplug'

        # Default selection based on complexity
        if complexity_level == ComplexityLevel.EXTREME:
            if 'mplug' in capable_models:
                return 'mplug'
            elif 'surya' in capable_models:
                return 'surya'
            else:
                return capable_models[0] if capable_models else 'paddleocr'

        elif complexity_level == ComplexityLevel.COMPLEX:
            if 'surya' in capable_models:
                return 'surya'
            elif 'mplug' in capable_models:
                return 'mplug'
            else:
                return capable_models[0] if capable_models else 'paddleocr'

        elif complexity_level == ComplexityLevel.MODERATE:
            if 'surya' in capable_models:
                return 'surya'
            elif 'paddleocr' in capable_models:
                return 'paddleocr'
            else:
                return capable_models[0] if capable_models else 'paddleocr'

        else:  # SIMPLE
            if 'paddleocr' in capable_models:
                return 'paddleocr'
            elif 'surya' in capable_models:
                return 'surya'
            else:
                return capable_models[0] if capable_models else 'paddleocr'

    def get_fallback_model(self, primary_model: str) -> str:
        """
        Get fallback model for a given primary model.

        Args:
            primary_model: Primary model name

        Returns:
            Fallback model name
        """
        # Fallback hierarchy
        fallback_map = {
            'mplug': 'surya',
            'surya': 'paddleocr',
            'layoutlm': 'paddleocr',
            'paddleocr': None  # No fallback for base model
        }

        return fallback_map.get(primary_model, 'paddleocr')

    def get_model_info(self, model_name: str) -> dict:
        """
        Get detailed information about a model.

        Args:
            model_name: Model name

        Returns:
            Model capabilities dict
        """
        return self.model_capabilities.get(model_name, {})

    def list_models(self, **filters) -> list:
        """
        List models matching given filters.

        Args:
            **filters: Key-value pairs to filter by (e.g., speed='fast')

        Returns:
            List of matching model names
        """
        matching = []

        for name, caps in self.model_capabilities.items():
            match = True
            for key, value in filters.items():
                if key not in caps:
                    match = False
                    break

                # Handle list filters (e.g., languages)
                if isinstance(caps[key], list):
                    if value not in caps[key]:
                        match = False
                        break
                else:
                    if caps[key] != value:
                        match = False
                        break

            if match:
                matching.append(name)

        return matching
