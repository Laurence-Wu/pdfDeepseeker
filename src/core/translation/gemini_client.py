"""
Gemini Client - Direct Google Gemini API Integration
Supports both direct Google Gemini API and OpenRouter for flexibility.
"""

import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import backoff
from functools import lru_cache
import os

from .prompt_engine import PromptEngine, DocumentType, PromptOptimizer


@dataclass
class TranslationRequest:
    """Translation request structure"""
    text: str
    source_lang: str
    target_lang: str
    context: Optional[Dict] = None
    constraints: Optional[Dict] = None
    document_type: str = "general"
    max_length: Optional[int] = None


@dataclass
class TranslationResponse:
    """Translation response structure"""
    translated_text: str
    confidence: float
    alternatives: List[str]
    tokens_used: int
    model_used: str
    metadata: Dict


class GeminiClient:
    """
    Gemini client supporting both direct Google API and OpenRouter.
    """

    def __init__(self, api_key: str, config: Dict = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Google Gemini API key or OpenRouter API key
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}

        # Detect if using OpenRouter or direct Gemini API
        self.use_openrouter = self.config.get('use_openrouter', False)

        if self.use_openrouter:
            # OpenRouter configuration
            self.base_url = self.config.get(
                'base_url',
                'https://openrouter.ai/api/v1'
            )
            self.model = self.config.get(
                'model',
                'google/gemini-2.5-flash'
            )
        else:
            # Direct Google Gemini API configuration
            self.base_url = self.config.get(
                'base_url',
                'https://generativelanguage.googleapis.com/v1beta'
            )
            self.model = self.config.get(
                'model',
                'gemini-2.0-flash-exp'
            )

        # Request configuration
        self.timeout = self.config.get('timeout', 60)
        self.max_retries = self.config.get('max_retries', 3)

        # Generation parameters
        self.generation_params = {
            'temperature': self.config.get('temperature', 0.3),
            'top_p': self.config.get('top_p', 0.9),
            'max_tokens': self.config.get('max_tokens', 2048),
            'frequency_penalty': 0,
            'presence_penalty': 0
        }

        # Rate limiting
        self.rate_limiter = RateLimiter(
            calls_per_minute=self.config.get('rate_limit', 60)
        )

        # Prompt Engine integration
        self.prompt_engine = PromptEngine(config)
        self.prompt_optimizer = PromptOptimizer()
        self.use_advanced_prompts = self.config.get('use_advanced_prompts', True)

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def translate(
        self,
        request: TranslationRequest
    ) -> TranslationResponse:
        """
        Translate text using Gemini API (direct or via OpenRouter).

        Args:
            request: Translation request

        Returns:
            TranslationResponse with translated text
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Generate prompt
        prompt = self._generate_prompt(request)

        if self.use_openrouter:
            return await self._translate_openrouter(request, prompt)
        else:
            return await self._translate_direct(request, prompt)

    async def _translate_direct(
        self,
        request: TranslationRequest,
        prompt: str
    ) -> TranslationResponse:
        """Translate using direct Google Gemini API"""

        # Build the full prompt
        system_prompt = self._get_system_prompt(request)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        # Prepare Google Gemini API request
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            'contents': [{
                'parts': [{
                    'text': full_prompt
                }]
            }],
            'generationConfig': {
                'temperature': self.generation_params['temperature'],
                'topP': self.generation_params['top_p'],
                'maxOutputTokens': self.generation_params['max_tokens'],
            }
        }

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                url,
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Parse Gemini API response
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        translated_text = candidate['content']['parts'][0]['text']
                    else:
                        raise ValueError("Unexpected response format")
                else:
                    raise ValueError("No candidates in response")

                # Apply post-processing
                translated_text = self._post_process(translated_text, request)

                # Calculate tokens (approximate)
                tokens_used = data.get('usageMetadata', {}).get('totalTokenCount', 0)

                return TranslationResponse(
                    translated_text=translated_text,
                    confidence=0.85,  # Default confidence for Gemini
                    alternatives=[],
                    tokens_used=tokens_used,
                    model_used=self.model,
                    metadata={
                        'provider': 'google_gemini_direct',
                        'original_response': data
                    }
                )

        except aiohttp.ClientError as e:
            return await self._handle_error(e, request)

    async def _translate_openrouter(
        self,
        request: TranslationRequest,
        prompt: str
    ) -> TranslationResponse:
        """Translate using OpenRouter API"""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/pdf-translator',
            'X-Title': 'PDF Translation Pipeline'
        }

        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': self._get_system_prompt(request)
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            **self.generation_params
        }

        # Add constraints if present
        if request.max_length:
            payload['max_tokens'] = min(
                request.max_length * 2,
                self.generation_params['max_tokens']
            )

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Parse response
                translated_text = data['choices'][0]['message']['content']

                # Apply post-processing
                translated_text = self._post_process(translated_text, request)

                return TranslationResponse(
                    translated_text=translated_text,
                    confidence=self._calculate_confidence(data),
                    alternatives=self._extract_alternatives(data),
                    tokens_used=data.get('usage', {}).get('total_tokens', 0),
                    model_used=data.get('model', self.model),
                    metadata={
                        'provider': 'openrouter',
                        'original_response': data
                    }
                )

        except aiohttp.ClientError as e:
            return await self._handle_error(e, request)

    def _generate_prompt(self, request: TranslationRequest) -> str:
        """
        Generate translation prompt with constraints.

        Args:
            request: Translation request

        Returns:
            Formatted prompt string
        """
        # Use advanced PromptEngine if enabled
        if self.use_advanced_prompts:
            try:
                # Convert document_type string to DocumentType enum
                try:
                    doc_type = DocumentType(request.document_type)
                except ValueError:
                    doc_type = DocumentType.GENERAL

                # Build constraints dict
                constraints = request.constraints or {}
                if request.max_length:
                    constraints['max_length'] = request.max_length
                    constraints['current_length'] = len(request.text)

                # Generate advanced prompt
                prompt = self.prompt_engine.generate_prompt(
                    text=request.text,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    document_type=doc_type,
                    context=request.context,
                    constraints=constraints if constraints else None,
                    metadata=None
                )

                # Optimize for model
                prompt = self.prompt_optimizer.optimize_for_model(
                    prompt=prompt,
                    model=self.model,
                    token_limit=self.generation_params['max_tokens']
                )

                return prompt

            except Exception as e:
                # Fallback to simple prompt on error
                print(f"PromptEngine error, using fallback: {e}")

        # Fallback: Simple prompt generation
        prompt_parts = [
            f"Translate the following text from {request.source_lang} to {request.target_lang}."
        ]

        # Add length constraint if specified
        if request.max_length:
            prompt_parts.append(
                f"CRITICAL: The translation MUST be ≤{request.max_length} characters."
            )

        # Add context if provided
        if request.context:
            if request.context.get('document_type'):
                prompt_parts.append(
                    f"Context: This is from a {request.context['document_type']} document."
                )
            if request.context.get('terminology'):
                prompt_parts.append(
                    f"Use this terminology: {request.context['terminology']}"
                )

        # Add the text to translate
        prompt_parts.append(f"\nText to translate:\n{request.text}")

        # Add output format instruction
        prompt_parts.append("\nProvide only the translation, no explanations.")

        return "\n".join(prompt_parts)

    def _get_system_prompt(self, request: TranslationRequest) -> str:
        """Generate system prompt based on document type"""

        base_prompt = (
            "You are a professional document translator. "
            "Preserve formatting, style, and technical accuracy. "
            "Never add explanations or notes to the translation."
        )

        # Document-specific prompts
        doc_prompts = {
            'scientific': (
                "Maintain scientific terminology and formula references. "
                "Preserve citation formats and figure/table references."
            ),
            'legal': (
                "Use precise legal terminology. "
                "Maintain clause structure and numbering."
            ),
            'technical': (
                "Preserve technical specifications and measurements. "
                "Keep product names and codes unchanged."
            ),
            'medical': (
                "Use standard medical terminology. "
                "Maintain drug names and dosage formats."
            )
        }

        doc_specific = doc_prompts.get(request.document_type, "")
        return f"{base_prompt} {doc_specific}".strip()

    def _post_process(self, text: str, request: TranslationRequest) -> str:
        """
        Post-process translated text.

        Args:
            text: Raw translated text
            request: Original request

        Returns:
            Processed text
        """
        # Remove any potential wrapper text
        text = text.strip()

        # Check length constraint
        if request.max_length and len(text) > request.max_length:
            # Attempt to shorten
            text = self._shorten_text(text, request.max_length)

        # Preserve special markers
        if request.constraints:
            text = self._apply_constraints(text, request.constraints)

        return text

    def _shorten_text(self, text: str, max_length: int) -> str:
        """Intelligently shorten text to fit length constraint"""

        if len(text) <= max_length:
            return text

        # Try removing extra spaces first
        text = ' '.join(text.split())
        if len(text) <= max_length:
            return text

        # Use ellipsis if necessary
        if max_length > 3:
            return text[:max_length-3] + '...'

        return text[:max_length]

    def _apply_constraints(self, text: str, constraints: Dict) -> str:
        """
        Apply translation constraints.

        Args:
            text: Translated text
            constraints: Constraint dictionary

        Returns:
            Text with constraints applied
        """
        # Placeholder for constraint application logic
        # Can be extended based on specific constraint types
        return text

    def _calculate_confidence(self, response_data: Dict) -> float:
        """Calculate translation confidence score"""

        # Base confidence from model
        confidence = 0.85

        # Adjust based on response metadata
        if response_data.get('usage', {}).get('total_tokens', 0) > 1000:
            confidence *= 0.95  # Lower confidence for long texts

        # Check for multiple choices
        if len(response_data.get('choices', [])) > 1:
            confidence *= 0.9

        return min(confidence, 1.0)

    def _extract_alternatives(self, response_data: Dict) -> List[str]:
        """Extract alternative translations if available"""

        alternatives = []
        choices = response_data.get('choices', [])

        # Get alternatives from additional choices
        for choice in choices[1:3]:  # Max 2 alternatives
            if 'message' in choice and 'content' in choice['message']:
                alternatives.append(choice['message']['content'])

        return alternatives

    async def _handle_error(
        self,
        error: Exception,
        request: TranslationRequest
    ) -> TranslationResponse:
        """Handle translation errors with fallback"""

        # Log error
        print(f"Translation error: {error}")

        # Return error response
        return TranslationResponse(
            translated_text=request.text,  # Return original
            confidence=0.0,
            alternatives=[],
            tokens_used=0,
            model_used='error',
            metadata={
                'error': str(error),
                'fallback': True
            }
        )


class RateLimiter:
    """Simple async rate limiter"""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit slot"""
        async with self.lock:
            current = asyncio.get_event_loop().time()
            time_since_last = current - self.last_call

            if time_since_last < self.interval:
                await asyncio.sleep(self.interval - time_since_last)

            self.last_call = asyncio.get_event_loop().time()
