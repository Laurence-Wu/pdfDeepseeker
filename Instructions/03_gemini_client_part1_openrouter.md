# Gemini Client Implementation - Part 1: OpenRouter Integration

## Overview
GeminiClient handles translation via OpenRouter API for China compatibility, not direct Google API.

## Core Implementation

```python
import aiohttp
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import backoff
from functools import lru_cache

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
    OpenRouter-based client for Gemini translation.
    Provides China-compatible access to Google's Gemini models.
    """
    
    def __init__(self, api_key: str, config: Dict = None):
        """
        Initialize OpenRouter client for Gemini.
        
        Args:
            api_key: OpenRouter API key
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config or {}
        
        # OpenRouter configuration
        self.base_url = self.config.get(
            'base_url', 
            'https://openrouter.ai/api/v1'
        )
        self.model = self.config.get(
            'model', 
            'google/gemini-pro-1.5'
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
        Translate text using Gemini via OpenRouter.
        
        Args:
            request: Translation request
            
        Returns:
            TranslationResponse with translated text
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Generate prompt
        prompt = self._generate_prompt(request)
        
        # Prepare OpenRouter request
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
                request.max_length * 2,  # Approximate token count
                self.generation_params['max_tokens']
            )
        
        try:
            # Make request to OpenRouter
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
                translated_text = self._post_process(
                    translated_text, 
                    request
                )
                
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
            # Handle API errors
            return await self._handle_error(e, request)
    
    def _generate_prompt(self, request: TranslationRequest) -> str:
        """
        Generate translation prompt with constraints.
        
        Args:
            request: Translation request
            
        Returns:
            Formatted prompt string
        """
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
```

## Usage Example

```python
async def translate_document():
    config = {
        'base_url': 'https://openrouter.ai/api/v1',
        'model': 'google/gemini-pro-1.5',
        'temperature': 0.3,
        'timeout': 60
    }
    
    async with GeminiClient(api_key="your_openrouter_key", config=config) as client:
        request = TranslationRequest(
            text="要翻译的文本",
            source_lang="zh",
            target_lang="en",
            max_length=50,
            document_type="technical"
        )
        
        response = await client.translate(request)
        print(f"Translation: {response.translated_text}")
        print(f"Confidence: {response.confidence}")
        print(f"Tokens used: {response.tokens_used}")

# Run
asyncio.run(translate_document())
```

## Configuration Notes

### OpenRouter Headers
- Always include HTTP-Referer for tracking
- X-Title helps with usage analytics
- Authorization uses OpenRouter API key, not Google's

### Model Selection
- Primary: `google/gemini-pro-1.5` (best quality)
- Fallback: `anthropic/claude-3-opus` (alternative)
- Budget: `meta-llama/llama-3.1-70b` (cost-effective)

### China Compatibility
- OpenRouter provides stable access from China
- No VPN required
- Automatic fallback to available routes
