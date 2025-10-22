"""
Gemini Client - Interfaces with Gemini via OpenRouter (China-compatible).
Handles translation requests with rate limiting and error handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """Translation request data."""
    text: str
    source_lang: str
    target_lang: str
    max_length: Optional[int] = None
    preserve_formatting: bool = True


@dataclass
class TranslationResponse:
    """Translation response data."""
    translated_text: str
    confidence: float
    processing_time: float
    tokens_used: int
    cost: float


class GeminiClient:
    """
    Client for Google Gemini AI via OpenRouter API.
    Provides China-compatible access to Gemini models.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 model: str = "google/gemini-pro-1.5", timeout: int = 60):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

        # Rate limiting
        self.requests_per_minute = 60
        self.last_request_time = None
        self.request_count = 0
        self.window_start = datetime.utcnow()

        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://pdf-translator.example.com",
            "X-Title": "PDF Translation Pipeline"
        }

    async def translate_xliff(self, xliff_content: str, source_lang: str,
                            target_lang: str) -> str:
        """
        Translate XLIFF content.

        Args:
            xliff_content: XLIFF document content
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated XLIFF content
        """
        logger.info(f"Translating XLIFF from {source_lang} to {target_lang}")

        # Parse XLIFF to extract translatable content
        units = self._parse_xliff_for_translation(xliff_content)

        # Translate each unit
        translated_units = []
        for unit in units:
            translated_unit = await self._translate_unit(unit, source_lang, target_lang)
            translated_units.append(translated_unit)

        # Reconstruct XLIFF with translations
        return self._rebuild_xliff_with_translations(xliff_content, translated_units)

    async def _translate_unit(self, unit: Dict[str, Any], source_lang: str,
                            target_lang: str) -> Dict[str, Any]:
        """
        Translate a single XLIFF unit.

        Args:
            unit: XLIFF unit data
            source_lang: Source language
            target_lang: Target language

        Returns:
            Unit with translated content
        """
        source_text = unit.get('source', '')
        max_length = unit.get('max_length')

        if not source_text:
            return unit

        # Apply rate limiting
        await self._apply_rate_limit()

        # Generate prompt for translation
        prompt = self._generate_translation_prompt(
            source_text, source_lang, target_lang, max_length
        )

        # Make request to OpenRouter
        response = await self._make_translation_request(prompt, source_lang, target_lang)

        if response:
            unit['target'] = response.translated_text
            unit['translation_metadata'] = {
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'tokens_used': response.tokens_used,
                'cost': response.cost
            }

        return unit

    async def _make_translation_request(self, prompt: str, source_lang: str,
                                      target_lang: str) -> Optional[TranslationResponse]:
        """
        Make translation request to OpenRouter API.

        Args:
            prompt: Translation prompt
            source_lang: Source language
            target_lang: Target language

        Returns:
            TranslationResponse or None if failed
        """
        request_data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.9
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=request_data
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_translation_response(data)
                else:
                    logger.error(f"Translation API error: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Translation request failed: {str(e)}")
            return None

    def _generate_translation_prompt(self, text: str, source_lang: str,
                                   target_lang: str, max_length: Optional[int] = None) -> str:
        """
        Generate translation prompt for Gemini.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum character length constraint

        Returns:
            Generated prompt
        """
        prompt_parts = [
            f"Translate the following text from {source_lang} to {target_lang}:",
            "",
            "IMPORTANT INSTRUCTIONS:",
            "- Preserve exact formatting and structure",
            "- Maintain the same character count when possible",
            "- Do not translate formulas, code, or technical identifiers",
            "- Keep line breaks and spacing",
        ]

        if max_length:
            prompt_parts.append(f"- Output must be ≤{max_length} characters")
            prompt_parts.append("- If impossible, abbreviate non-critical words")

        prompt_parts.extend([
            "",
            "Text to translate:",
            text
        ])

        return "\n".join(prompt_parts)

    def _parse_xliff_for_translation(self, xliff_content: str) -> List[Dict[str, Any]]:
        """
        Parse XLIFF content to extract translatable units.

        Args:
            xliff_content: XLIFF document content

        Returns:
            List of translatable units
        """
        # This is a simplified parser - in production, use a proper XLIFF parser
        units = []

        try:
            # Basic XML parsing for translatable units
            # Look for <trans-unit> elements
            import re

            # Find all trans-unit elements
            trans_unit_pattern = r'<trans-unit[^>]*id="([^"]*)"[^>]*>.*?</trans-unit>'
            units_matches = re.findall(trans_unit_pattern, xliff_content, re.DOTALL)

            for unit_id in units_matches:
                # Extract source content
                source_pattern = rf'<source[^>]*>(.*?)</source>'
                source_match = re.search(source_pattern, xliff_content, re.DOTALL)

                if source_match:
                    units.append({
                        'id': unit_id,
                        'source': source_match.group(1).strip(),
                        'target': '',
                        'max_length': None  # Will be set from constraints
                    })

        except Exception as e:
            logger.error(f"Failed to parse XLIFF: {str(e)}")

        return units

    def _rebuild_xliff_with_translations(self, original_xliff: str,
                                       translated_units: List[Dict[str, Any]]) -> str:
        """
        Rebuild XLIFF document with translated content.

        Args:
            original_xliff: Original XLIFF content
            translated_units: Translated units

        Returns:
            Updated XLIFF content
        """
        # This is a simplified rebuilder - in production, use proper XML manipulation
        updated_xliff = original_xliff

        for unit in translated_units:
            if unit.get('target'):
                # Replace target content
                unit_id = unit['id']
                target_text = unit['target']

                # Simple string replacement (not robust for production)
                pattern = rf'<target[^>]*>.*?</target>'
                replacement = f'<target>{target_text}</target>'

                # This is a simplified approach - proper implementation would use XML parsing
                updated_xliff = updated_xliff.replace(
                    f'<target></target>',  # Assuming empty target
                    replacement
                )

        return updated_xliff

    def _parse_translation_response(self, response_data: Dict[str, Any]) -> TranslationResponse:
        """
        Parse OpenRouter API response.

        Args:
            response_data: API response data

        Returns:
            TranslationResponse object
        """
        try:
            content = response_data['choices'][0]['message']['content']
            usage = response_data['usage']

            return TranslationResponse(
                translated_text=content.strip(),
                confidence=0.95,  # Placeholder - could be derived from response
                processing_time=0.1,  # Placeholder
                tokens_used=usage.get('total_tokens', 0),
                cost=0.001  # Placeholder - calculate based on token usage
            )

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse translation response: {str(e)}")
            return TranslationResponse(
                translated_text="",
                confidence=0.0,
                processing_time=0.0,
                tokens_used=0,
                cost=0.0
            )

    async def _apply_rate_limit(self):
        """Apply rate limiting to API requests."""
        now = datetime.utcnow()

        # Reset window if needed
        if now - self.window_start > timedelta(minutes=1):
            self.request_count = 0
            self.window_start = now

        # Check if we've exceeded rate limit
        if self.request_count >= self.requests_per_minute:
            # Calculate wait time
            wait_time = 60 - (now - self.window_start).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.window_start = datetime.utcnow()

        self.request_count += 1
        self.last_request_time = now

    async def health_check(self) -> bool:
        """
        Check if the OpenRouter API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception:
            return False
