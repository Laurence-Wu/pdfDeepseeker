"""
Translation Strategies - Multiple fallback approaches for difficult translations
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import asyncio
import re

from .gemini_client import GeminiClient, TranslationRequest, TranslationResponse


@dataclass
class StrategyResult:
    """Result from a translation strategy attempt"""
    success: bool
    translated_text: Optional[str]
    confidence: float
    strategy_name: str
    error_message: Optional[str] = None


class TranslationStrategy:
    """Base class for translation strategies"""

    def __init__(self, client: GeminiClient, config: Dict[str, Any]):
        self.client = client
        self.config = config

    async def attempt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[Dict] = None
    ) -> StrategyResult:
        """Attempt translation with this strategy"""
        raise NotImplementedError


class DirectTranslationStrategy(TranslationStrategy):
    """Standard direct translation"""

    async def attempt(self, text: str, source_lang: str, target_lang: str,
                     context: Optional[Dict] = None) -> StrategyResult:
        try:
            request = TranslationRequest(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                document_type="general"
            )

            response = await self.client.translate(request)

            return StrategyResult(
                success=True,
                translated_text=response.translated_text,
                confidence=response.confidence,
                strategy_name="direct"
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                translated_text=None,
                confidence=0.0,
                strategy_name="direct",
                error_message=str(e)
            )


class ExplicitInstructionStrategy(TranslationStrategy):
    """Add explicit translation instruction"""

    async def attempt(self, text: str, source_lang: str, target_lang: str,
                     context: Optional[Dict] = None) -> StrategyResult:
        try:
            # Add explicit instruction
            enhanced_text = f"Please translate the following text from {source_lang} to {target_lang}:\n\n{text}"

            request = TranslationRequest(
                text=enhanced_text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                document_type="technical"
            )

            response = await self.client.translate(request)

            return StrategyResult(
                success=True,
                translated_text=response.translated_text,
                confidence=response.confidence,
                strategy_name="explicit_instruction"
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                translated_text=None,
                confidence=0.0,
                strategy_name="explicit_instruction",
                error_message=str(e)
            )


class ContextEnhancedStrategy(TranslationStrategy):
    """Add contextual information to help translation"""

    async def attempt(self, text: str, source_lang: str, target_lang: str,
                     context: Optional[Dict] = None) -> StrategyResult:
        try:
            # Build context-rich prompt
            context_parts = []
            if context:
                if 'document_type' in context:
                    context_parts.append(f"Document type: {context['document_type']}")
                if 'location' in context:
                    context_parts.append(f"Location: {context['location']}")
                if 'surrounding_text' in context:
                    context_parts.append(f"Context: {context['surrounding_text']}")

            context_str = "\n".join(context_parts) if context_parts else ""

            enhanced_text = f"""Context information:
{context_str}

Text to translate from {source_lang} to {target_lang}:
{text}

Important: Translate all content, including technical terms and proper nouns when appropriate."""

            request = TranslationRequest(
                text=enhanced_text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                document_type="technical"
            )

            response = await self.client.translate(request)

            return StrategyResult(
                success=True,
                translated_text=response.translated_text,
                confidence=response.confidence,
                strategy_name="context_enhanced"
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                translated_text=None,
                confidence=0.0,
                strategy_name="context_enhanced",
                error_message=str(e)
            )


class BatchTranslationStrategy(TranslationStrategy):
    """Translate multiple items together with separators"""

    async def attempt(self, text: str, source_lang: str, target_lang: str,
                     context: Optional[Dict] = None) -> StrategyResult:
        try:
            # For batch mode, text should already contain separators
            batch_prompt = f"""Translate the following items from {source_lang} to {target_lang}.
Preserve the '|' separator between items.
Translate each item even if it's a single word or technical term.

Items to translate:
{text}"""

            request = TranslationRequest(
                text=batch_prompt,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                document_type="technical"
            )

            response = await self.client.translate(request)

            return StrategyResult(
                success=True,
                translated_text=response.translated_text,
                confidence=response.confidence,
                strategy_name="batch"
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                translated_text=None,
                confidence=0.0,
                strategy_name="batch",
                error_message=str(e)
            )


class WordByWordStrategy(TranslationStrategy):
    """Translate word-by-word with explicit marking"""

    async def attempt(self, text: str, source_lang: str, target_lang: str,
                     context: Optional[Dict] = None) -> StrategyResult:
        try:
            enhanced_text = f"""Translate this term to {target_lang} language.
Even if it's a technical term, provide the {target_lang} equivalent or transliteration.

Term: "{text}"

Provide only the translation, no explanation."""

            request = TranslationRequest(
                text=enhanced_text,
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                document_type="technical"
            )

            response = await self.client.translate(request)

            # Clean up response
            cleaned = response.translated_text.strip()
            # Remove quotes if present
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)

            return StrategyResult(
                success=True,
                translated_text=cleaned,
                confidence=response.confidence * 0.7,  # Lower confidence for word-by-word
                strategy_name="word_by_word"
            )
        except Exception as e:
            return StrategyResult(
                success=False,
                translated_text=None,
                confidence=0.0,
                strategy_name="word_by_word",
                error_message=str(e)
            )


class MultiStrategyTranslator:
    """Tries multiple translation strategies with fallbacks"""

    def __init__(self, client: GeminiClient, config: Dict[str, Any]):
        self.client = client
        self.config = config

        # Initialize strategies in order of preference
        self.strategies = [
            DirectTranslationStrategy(client, config),
            BatchTranslationStrategy(client, config),
            ExplicitInstructionStrategy(client, config),
            ContextEnhancedStrategy(client, config),
            WordByWordStrategy(client, config)
        ]

    async def translate_with_fallback(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[Dict] = None,
        delay_between_attempts: float = 2.0,
        try_all_strategies: bool = False
    ) -> Tuple[StrategyResult, List[str]]:
        """
        Try multiple strategies until one succeeds.

        Args:
            try_all_strategies: If False (default), only tries fallback strategies if first fails
                              If True, tries all strategies until one produces a translation

        Returns:
            Tuple of (best result, list of attempted strategy names)
        """
        attempted_strategies = []
        best_result = None

        # If try_all_strategies is False, only try first strategy unless it fails
        strategies_to_try = self.strategies if try_all_strategies else [self.strategies[0]]

        for strategy in strategies_to_try:
            strategy_name = strategy.__class__.__name__

            try:
                result = await strategy.attempt(text, source_lang, target_lang, context)
                attempted_strategies.append(strategy_name)

                # Check if translation actually changed the text
                if result.success and result.translated_text:
                    text_changed = result.translated_text != text

                    if text_changed:
                        # Success! Return immediately
                        return result, attempted_strategies
                    else:
                        # Text didn't change
                        if best_result is None or result.confidence > best_result.confidence:
                            best_result = result

                        # If not trying all strategies and first one didn't change text,
                        # now try remaining strategies
                        if not try_all_strategies:
                            strategies_to_try = self.strategies[1:]  # Try remaining strategies
                            await asyncio.sleep(delay_between_attempts)
                            continue

                # Delay before trying next strategy (only if trying all)
                if try_all_strategies:
                    await asyncio.sleep(delay_between_attempts)

            except Exception as e:
                attempted_strategies.append(f"{strategy_name} (failed: {str(e)})")
                # On exception, try next strategy
                if not try_all_strategies:
                    strategies_to_try = self.strategies[1:]
                continue

        # If we get here, no strategy produced a different translation
        # Return the best result we found (even if unchanged) or create a failure result
        if best_result:
            return best_result, attempted_strategies
        else:
            return StrategyResult(
                success=False,
                translated_text=text,  # Fallback to original
                confidence=0.0,
                strategy_name="all_failed",
                error_message="All translation strategies failed"
            ), attempted_strategies
