#!/usr/bin/env python3
"""
Complete Translation Test with Current Configuration
Tests the translation system with whatever API is configured
"""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.translation.gemini_client import GeminiClient, TranslationRequest


async def test_translation_system():
    """Test the complete translation system with current config"""

    print("\n" + "=" * 80)
    print("COMPLETE TRANSLATION SYSTEM TEST")
    print("=" * 80)
    print()

    # Detect which API to use
    use_openrouter = os.getenv('USE_OPENROUTER', 'false').lower() == 'true'

    if use_openrouter:
        api_key = os.getenv('OPENROUTER_API_KEY')
        model = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.5-flash')
        provider = "OpenRouter"
    else:
        api_key = os.getenv('GEMINI_API_KEY')
        model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        provider = "Direct Google Gemini API"

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"API Key: {api_key[:15] if api_key else 'NOT SET'}...")
    print()

    if not api_key or api_key.startswith('your_'):
        print("❌ API key not configured properly")
        print()
        if use_openrouter:
            print("Set OPENROUTER_API_KEY in .env file")
        else:
            print("Set GEMINI_API_KEY in .env file")
        return False

    # Test cases
    test_cases = [
        {
            'name': 'Simple Translation',
            'text': 'Hello, world! This is a test.',
            'source': 'en',
            'target': 'zh',
            'type': 'general'
        },
        {
            'name': 'Scientific Document',
            'text': 'This study investigates the quantum properties of photonic crystals.',
            'source': 'en',
            'target': 'zh',
            'type': 'scientific'
        },
        {
            'name': 'Technical Manual',
            'text': 'The system requires 8GB RAM and a 2GHz dual-core processor.',
            'source': 'en',
            'target': 'es',
            'type': 'technical'
        },
        {
            'name': 'Legal Document',
            'text': 'The parties hereby agree to the terms and conditions.',
            'source': 'en',
            'target': 'fr',
            'type': 'legal'
        },
        {
            'name': 'Business Document',
            'text': 'Our quarterly revenue exceeded projections by 15 percent.',
            'source': 'en',
            'target': 'de',
            'type': 'business'
        }
    ]

    config = {
        'use_openrouter': use_openrouter,
        'model': model,
        'temperature': 0.3,
        'max_tokens': 500,
        'use_advanced_prompts': True
    }

    print("=" * 80)
    print("RUNNING TRANSLATION TESTS")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    try:
        async with GeminiClient(api_key=api_key, config=config) as client:
            for i, test in enumerate(test_cases, 1):
                print(f"Test {i}: {test['name']}")
                print(f"  Type: {test['type'].upper()}")
                print(f"  {test['source'].upper()} → {test['target'].upper()}")
                print()
                print(f"  Original: {test['text']}")

                try:
                    request = TranslationRequest(
                        text=test['text'],
                        source_lang=test['source'],
                        target_lang=test['target'],
                        document_type=test['type']
                    )

                    response = await client.translate(request)

                    if response.confidence > 0:
                        print(f"  ✅ Translation: {response.translated_text}")
                        print(f"  Confidence: {response.confidence:.2f}")
                        print(f"  Tokens: {response.tokens_used}")
                        print(f"  Model: {response.model_used}")
                        passed += 1
                    else:
                        print(f"  ⚠️  Translation: {response.translated_text}")
                        print(f"  (Using fallback, API may have issues)")
                        failed += 1

                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    failed += 1

                print()

    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if passed == len(test_cases):
        print("🎉 ALL TRANSLATION TESTS PASSED!")
        print()
        print(f"Your {provider} configuration is working perfectly!")
        print(f"Model: {model}")
        print()
        return True
    elif passed > 0:
        print(f"⚠️  {passed}/{len(test_cases)} tests passed")
        print("Some translations are working, but there may be issues.")
        return True
    else:
        print("❌ All tests failed")
        print("Please check your API key and configuration.")
        return False


def main():
    result = asyncio.run(test_translation_system())
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
