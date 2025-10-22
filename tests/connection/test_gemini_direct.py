#!/usr/bin/env python3
"""
Test script for direct Google Gemini API connection
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.translation.gemini_client import GeminiClient, TranslationRequest


async def test_gemini_direct():
    """Test direct Google Gemini API"""

    print("\n" + "=" * 80)
    print("GOOGLE GEMINI API - DIRECT CONNECTION TEST")
    print("=" * 80)
    print()

    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key or api_key == 'your_google_gemini_api_key_here':
        print("❌ GEMINI_API_KEY not set")
        print()
        print("To get your Google Gemini API key:")
        print("  1. Go to https://makersuite.google.com/app/apikey")
        print("  2. Create a new API key")
        print("  3. Set it in your .env file:")
        print("     GEMINI_API_KEY=your_key_here")
        print()
        print("Or export it:")
        print("  export GEMINI_API_KEY='your_key_here'")
        print()
        return False

    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    print()

    # Test 1: Simple translation
    print("=" * 80)
    print("TEST 1: Simple Translation (EN → ZH)")
    print("=" * 80)
    print()

    try:
        config = {
            'use_openrouter': False,  # Use direct Gemini API
            'model': 'gemini-2.0-flash-exp',
            'temperature': 0.3,
            'max_tokens': 200
        }

        async with GeminiClient(api_key=api_key, config=config) as client:
            request = TranslationRequest(
                text="Hello, world! This is a test.",
                source_lang="en",
                target_lang="zh",
                document_type="general"
            )

            print("Translating: 'Hello, world! This is a test.'")
            print()

            response = await client.translate(request)

            print(f"✅ Translation successful!")
            print(f"   Original:    {request.text}")
            print(f"   Translation: {response.translated_text}")
            print(f"   Confidence:  {response.confidence:.2f}")
            print(f"   Tokens used: {response.tokens_used}")
            print(f"   Model:       {response.model_used}")
            print(f"   Provider:    {response.metadata.get('provider')}")
            print()

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

    # Test 2: Document type translations
    print("=" * 80)
    print("TEST 2: Document Type Translations")
    print("=" * 80)
    print()

    test_cases = [
        {
            'text': 'This is a scientific paper about quantum mechanics.',
            'source': 'en',
            'target': 'zh',
            'type': 'scientific'
        },
        {
            'text': 'The contract terms are as follows.',
            'source': 'en',
            'target': 'es',
            'type': 'legal'
        },
        {
            'text': 'The CPU operates at 3.2GHz.',
            'source': 'en',
            'target': 'fr',
            'type': 'technical'
        }
    ]

    try:
        config = {
            'use_openrouter': False,
            'model': 'gemini-2.0-flash-exp',
            'temperature': 0.3,
            'max_tokens': 200,
            'use_advanced_prompts': True
        }

        async with GeminiClient(api_key=api_key, config=config) as client:
            for i, test_case in enumerate(test_cases, 1):
                print(f"Test {i}: {test_case['type'].upper()} ({test_case['source']} → {test_case['target']})")

                request = TranslationRequest(
                    text=test_case['text'],
                    source_lang=test_case['source'],
                    target_lang=test_case['target'],
                    document_type=test_case['type']
                )

                response = await client.translate(request)

                print(f"  Original:    {test_case['text']}")
                print(f"  Translation: {response.translated_text}")
                print(f"  Tokens:      {response.tokens_used}")
                print()

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

    # Test 3: Available models
    print("=" * 80)
    print("TEST 3: Available Gemini Models")
    print("=" * 80)
    print()

    print("Current recommended models:")
    print("  • gemini-2.0-flash-exp (Experimental, fast)")
    print("  • gemini-1.5-flash (Stable, fast)")
    print("  • gemini-1.5-pro (Most capable)")
    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("✅ Direct Google Gemini API connection: WORKING")
    print("✅ Translation: SUCCESSFUL")
    print("✅ Multiple document types: WORKING")
    print()
    print("🎉 Your Google Gemini API is configured correctly!")
    print()
    print("Configuration:")
    print(f"  Model: gemini-2.0-flash-exp")
    print(f"  Provider: Direct Google API")
    print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
    print()

    return True


def main():
    """Main entry point"""
    result = asyncio.run(test_gemini_direct())
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
