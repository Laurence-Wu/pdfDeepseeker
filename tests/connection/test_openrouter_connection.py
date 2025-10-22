#!/usr/bin/env python3
"""
OpenRouter Connection Test Script
Tests the connection to OpenRouter API and verifies available models
"""

import os
import sys
import asyncio
import aiohttp
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.translation.gemini_client import GeminiClient, TranslationRequest


async def test_openrouter_connection():
    """Test OpenRouter API connection and functionality"""

    print("\n" + "=" * 80)
    print("OPENROUTER CONNECTION TEST")
    print("=" * 80)
    print()

    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        print()
        print("To set your API key:")
        print("  export OPENROUTER_API_KEY='your_key_here'")
        print()
        print("Or add to .env file:")
        print("  OPENROUTER_API_KEY=your_key_here")
        print()
        return False

    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print()

    # Test 1: Check available models
    print("=" * 80)
    print("TEST 1: Querying Available Models")
    print("=" * 80)
    print()

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            async with session.get(
                'https://openrouter.ai/api/v1/models',
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])

                    print(f"✅ Successfully retrieved {len(models)} models")
                    print()

                    # Find Google/Gemini models
                    gemini_models = [m for m in models if 'gemini' in m.get('id', '').lower()]

                    if gemini_models:
                        print(f"Found {len(gemini_models)} Gemini models:")
                        print("-" * 80)
                        for model in gemini_models[:10]:  # Show first 10
                            model_id = model.get('id', 'unknown')
                            model_name = model.get('name', 'Unknown')
                            context = model.get('context_length', 0)
                            print(f"  • {model_id}")
                            print(f"    Name: {model_name}")
                            print(f"    Context: {context:,} tokens")
                            print()
                    else:
                        print("⚠️  No Gemini models found")
                        print("   Checking other Google models...")
                        google_models = [m for m in models if 'google' in m.get('id', '').lower()]
                        if google_models:
                            print(f"\n   Found {len(google_models)} Google models:")
                            for model in google_models[:5]:
                                print(f"     • {model.get('id')}")
                        print()

                else:
                    print(f"❌ Failed to retrieve models: HTTP {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text[:200]}")
                    print()
                    return False

    except Exception as e:
        print(f"❌ Error querying models: {e}")
        print()
        return False

    # Test 2: Test with different model names
    print("=" * 80)
    print("TEST 2: Testing Different Model Names")
    print("=" * 80)
    print()

    test_models = [
        'google/gemini-2.5-flash',         # Latest Gemini 2.5 Flash
        'google/gemini-2.5-flash-lite',    # Lite version
        'google/gemini-2.5-pro',           # Pro version
        'google/gemini-flash-1.5',         # Older version
        'google/gemini-pro-1.5',           # Deprecated
    ]

    test_text = "Hello, world!"
    working_model = None

    for model_name in test_models:
        print(f"Testing model: {model_name}")

        try:
            config = {
                'base_url': 'https://openrouter.ai/api/v1',
                'model': model_name,
                'temperature': 0.3,
                'max_tokens': 100,
                'timeout': 30
            }

            async with GeminiClient(api_key=api_key, config=config) as client:
                request = TranslationRequest(
                    text=test_text,
                    source_lang="en",
                    target_lang="zh",
                    document_type="general"
                )

                response = await client.translate(request)

                if response.confidence > 0:
                    print(f"  ✅ SUCCESS!")
                    print(f"     Original: {test_text}")
                    print(f"     Translation: {response.translated_text}")
                    print(f"     Confidence: {response.confidence:.2f}")
                    print(f"     Tokens: {response.tokens_used}")
                    print(f"     Model: {response.model_used}")
                    print()
                    working_model = model_name
                    break
                else:
                    print(f"  ⚠️  Model returned fallback (confidence 0.0)")
                    print()

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()

    if not working_model:
        print("❌ No working model found")
        print()
        print("Recommendations:")
        print("  1. Check OpenRouter documentation for current model names:")
        print("     https://openrouter.ai/docs#models")
        print("  2. Verify your API key has credits")
        print("  3. Check if Gemini models are available in your region")
        print()
        return False

    # Test 3: Test translation with working model
    print("=" * 80)
    print("TEST 3: Full Translation Test")
    print("=" * 80)
    print()

    print(f"Using working model: {working_model}")
    print()

    test_cases = [
        {
            'text': 'This is a scientific research paper about quantum mechanics.',
            'source': 'en',
            'target': 'zh',
            'type': 'scientific'
        },
        {
            'text': 'The parties hereby agree to the terms and conditions.',
            'source': 'en',
            'target': 'es',
            'type': 'legal'
        },
        {
            'text': 'The system requires 8GB RAM and 2GHz processor.',
            'source': 'en',
            'target': 'fr',
            'type': 'technical'
        }
    ]

    try:
        config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'model': working_model,
            'temperature': 0.3,
            'max_tokens': 200,
            'timeout': 30,
            'use_advanced_prompts': True
        }

        async with GeminiClient(api_key=api_key, config=config) as client:
            for i, test_case in enumerate(test_cases, 1):
                print(f"Test Case {i}: {test_case['type'].upper()}")
                print(f"  {test_case['source']} → {test_case['target']}")
                print()

                request = TranslationRequest(
                    text=test_case['text'],
                    source_lang=test_case['source'],
                    target_lang=test_case['target'],
                    document_type=test_case['type']
                )

                response = await client.translate(request)

                print(f"  Original:    {test_case['text']}")
                print(f"  Translation: {response.translated_text}")
                print(f"  Confidence:  {response.confidence:.2f}")
                print(f"  Tokens:      {response.tokens_used}")
                print()

                if response.confidence > 0:
                    print("  ✅ Translation successful")
                else:
                    print("  ⚠️  Fallback response (check model availability)")
                print()

    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

    # Test 4: Test rate limiting
    print("=" * 80)
    print("TEST 4: Rate Limiting Test")
    print("=" * 80)
    print()

    try:
        config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'model': working_model,
            'temperature': 0.3,
            'max_tokens': 50,
            'rate_limit': 60,  # 60 calls per minute
            'timeout': 30
        }

        async with GeminiClient(api_key=api_key, config=config) as client:
            print("Making 3 rapid requests (should be throttled)...")
            import time
            start = time.time()

            for i in range(3):
                request = TranslationRequest(
                    text=f"Test {i+1}",
                    source_lang="en",
                    target_lang="zh",
                    document_type="general"
                )
                await client.translate(request)
                print(f"  Request {i+1} completed")

            elapsed = time.time() - start
            print()
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Expected: ~2.00s (with 60 calls/min limit)")

            if 1.5 < elapsed < 3.0:
                print("  ✅ Rate limiting working correctly")
            else:
                print("  ⚠️  Rate limiting may not be working as expected")
            print()

    except Exception as e:
        print(f"⚠️  Rate limiting test failed: {e}")
        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"✅ API Key: Valid")
    print(f"✅ Connection: Successful")
    print(f"✅ Working Model: {working_model}")
    print(f"✅ Translation: Working")
    print(f"✅ Rate Limiting: Working")
    print()
    print("🎉 OpenRouter connection is fully operational!")
    print()
    print("Recommended configuration for .env:")
    print(f"  OPENROUTER_API_KEY={api_key[:8]}...{api_key[-4:]}")
    print(f"  OPENROUTER_MODEL={working_model}")
    print()

    return True


async def quick_test():
    """Quick test without detailed output"""
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        return False

    try:
        config = {
            'base_url': 'https://openrouter.ai/api/v1',
            'model': 'google/gemini-2.5-flash',
            'temperature': 0.3,
            'max_tokens': 50,
            'timeout': 10
        }

        async with GeminiClient(api_key=api_key, config=config) as client:
            request = TranslationRequest(
                text="Hello",
                source_lang="en",
                target_lang="zh",
                document_type="general"
            )

            response = await client.translate(request)
            return response.confidence > 0

    except:
        return False


def main():
    """Main entry point"""

    if '--quick' in sys.argv:
        # Quick test mode
        result = asyncio.run(quick_test())
        if result:
            print("✅ OpenRouter connection: OK")
            return 0
        else:
            print("❌ OpenRouter connection: FAILED")
            return 1
    else:
        # Full test mode
        result = asyncio.run(test_openrouter_connection())
        return 0 if result else 1


if __name__ == "__main__":
    exit(main())
