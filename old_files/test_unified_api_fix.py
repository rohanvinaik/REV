#!/usr/bin/env python3
"""
Test the fixed unified API to verify mock responses have been removed.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/rohanvinaik/REV')

async def test_unified_api_fix():
    """Test the unified API with real model responses."""
    
    print("=" * 70)
    print("UNIFIED API MOCK RESPONSE FIX TEST")
    print("=" * 70)
    
    # Import the UnifiedAPI class
    from src.api.unified_api import UnifiedVerificationAPI
    
    # Create API instance (without starting the server)
    api = UnifiedVerificationAPI()
    
    # Test cases
    test_cases = [
        {
            'name': 'Local Model Test',
            'model_id': 'pythia-70m',
            'challenge': 'What is the capital of France?',
            'should_work': True
        },
        {
            'name': 'Non-existent Model Test', 
            'model_id': 'nonexistent-model',
            'challenge': 'Test prompt',
            'should_work': False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 50)
        print(f"Model: {test_case['model_id']}")
        print(f"Challenge: {test_case['challenge']}")
        
        try:
            # Call the fixed _get_model_response method
            response = await api._get_model_response(
                model_id=test_case['model_id'],
                challenge=test_case['challenge'],
                api_configs=None
            )
            
            # Analyze response
            print(f"✅ SUCCESS: Got response")
            print(f"   Text: {response['text'][:100]}...")
            print(f"   Provider: {response['metadata'].get('provider', 'unknown')}")
            print(f"   Logits count: {len(response.get('logits', []))}")
            
            # Check if it's a real response (not mock)
            is_mock = False
            if response['text'].startswith("Response from"):
                is_mock = True
                print(f"   ❌ MOCK DATA DETECTED!")
            elif "Model activation summary" in response['text']:
                print(f"   ✅ REAL MODEL EXECUTION detected")
            elif response['metadata'].get('provider') in ['openai', 'anthropic', 'cohere', 'http']:
                print(f"   ✅ REAL API CALL detected")
            
            results.append({
                'test': test_case['name'],
                'success': True,
                'mock_detected': is_mock,
                'provider': response['metadata'].get('provider')
            })
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ ERROR: {error_msg}")
            
            # Check if it's a proper error (not mock fallback)
            if test_case['should_work']:
                print(f"   ⚠️  Expected success but got error")
                results.append({
                    'test': test_case['name'], 
                    'success': False,
                    'mock_detected': False,
                    'error': error_msg
                })
            else:
                print(f"   ✅ Expected error - no mock fallback")
                results.append({
                    'test': test_case['name'],
                    'success': True,  # Error is expected
                    'mock_detected': False,
                    'error': error_msg
                })
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful_tests = sum(1 for r in results if r['success'])
    mock_detections = sum(1 for r in results if r.get('mock_detected', False))
    
    print(f"Tests passed: {successful_tests}/{len(results)}")
    print(f"Mock responses detected: {mock_detections}")
    
    if mock_detections > 0:
        print(f"❌ FAILURE: Mock responses still present!")
        return False
    elif successful_tests == len(results):
        print(f"✅ SUCCESS: All tests passed, no mock responses detected!")
        return True
    else:
        print(f"⚠️  PARTIAL: Some tests failed but no mock responses")
        return True
    
    return results

if __name__ == "__main__":
    success = asyncio.run(test_unified_api_fix())
    exit(0 if success else 1)