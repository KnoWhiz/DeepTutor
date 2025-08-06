#!/usr/bin/env python3
"""
Test script for Claude proxy integration.
This script tests the integration between the API handler and Claude proxy.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

def test_claude_proxy_integration():
    """Test the Claude proxy integration with the API handler."""
    print("🧪 Testing Claude Proxy Integration")
    print("="*50)
    
    try:
        from api_handler import ApiHandler
        
        # Test configuration
        config = {
            'llm_source': 'claude_proxy',
            'temperature': 0.7,
            'creative_temperature': 0.9,
            'stream': False,
            'openai_key_dir': '.'
        }
        
        print("1. Initializing API Handler...")
        handler = ApiHandler(config)
        print("✅ API Handler initialized successfully")
        
        print("\n2. Testing model creation...")
        basic_model = handler.models['basic']['instance']
        advanced_model = handler.models['advanced']['instance']
        creative_model = handler.models['creative']['instance']
        print("✅ All models created successfully")
        
        print(f"   - Basic model type: {type(basic_model)}")
        print(f"   - Advanced model type: {type(advanced_model)}")
        print(f"   - Creative model type: {type(creative_model)}")
        
        print("\n3. Testing basic model invocation...")
        test_message = "Hello! Please respond with a simple greeting."
        response = basic_model.invoke(test_message)
        print("✅ Basic model invocation successful")
        print(f"   Response: {response.content[:100]}...")
        
        print("\n4. Testing advanced model invocation...")
        response = advanced_model.invoke("What is 2+2? Please give a brief answer.")
        print("✅ Advanced model invocation successful")
        print(f"   Response: {response.content[:100]}...")
        
        print("\n5. Testing creative model invocation...")
        response = creative_model.invoke("Write a short creative story about a robot.")
        print("✅ Creative model invocation successful")
        print(f"   Response: {response.content[:100]}...")
        
        print("\n🎉 All tests passed! Claude proxy integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure the Claude proxy server is running")
        print("2. Check your environment variables")
        print("3. Run 'python setup_claude_proxy.py' to test the setup")
        return False


def test_streaming():
    """Test streaming functionality."""
    print("\n🔄 Testing Streaming Functionality")
    print("="*40)
    
    try:
        from api_handler import ApiHandler
        
        # Test configuration with streaming enabled
        config = {
            'llm_source': 'claude_proxy',
            'temperature': 0.7,
            'creative_temperature': 0.9,
            'stream': True,
            'openai_key_dir': '.'
        }
        
        print("1. Initializing streaming API Handler...")
        handler = ApiHandler(config)
        print("✅ Streaming API Handler initialized")
        
        print("\n2. Testing streaming response...")
        streaming_model = handler.models['basic']['instance']
        
        print("   Streaming response:")
        print("   ", end="", flush=True)
        
        for chunk in streaming_model.stream("Count from 1 to 5:"):
            print(chunk.content, end="", flush=True)
        
        print("\n✅ Streaming test successful")
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {str(e)}")
        return False


def test_fallback():
    """Test fallback functionality."""
    print("\n🔄 Testing Fallback Functionality")
    print("="*40)
    
    try:
        from api_handler import ApiHandler
        
        # Test configuration
        config = {
            'llm_source': 'claude_proxy',
            'temperature': 0.7,
            'creative_temperature': 0.9,
            'stream': False,
            'openai_key_dir': '.'
        }
        
        print("1. Testing fallback configuration...")
        handler = ApiHandler(config)
        
        # Check that backup model is Azure OpenAI
        backup_model = handler.models['backup']['instance']
        print(f"   Backup model type: {type(backup_model)}")
        
        if 'AzureChatOpenAI' in str(type(backup_model)):
            print("✅ Fallback to Azure OpenAI configured correctly")
        else:
            print("⚠️  Fallback model is not Azure OpenAI")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Claude Proxy Integration Test Suite")
    print("="*60)
    
    # Check environment
    required_vars = ['CLAUDE_PROXY_URL', 'AZURE_OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    # Run tests
    tests = [
        ("Basic Integration", test_claude_proxy_integration),
        ("Streaming", test_streaming),
        ("Fallback", test_fallback)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Claude proxy integration is ready to use.")
        print("\nNext steps:")
        print("1. Set 'llm_source': 'claude_proxy' in your config.json")
        print("2. Make sure the proxy server is running")
        print("3. Your application will now use Claude Code through the proxy")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 