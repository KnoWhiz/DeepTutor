"""
Test script to verify proxy integration with Claude Code.
This tests that requests are properly routed through the proxy server.
"""

import os
import sys
from pathlib import Path
import requests
import json

# Add the pipeline directory to Python path for imports
current_dir = Path(__file__).parent
pipeline_dir = current_dir.parent.parent.parent / "pipeline"
sys.path.append(str(pipeline_dir))

from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.config import load_config


def test_proxy_connection():
    """Test direct connection to the proxy server."""
    
    print("🔍 Testing Proxy Connection")
    print("=" * 50)
    
    proxy_url = "http://localhost:8082"
    
    try:
        # Test basic connectivity
        response = requests.get(f"{proxy_url}/health", timeout=5)
        print(f"✅ Proxy health check: {response.status_code}")
        
        # Test Claude API endpoint
        test_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Hello! Just say 'Hi' back."}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": "test-key"
        }
        
        response = requests.post(
            f"{proxy_url}/v1/messages",
            json=test_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Proxy API test successful")
            print(f"   Response: {result.get('content', [{}])[0].get('text', 'No text')[:100]}...")
        else:
            print(f"❌ Proxy API test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to proxy server")
        print("   Make sure the proxy is running on localhost:8082")
    except Exception as e:
        print(f"❌ Error testing proxy: {e}")


def test_pipeline_with_proxy():
    """Test that the pipeline uses the proxy correctly."""
    
    print("\n🧪 Testing Pipeline with Proxy")
    print("=" * 50)
    
    # Check environment variables
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"ANTHROPIC_BASE_URL: {base_url}")
    print(f"ANTHROPIC_API_KEY: {'Set' if api_key else 'Not set'}")
    
    if not base_url or "localhost" not in base_url:
        print("❌ ANTHROPIC_BASE_URL not set to proxy URL")
        print("   Set: export ANTHROPIC_BASE_URL=http://localhost:8082")
        return
    
    try:
        # Load configuration
        config = load_config()
        print(f"LLM Source: {config['llm']['llm_source']}")
        
        # Test getting a model instance
        llm = get_llm('advanced', config['llm'])
        
        # Check if it's using the proxy
        model_class = type(llm).__name__
        print(f"Model class: {model_class}")
        
        if 'Anthropic' in model_class:
            print("✅ Using Claude Code with proxy")
            
            # Check if base_url is set correctly
            if hasattr(llm, 'base_url'):
                print(f"Base URL: {llm.base_url}")
                if "localhost" in str(llm.base_url):
                    print("✅ Base URL points to proxy")
                else:
                    print("❌ Base URL does not point to proxy")
            else:
                print("❓ Cannot determine base URL")
        else:
            print("❌ Not using Claude Code model")
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        import traceback
        traceback.print_exc()


def test_proxy_model_mapping():
    """Test that the proxy correctly maps Claude models to target models."""
    
    print("\n🗺️ Testing Proxy Model Mapping")
    print("=" * 50)
    
    proxy_url = "http://localhost:8082"
    
    # Test different Claude models
    test_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-5-opus-20241022"
    ]
    
    for model in test_models:
        try:
            test_data = {
                "model": model,
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hi"}
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": "test-key"
            }
            
            response = requests.post(
                f"{proxy_url}/v1/messages",
                json=test_data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ {model} -> mapped successfully")
            else:
                print(f"❌ {model} -> mapping failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")


def main():
    """Run all proxy integration tests."""
    
    print("🚀 Claude Code Proxy Integration Test")
    print("=" * 60)
    
    # Test 1: Direct proxy connection
    test_proxy_connection()
    
    # Test 2: Pipeline integration
    test_pipeline_with_proxy()
    
    # Test 3: Model mapping
    test_proxy_model_mapping()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary")
    print("=" * 60)
    print("To use the proxy with your pipeline:")
    print("1. Start the proxy server: cd proxy/ && python start_proxy.py")
    print("2. Set environment: export ANTHROPIC_BASE_URL=http://localhost:8082")
    print("3. Set API key: export ANTHROPIC_API_KEY=any-value")
    print("4. Run your pipeline - all Claude Code requests will go through the proxy!")


if __name__ == "__main__":
    main()
