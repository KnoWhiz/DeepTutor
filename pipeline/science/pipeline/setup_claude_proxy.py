#!/usr/bin/env python3
"""
Setup script for Claude Code Proxy integration.
This script helps configure and test the Claude proxy integration.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from claude_proxy_wrapper import ClaudeProxyClient, ClaudeProxyError


def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment configuration...")
    
    required_vars = [
        "CLAUDE_PROXY_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    print("✅ Environment variables are properly configured")
    return True


async def test_proxy_connection():
    """Test connection to the Claude proxy server."""
    print("\n🔗 Testing Claude proxy connection...")
    
    proxy_url = os.getenv("CLAUDE_PROXY_URL", "http://localhost:8082")
    proxy_key = os.getenv("CLAUDE_PROXY_KEY", "your-key")
    
    try:
        async with ClaudeProxyClient(base_url=proxy_url, api_key=proxy_key) as client:
            # Test health check
            print(f"Testing connection to {proxy_url}...")
            
            # Simple test message
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello! Please respond with 'Connection successful'"}],
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                temperature=0.1
            )
            
            print(f"✅ Proxy connection successful!")
            print(f"Response: {response.text}")
            return True
            
    except Exception as e:
        print(f"❌ Proxy connection failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the Claude proxy server is running")
        print("2. Check that CLAUDE_PROXY_URL is correct")
        print("3. Verify the proxy server is accessible")
        return False


def test_api_handler_integration():
    """Test the API handler integration."""
    print("\n🧪 Testing API handler integration...")
    
    try:
        from api_handler import ApiHandler
        
        # Create a test configuration
        test_config = {
            'llm_source': 'claude_proxy',
            'temperature': 0.7,
            'creative_temperature': 0.9,
            'stream': False,
            'openai_key_dir': '.'
        }
        
        # Initialize API handler
        handler = ApiHandler(test_config)
        
        # Test model creation
        basic_model = handler.get_models(
            api_key="test-key",
            temperature=0.7,
            deployment_name='gpt-4.1-mini',
            host='claude_proxy',
            stream=False
        )
        
        print("✅ API handler integration successful!")
        print(f"Model type: {type(basic_model)}")
        return True
        
    except Exception as e:
        print(f"❌ API handler integration failed: {str(e)}")
        return False


def create_config_example():
    """Create an example configuration file."""
    print("\n📝 Creating example configuration...")
    
    config_example = {
        "llm_source": "claude_proxy",
        "temperature": 0.7,
        "creative_temperature": 0.9,
        "stream": False,
        "openai_key_dir": ".",
        "embedding_model": "text-embedding-ada-002"
    }
    
    config_path = Path(__file__).parent / "config_claude_proxy_example.json"
    
    with open(config_path, 'w') as f:
        json.dump(config_example, f, indent=2)
    
    print(f"✅ Example configuration created: {config_path}")
    return True


def print_setup_instructions():
    """Print setup instructions."""
    print("\n" + "="*60)
    print("🚀 CLAUDE PROXY SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Start the Claude proxy server:")
    print("   cd pipeline/science/features_lab/claude-code-proxy")
    print("   python start_proxy.py")
    
    print("\n2. Configure your environment variables:")
    print("   - Copy claude_proxy_config_example.env to your .env file")
    print("   - Update the values with your actual API keys")
    
    print("\n3. Update your config.json:")
    print("   - Set 'llm_source' to 'claude_proxy'")
    print("   - Use config_claude_proxy_example.json as reference")
    
    print("\n4. Test the integration:")
    print("   python setup_claude_proxy.py")
    
    print("\n5. Use in your application:")
    print("   - The API handler will automatically use Claude Code")
    print("   - All existing code will work without changes")
    print("   - Fallback to Azure OpenAI if Claude proxy fails")
    
    print("\n" + "="*60)


async def main():
    """Main setup function."""
    print("🔧 Claude Proxy Integration Setup")
    print("="*40)
    
    # Check environment
    if not check_environment():
        print_setup_instructions()
        return
    
    # Test proxy connection
    if not await test_proxy_connection():
        print_setup_instructions()
        return
    
    # Test API handler integration
    if not test_api_handler_integration():
        print_setup_instructions()
        return
    
    # Create example config
    create_config_example()
    
    print("\n🎉 Setup completed successfully!")
    print("Your Claude proxy integration is ready to use.")
    print("\nTo use it in your application:")
    print("1. Set 'llm_source': 'claude_proxy' in your config.json")
    print("2. Make sure the proxy server is running")
    print("3. All existing API calls will now use Claude Code through the proxy")


if __name__ == "__main__":
    asyncio.run(main()) 