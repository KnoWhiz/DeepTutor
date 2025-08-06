#!/usr/bin/env python3
"""
Correct setup for Claude Code Proxy integration.
This script sets up the proxy to work with Claude Code CLI and Azure OpenAI.
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def create_proxy_config():
    """Create the correct proxy configuration for Azure OpenAI."""
    print("🔧 Creating Claude Proxy Configuration for Azure OpenAI")
    print("="*60)
    
    # Get Azure OpenAI details
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    if not azure_key or not azure_endpoint:
        print("❌ Missing Azure OpenAI configuration")
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        return False
    
    # Create proxy environment file
    proxy_env_content = f"""# Claude Proxy Configuration for Azure OpenAI
OPENAI_API_KEY={azure_key}
OPENAI_BASE_URL={azure_endpoint}
BIG_MODEL={azure_deployment}
MIDDLE_MODEL={azure_deployment}
SMALL_MODEL={azure_deployment.replace('gpt-4', 'gpt-35-turbo')}

# Server settings
HOST=0.0.0.0
PORT=8082
LOG_LEVEL=INFO

# Security (optional)
ANTHROPIC_API_KEY=your-claude-key

# Performance
MAX_TOKENS_LIMIT=128000
REQUEST_TIMEOUT=90
"""
    
    proxy_env_path = Path("pipeline/science/features_lab/claude-code-proxy/.env")
    proxy_env_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(proxy_env_path, 'w') as f:
        f.write(proxy_env_content)
    
    print(f"✅ Proxy configuration created: {proxy_env_path}")
    return True

def create_claude_config():
    """Create Claude Code CLI configuration."""
    print("\n📝 Creating Claude Code CLI Configuration")
    print("="*50)
    
    # Create .claude directory in user home
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(exist_ok=True)
    
    # Create config file
    claude_config = {
        "default_model": "claude-3-5-sonnet-20241022",
        "api_base": "http://localhost:8082",
        "api_key": "your-claude-key",
        "organization": None
    }
    
    config_path = claude_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(claude_config, f, indent=2)
    
    print(f"✅ Claude config created: {config_path}")
    return True

def create_usage_examples():
    """Create usage examples."""
    print("\n📚 Creating Usage Examples")
    print("="*40)
    
    examples = {
        "basic_usage": {
            "description": "Basic Claude Code CLI usage with proxy",
            "command": "claude",
            "notes": "This will use Claude Code CLI with your Azure OpenAI server"
        },
        "streaming": {
            "description": "Streaming response",
            "command": "claude --stream",
            "notes": "Get streaming responses from Azure OpenAI"
        },
        "model_selection": {
            "description": "Use specific Claude model (maps to Azure deployment)",
            "command": "claude --model claude-3-5-opus-20241022",
            "notes": "Opus maps to your BIG_MODEL deployment"
        },
        "file_processing": {
            "description": "Process files with Claude Code",
            "command": "claude --file your_file.txt",
            "notes": "Process files using Azure OpenAI"
        }
    }
    
    examples_path = Path("pipeline/science/pipeline/claude_usage_examples.json")
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✅ Usage examples created: {examples_path}")
    return True

def print_setup_instructions():
    """Print the correct setup instructions."""
    print("\n" + "="*70)
    print("🚀 CORRECT CLAUDE PROXY SETUP FOR AZURE OPENAI")
    print("="*70)
    
    print("\n1. Start the Claude proxy server:")
    print("   cd pipeline/science/features_lab/claude-code-proxy")
    print("   python start_proxy.py")
    print("   # This will run on http://localhost:8082")
    
    print("\n2. Use Claude Code CLI:")
    print("   # Install Claude Code CLI if not already installed")
    print("   pip install anthropic")
    print("   ")
    print("   # Set environment variables")
    print("   export ANTHROPIC_BASE_URL=http://localhost:8082")
    print("   export ANTHROPIC_API_KEY=your-claude-key")
    print("   ")
    print("   # Use Claude Code CLI")
    print("   claude")
    
    print("\n3. Integration with your existing code:")
    print("   # Instead of using the API handler, use Claude Code CLI directly")
    print("   # The proxy will convert Claude requests to Azure OpenAI format")
    
    print("\n4. Model mapping:")
    print("   - claude-3-5-haiku → SMALL_MODEL (gpt-35-turbo)")
    print("   - claude-3-5-sonnet → MIDDLE_MODEL (your deployment)")
    print("   - claude-3-5-opus → BIG_MODEL (your deployment)")
    
    print("\n" + "="*70)

def test_proxy_connection():
    """Test the proxy connection."""
    print("\n🧪 Testing Proxy Connection")
    print("="*40)
    
    try:
        import httpx
        
        # Test basic connection
        response = httpx.get("http://localhost:8082/health")
        if response.status_code == 200:
            print("✅ Proxy server is running")
            return True
        else:
            print("❌ Proxy server not responding correctly")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to proxy: {str(e)}")
        print("Make sure the proxy server is running on http://localhost:8082")
        return False

def main():
    """Main setup function."""
    print("🔧 Correct Claude Proxy Setup for Azure OpenAI")
    print("="*60)
    
    # Check Azure OpenAI configuration
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("❌ Missing Azure OpenAI configuration")
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        print_setup_instructions()
        return False
    
    # Create configurations
    if not create_proxy_config():
        return False
    
    if not create_claude_config():
        return False
    
    if not create_usage_examples():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the proxy server")
    print("2. Use Claude Code CLI with the proxy")
    print("3. All Claude requests will be converted to Azure OpenAI format")
    
    print_setup_instructions()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 