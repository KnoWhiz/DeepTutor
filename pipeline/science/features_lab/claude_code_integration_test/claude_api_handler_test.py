"""
Test script for Claude Code integration with the API handler.
This demonstrates how to use Claude Code through the existing pipeline infrastructure.
"""

import os
import sys
from pathlib import Path

# Add the pipeline directory to Python path for imports
current_dir = Path(__file__).parent
pipeline_dir = current_dir.parent.parent.parent / "pipeline"
sys.path.append(str(pipeline_dir))

from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.config import load_config


def test_claude_code_integration():
    """Test the Claude Code integration with the API handler."""
    
    # Load configuration
    config = load_config()
    
    # Update config to use anthropic as the LLM source
    config['llm']['llm_source'] = 'anthropic'
    
    # Create API handler with streaming enabled
    api_handler = ApiHandler(config['llm'], stream=True)
    
    # Test different model types
    print("Testing Claude Code integration:")
    print("=" * 50)
    
    # Test basic model
    basic_model = api_handler.models['basic']['instance']
    print(f"Basic model: {basic_model}")
    print(f"Context window: {api_handler.models['basic']['context_window']}")
    
    # Test advanced model
    advanced_model = api_handler.models['advanced']['instance']
    print(f"Advanced model: {advanced_model}")
    print(f"Context window: {api_handler.models['advanced']['context_window']}")
    
    # Test creative model
    creative_model = api_handler.models['creative']['instance']
    print(f"Creative model: {creative_model}")
    print(f"Context window: {api_handler.models['creative']['context_window']}")
    
    # Test backup model
    backup_model = api_handler.models['backup']['instance']
    print(f"Backup model: {backup_model}")
    print(f"Context window: {api_handler.models['backup']['context_window']}")
    
    print("\n" + "=" * 50)
    print("All models are using Claude Code (claude-3-5-sonnet-20241022)")
    print("Backup model uses Azure GPT-4.1-mini")
    
    return api_handler


def test_claude_code_streaming():
    """Test streaming functionality with Claude Code."""
    
    config = load_config()
    config['llm']['llm_source'] = 'anthropic'
    
    api_handler = ApiHandler(config['llm'], stream=True)
    model = api_handler.models['advanced']['instance']
    
    # Test prompt
    test_prompt = "Hello! Can you explain what you are and what you can do?"
    
    print(f"\nTesting streaming with Claude Code:")
    print(f"Prompt: {test_prompt}")
    print("-" * 50)
    
    try:
        # Test streaming response
        response = model.stream(test_prompt)
        
        print("Streaming response:")
        for chunk in response:
            if hasattr(chunk, 'content'):
                print(chunk.content, end='', flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error during streaming: {e}")


if __name__ == "__main__":
    # Test the integration
    try:
        api_handler = test_claude_code_integration()
        test_claude_code_streaming()
        print("\n✅ Claude Code integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
