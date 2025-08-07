"""
Simple verification script to confirm Claude Code is being used.
"""

import os
import sys
from pathlib import Path

# Add the pipeline directory to Python path for imports
current_dir = Path(__file__).parent
pipeline_dir = current_dir.parent.parent.parent / "pipeline"
sys.path.append(str(pipeline_dir))

from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler


def verify_claude_code_usage():
    """Verify that Claude Code is being used instead of Azure/OpenAI models."""
    
    print("🔍 Verifying Claude Code Usage")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration LLM Source: {config['llm']['llm_source']}")
    
    # Test different model types
    model_types = ['basic', 'advanced', 'creative', 'backup']
    
    for model_type in model_types:
        print(f"\n--- Testing {model_type.upper()} model ---")
        
        try:
            # Get the model instance
            llm = get_llm(model_type, config['llm'])
            
            # Check the model type
            model_class = type(llm).__name__
            print(f"Model class: {model_class}")
            
            # Check if it's Claude Code
            if 'Anthropic' in model_class:
                print("✅ Using Claude Code (Anthropic)")
            elif 'Azure' in model_class:
                print("❌ Using Azure model")
            elif 'OpenAI' in model_class:
                print("❌ Using OpenAI model")
            elif 'SambaNova' in model_class:
                print("❌ Using SambaNova model")
            else:
                print(f"❓ Unknown model type: {model_class}")
            
            # Print model details
            if hasattr(llm, 'model'):
                print(f"Model name: {llm.model}")
            if hasattr(llm, 'temperature'):
                print(f"Temperature: {llm.temperature}")
            if hasattr(llm, 'max_tokens'):
                print(f"Max tokens: {llm.max_tokens}")
                
        except Exception as e:
            print(f"❌ Error getting {model_type} model: {e}")
    
    print("\n" + "=" * 50)
    
    # Test API handler directly
    print("\n--- Testing API Handler ---")
    try:
        api_handler = ApiHandler(config['llm'], stream=False)
        
        print("Available models:")
        for model_name, model_info in api_handler.models.items():
            model_instance = model_info['instance']
            model_class = type(model_instance).__name__
            context_window = model_info['context_window']
            
            print(f"  {model_name}: {model_class} (context: {context_window:,} tokens)")
            
            if 'Anthropic' in model_class:
                print(f"    ✅ Claude Code model")
            else:
                print(f"    ❌ Not Claude Code")
                
    except Exception as e:
        print(f"❌ Error with API handler: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Verification Complete!")
    
    if config['llm']['llm_source'] == 'anthropic':
        print("✅ Configuration is set to use Claude Code")
    else:
        print("❌ Configuration is NOT set to use Claude Code")


if __name__ == "__main__":
    verify_claude_code_usage()
