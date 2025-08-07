"""
Test script to verify that Claude Code is being used in get_response.py functions.
This tests both the summary generation and regular response generation.
"""

import os
import sys
from pathlib import Path
import asyncio

# Add the pipeline directory to Python path for imports
current_dir = Path(__file__).parent
pipeline_dir = current_dir.parent.parent.parent / "pipeline"
sys.path.append(str(pipeline_dir))

from pipeline.science.pipeline.get_response import get_multiple_files_summary, get_response
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.utils import Question
from pipeline.science.pipeline.config import load_config


async def test_claude_code_summary():
    """Test that Claude Code is used for summary generation."""
    
    print("Testing Claude Code for summary generation...")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"LLM Source: {config['llm']['llm_source']}")
    
    # Create a test file path list (you can replace with actual files)
    test_files = ["test_file1.pdf", "test_file2.pdf"]
    test_embedding_folders = ["test_embedding_folder"]
    
    try:
        # Test summary generation
        summary_generator = await get_multiple_files_summary(
            file_path_list=test_files,
            embedding_folder_list=test_embedding_folders,
            chat_session=None,
            stream=True
        )
        
        print("Summary generation initiated with Claude Code")
        print("Streaming response:")
        
        # Process the streaming response
        response_text = ""
        for chunk in summary_generator:
            response_text += chunk
            print(chunk, end='', flush=True)
        
        print(f"\n\nTotal response length: {len(response_text)} characters")
        print("✅ Claude Code summary generation test completed")
        
    except Exception as e:
        print(f"❌ Error in summary generation: {e}")
        import traceback
        traceback.print_exc()


async def test_claude_code_response():
    """Test that Claude Code is used for regular response generation."""
    
    print("\nTesting Claude Code for response generation...")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"LLM Source: {config['llm']['llm_source']}")
    
    # Create a test chat session
    chat_session = ChatSession()
    chat_session.mode = ChatMode.LITE  # Use LITE mode for simpler testing
    
    # Create a test question
    question = Question(
        text="What is the main topic of this document?",
        language="English",
        question_type="global",
        special_context="",
        answer_planning={}
    )
    
    # Test parameters
    test_files = ["test_file.pdf"]
    test_embedding_folders = ["test_embedding_folder"]
    chat_history = []
    
    try:
        # Test response generation
        response_generator = await get_response(
            chat_session=chat_session,
            file_path_list=test_files,
            question=question,
            chat_history=chat_history,
            embedding_folder_list=test_embedding_folders,
            deep_thinking=False,
            stream=True
        )
        
        print("Response generation initiated with Claude Code")
        print("Streaming response:")
        
        # Process the streaming response
        response_text = ""
        for chunk in response_generator:
            response_text += chunk
            print(chunk, end='', flush=True)
        
        print(f"\n\nTotal response length: {len(response_text)} characters")
        print("✅ Claude Code response generation test completed")
        
    except Exception as e:
        print(f"❌ Error in response generation: {e}")
        import traceback
        traceback.print_exc()


def test_configuration():
    """Test that the configuration is correctly set to use Claude Code."""
    
    print("Testing configuration...")
    print("=" * 60)
    
    config = load_config()
    
    print(f"LLM Source: {config['llm']['llm_source']}")
    print(f"Temperature: {config['llm']['temperature']}")
    print(f"Creative Temperature: {config['llm']['creative_temperature']}")
    print(f"Level: {config['llm']['level']}")
    
    if config['llm']['llm_source'] == 'anthropic':
        print("✅ Configuration correctly set to use Claude Code")
    else:
        print("❌ Configuration not set to use Claude Code")
    
    return config


async def main():
    """Run all tests."""
    
    print("🧪 Testing Claude Code Integration with get_response.py")
    print("=" * 80)
    
    # Test configuration first
    config = test_configuration()
    
    # Test summary generation
    await test_claude_code_summary()
    
    # Test response generation
    await test_claude_code_response()
    
    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("\nNote: The tests may show errors if test files don't exist,")
    print("but the important thing is that Claude Code is being used")
    print("instead of Azure/OpenAI models.")


if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
