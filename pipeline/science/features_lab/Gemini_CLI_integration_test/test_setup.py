#!/usr/bin/env python3
"""
Test script to verify the Gemini CLI chatbot setup.

This script tests the core functionality without launching the full Streamlit interface.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gemini_cli_handler import GeminiCLIHandler, main_streaming_function
    from pdf_converter import PDFConverter
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_environment_setup():
    """Test if environment is set up correctly."""
    print("\n🔍 Testing environment setup...")
    
    # Test Gemini CLI handler initialization
    try:
        handler = GeminiCLIHandler()
        print("✅ GeminiCLIHandler initialized successfully")
    except Exception as e:
        print(f"❌ GeminiCLIHandler initialization failed: {e}")
        return False
    
    # Test PDF converter
    try:
        converter = PDFConverter()
        print("✅ PDFConverter initialized successfully")
    except Exception as e:
        print(f"❌ PDFConverter initialization failed: {e}")
        return False
    
    return True


def test_working_directory():
    """Test working directory creation."""
    print("\n📁 Testing working directory setup...")
    
    working_dir = Path("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files")
    
    try:
        working_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Working directory created: {working_dir}")
        
        # Test write permissions
        test_file = working_dir / "test_write.txt"
        test_file.write_text("test content")
        test_file.unlink()  # Clean up
        print("✅ Write permissions verified")
        
        return True
    except Exception as e:
        print(f"❌ Working directory setup failed: {e}")
        return False


def create_test_file():
    """Create a simple test file for testing."""
    print("\n📝 Creating test file...")
    
    working_dir = Path("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files")
    test_file = working_dir / "test_document.txt"
    
    test_content = """
# Test Document for Gemini CLI Chatbot

## Introduction
This is a test document created to verify the functionality of the Gemini CLI chatbot integration.

## Key Points
1. **Artificial Intelligence**: The field of AI is rapidly evolving with new breakthroughs in machine learning.
2. **Natural Language Processing**: NLP techniques are becoming more sophisticated for understanding human language.
3. **Research Applications**: AI is being applied to solve complex research problems across various domains.

## Methodology
The test involves creating a simple text document with structured content that can be analyzed by the Gemini CLI system.

## Conclusion
This document serves as a basic test case for the chatbot's document analysis capabilities.

## References
- Sample reference for testing citation functionality
- Another reference to test bibliography generation
"""
    
    try:
        test_file.write_text(test_content.strip())
        print(f"✅ Test file created: {test_file}")
        return str(test_file)
    except Exception as e:
        print(f"❌ Failed to create test file: {e}")
        return None


def test_basic_functionality(test_file_path: str):
    """Test basic functionality with a simple query."""
    print("\n🧪 Testing basic functionality...")
    print("Note: This will make a real API call to Gemini CLI")
    
    # Simple test query
    test_query = "What are the main topics discussed in this document?"
    
    try:
        print(f"📤 Query: {test_query}")
        print("🤖 Gemini CLI Response:")
        print("-" * 50)
        
        # Test the streaming function
        response_chunks = []
        for chunk in main_streaming_function(test_file_path, test_query):
            print(chunk)
            response_chunks.append(chunk)
        
        print("-" * 50)
        
        if response_chunks:
            print("✅ Streaming function works correctly")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Gemini CLI Chatbot Setup Test")
    print("=" * 50)
    
    # Test environment setup
    if not test_environment_setup():
        print("\n❌ Environment setup test failed")
        sys.exit(1)
    
    # Test working directory
    if not test_working_directory():
        print("\n❌ Working directory test failed")
        sys.exit(1)
    
    # Create test file
    test_file_path = create_test_file()
    if not test_file_path:
        print("\n❌ Test file creation failed")
        sys.exit(1)
    
    # Ask user if they want to test API functionality
    print("\n🔑 API Test Option")
    print("The next test will make a real API call to Gemini CLI.")
    print("This requires a valid GEMINI_API_KEY in your .env file.")
    
    response = input("\nDo you want to test API functionality? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        if test_basic_functionality(test_file_path):
            print("\n✅ All tests passed! The chatbot is ready to use.")
        else:
            print("\n⚠️  Basic setup works, but API test failed.")
            print("Please check your GEMINI_API_KEY in the .env file.")
    else:
        print("\n✅ Basic setup tests passed!")
        print("To test API functionality, run this script again and choose 'y'.")
    
    # Clean up test file
    try:
        Path(test_file_path).unlink()
        print(f"🗑️  Test file cleaned up: {test_file_path}")
    except Exception:
        pass  # Ignore cleanup errors
    
    print("\n🚀 Ready to launch the chatbot with: python run_chatbot.py")


if __name__ == "__main__":
    main() 