#!/usr/bin/env python3
"""
Test script for Claude Code SDK chatbot implementation.

This script demonstrates how to use the get_claude_code_response function
with the same interface as the original get_response function.
"""

import os
import sys
from claude_code_sdk import get_claude_code_response, ChatSession, Question, ChatMode

def test_basic_functionality():
    """Test basic functionality of the Claude Code SDK chatbot."""
    
    # Create a test session
    session = ChatSession()
    session.initialize()
    session.set_mode(ChatMode.ADVANCED)
    
    # Create a test question
    question = Question(
        text="Can you analyze the code structure and suggest improvements?",
        language="English",
        question_type="global",
        special_context="Focus on code quality and best practices"
    )
    
    # Test file paths (adjust these to your actual files)
    file_paths = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline/get_response.py"
    ]
    
    # Codebase folder directory
    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline"
    
    # Empty chat history for testing
    chat_history = []
    
    print("Testing Claude Code SDK chatbot...")
    print("=" * 50)
    print(f"Question: {question.text}")
    print(f"Files to analyze: {file_paths}")
    print(f"Codebase directory: {codebase_dir}")
    print("=" * 50)
    
    try:
        # Test the function
        response_generator = get_claude_code_response(
            chat_session=session,
            file_path_list=file_paths,
            question=question,
            chat_history=chat_history,
            codebase_folder_dir=codebase_dir,
            deep_thinking=True,
            stream=True
        )
        
        print("Streaming response:")
        print("-" * 30)
        
        for chunk in response_generator:
            print(chunk, end="", flush=True)
            
        print("\n" + "-" * 30)
        print("Test completed successfully!")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    
    return True

def test_with_different_questions():
    """Test with different types of questions."""
    
    session = ChatSession()
    session.initialize()
    
    # Test questions
    test_questions = [
        Question(
            text="What are the main functions in this codebase?",
            language="English",
            question_type="global"
        ),
        Question(
            text="How does the get_response function work?",
            language="English", 
            question_type="local",
            special_context="Focus on the streaming implementation"
        ),
        Question(
            text="Can you generate a unit test for the ChatSession class?",
            language="English",
            question_type="global"
        )
    ]
    
    file_paths = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline/session_manager.py"
    ]
    
    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/pipeline"
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question.text}")
        print(f"{'='*60}")
        
        try:
            response_generator = get_claude_code_response(
                chat_session=session,
                file_path_list=file_paths,
                question=question,
                chat_history=[],
                codebase_folder_dir=codebase_dir,
                deep_thinking=True,
                stream=True
            )
            
            for chunk in response_generator:
                print(chunk, end="", flush=True)
                
        except Exception as e:
            print(f"Error in test {i}: {e}")

if __name__ == "__main__":
    print("Claude Code SDK Chatbot Test Suite")
    print("=" * 40)
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set your API key in the .env file or environment.")
        sys.exit(1)
    
    # Run basic test
    print("\n1. Running basic functionality test...")
    success = test_basic_functionality()
    
    if success:
        print("\n2. Running additional question tests...")
        test_with_different_questions()
    
    print("\n" + "="*40)
    print("All tests completed!")
