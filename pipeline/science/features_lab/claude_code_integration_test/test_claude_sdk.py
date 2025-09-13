#!/usr/bin/env python3
"""Test script for Claude Code SDK integration.

This script tests the functionality of the Claude Code SDK chatbot implementation.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict
import os
from dotenv import load_dotenv

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from claude_code_sdk import (
    ChatSession,
    Question,
    ClaudeCodeSDK,
    get_response,
    simple_query
)


class TestColors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print a colored header."""
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'=' * 80}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}{text}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}{'=' * 80}{TestColors.ENDC}\n")


def print_test(test_name: str):
    """Print test name."""
    print(f"{TestColors.CYAN}{TestColors.BOLD}Testing: {test_name}{TestColors.ENDC}")


def print_success(message: str):
    """Print success message."""
    print(f"{TestColors.GREEN}✓ {message}{TestColors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{TestColors.FAIL}✗ {message}{TestColors.ENDC}")


def print_response(response: str):
    """Print the response with formatting."""
    print(f"{TestColors.BLUE}Response:{TestColors.ENDC}")
    print(response)


async def test_basic_functionality():
    """Test basic SDK functionality."""
    print_header("Test 1: Basic Functionality")
    
    try:
        # Test SDK initialization
        print_test("SDK Initialization")
        codebase_path = "./test_files"
        sdk = ClaudeCodeSDK(codebase_path)
        print_success("SDK initialized successfully")
        
        # Test file loading
        print_test("File Loading")
        files = sdk.load_codebase_files()
        print_success(f"Loaded {len(files)} files from codebase")
        for file_path in files.keys():
            print(f"  - {file_path}")
        
        # Test context formatting
        print_test("Context Formatting")
        context = sdk.format_codebase_context(files)
        print_success(f"Context formatted ({len(context)} characters)")
        
        return True
    except Exception as e:
        print_error(f"Basic functionality test failed: {e}")
        return False


async def test_simple_query():
    """Test simple query functionality."""
    print_header("Test 2: Simple Query")
    
    try:
        print_test("Running simple query about hello_world function")
        
        response_chunks = []
        async for chunk in simple_query(
            question_text="What does the hello_world function do in sample_code.py?",
            file_paths=["sample_code.py"]
        ):
            response_chunks.append(chunk)
        
        response = "".join(response_chunks)
        print_success("Query completed successfully")
        print_response(response[:500] + "..." if len(response) > 500 else response)
        
        return True
    except Exception as e:
        print_error(f"Simple query test failed: {e}")
        return False


async def test_streaming_response():
    """Test streaming response functionality."""
    print_header("Test 3: Streaming Response")
    
    try:
        # Create test session
        session = ChatSession(
            session_id="test_stream_001",
            mode="ADVANCED",
            model="claude-3-5-sonnet-20241022"
        )
        
        # Create question
        question = Question(
            text="Explain the Calculator class and its methods",
            special_context="Focus on the implementation details"
        )
        
        print_test("Streaming response for Calculator class analysis")
        
        # Get streaming response
        response_chunks = []
        chunk_count = 0
        
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample_code.py"],
            question=question,
            chat_history=[],
            embedding_folder_list=[],
            deep_thinking=True,
            stream=True
        ):
            response_chunks.append(chunk)
            chunk_count += 1
            # Print progress dots
            if chunk_count % 10 == 0:
                print(".", end="", flush=True)
        
        print()  # New line after dots
        response = "".join(response_chunks)
        print_success(f"Received {chunk_count} chunks")
        print_response(response[:500] + "..." if len(response) > 500 else response)
        
        return True
    except Exception as e:
        print_error(f"Streaming response test failed: {e}")
        return False


async def test_chat_history():
    """Test chat with history functionality."""
    print_header("Test 4: Chat with History")
    
    try:
        session = ChatSession(
            session_id="test_history_001",
            mode="ADVANCED"
        )
        
        # First question
        print_test("First question about JavaScript code")
        question1 = Question(text="What does the greet function do?")
        
        chat_history = []
        response1_chunks = []
        
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample.js"],
            question=question1,
            chat_history=chat_history,
            stream=True
        ):
            response1_chunks.append(chunk)
        
        response1 = "".join(response1_chunks)
        print_success("First question answered")
        
        # Update chat history
        chat_history.append({"role": "user", "content": question1.text})
        chat_history.append({"role": "assistant", "content": response1})
        
        # Follow-up question
        print_test("Follow-up question with context")
        question2 = Question(text="Can you also explain the Person class?")
        
        response2_chunks = []
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample.js"],
            question=question2,
            chat_history=chat_history,
            stream=True
        ):
            response2_chunks.append(chunk)
        
        response2 = "".join(response2_chunks)
        print_success("Follow-up question answered with history context")
        print_response(response2[:500] + "..." if len(response2) > 500 else response2)
        
        return True
    except Exception as e:
        print_error(f"Chat history test failed: {e}")
        return False


async def test_multi_file_analysis():
    """Test analysis across multiple files."""
    print_header("Test 5: Multi-file Analysis")
    
    try:
        session = ChatSession(
            session_id="test_multi_001",
            mode="ADVANCED"
        )
        
        question = Question(
            text="Compare the Python and JavaScript implementations. What are the main differences in syntax and structure?"
        )
        
        print_test("Analyzing multiple files simultaneously")
        
        response_chunks = []
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample_code.py", "sample.js"],
            question=question,
            chat_history=[],
            stream=True
        ):
            response_chunks.append(chunk)
        
        response = "".join(response_chunks)
        print_success("Multi-file analysis completed")
        print_response(response[:500] + "..." if len(response) > 500 else response)
        
        return True
    except Exception as e:
        print_error(f"Multi-file analysis test failed: {e}")
        return False


async def test_deep_thinking_mode():
    """Test deep thinking mode."""
    print_header("Test 6: Deep Thinking Mode")
    
    try:
        session = ChatSession(
            session_id="test_deep_001",
            mode="ADVANCED"
        )
        
        question = Question(
            text="How would you refactor the Calculator class to follow SOLID principles? Provide specific code examples."
        )
        
        print_test("Deep thinking mode ON")
        
        # Test with deep thinking ON
        response_deep_chunks = []
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample_code.py"],
            question=question,
            chat_history=[],
            deep_thinking=True,
            stream=True
        ):
            response_deep_chunks.append(chunk)
        
        response_deep = "".join(response_deep_chunks)
        
        print_test("Deep thinking mode OFF")
        
        # Test with deep thinking OFF
        response_normal_chunks = []
        async for chunk in await get_response(
            chat_session=session,
            file_path_list=["sample_code.py"],
            question=question,
            chat_history=[],
            deep_thinking=False,
            stream=True
        ):
            response_normal_chunks.append(chunk)
        
        response_normal = "".join(response_normal_chunks)
        
        print_success(f"Deep thinking response: {len(response_deep)} chars")
        print_success(f"Normal response: {len(response_normal)} chars")
        
        if len(response_deep) > len(response_normal):
            print_success("Deep thinking mode provides more detailed response")
        
        return True
    except Exception as e:
        print_error(f"Deep thinking mode test failed: {e}")
        return False


async def run_interactive_mode():
    """Run an interactive chat session."""
    print_header("Interactive Mode")
    print("Type 'quit' or 'exit' to stop the interactive session.\n")
    
    session = ChatSession(
        session_id="interactive_001",
        mode="ADVANCED"
    )
    
    chat_history = []
    
    # Load all files initially
    sdk = ClaudeCodeSDK("./test_files")
    all_files = list(sdk.load_codebase_files().keys())
    
    while True:
        try:
            # Get user input
            user_input = input(f"{TestColors.CYAN}You: {TestColors.ENDC}").strip()
            
            if user_input.lower() in ["quit", "exit"]:
                print(f"{TestColors.WARNING}Ending interactive session.{TestColors.ENDC}")
                break
            
            if not user_input:
                continue
            
            # Create question
            question = Question(text=user_input)
            
            # Get response
            print(f"{TestColors.GREEN}Assistant: {TestColors.ENDC}", end="", flush=True)
            
            response_chunks = []
            async for chunk in await get_response(
                chat_session=session,
                file_path_list=None,  # Use all files
                question=question,
                chat_history=chat_history,
                deep_thinking=True,
                stream=True
            ):
                # Remove response tags for cleaner output
                clean_chunk = chunk.replace("<response>", "").replace("</response>", "")
                print(clean_chunk, end="", flush=True)
                response_chunks.append(clean_chunk)
            
            print("\n")  # New line after response
            
            # Update chat history
            response = "".join(response_chunks)
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            
            # Keep chat history limited to last 10 exchanges
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
            
        except KeyboardInterrupt:
            print(f"\n{TestColors.WARNING}Interrupted. Type 'quit' to exit.{TestColors.ENDC}")
            continue
        except Exception as e:
            print_error(f"Error in interactive mode: {e}")


async def main():
    """Main test runner."""
    # Check for API key
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print_error("ANTHROPIC_API_KEY not found in environment variables!")
        print("Please create a .env file with your API key.")
        return
    
    print_header("Claude Code SDK Test Suite")
    print(f"Testing with codebase at: {Path('./test_files').absolute()}\n")
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Simple Query", test_simple_query),
        ("Streaming Response", test_streaming_response),
        ("Chat History", test_chat_history),
        ("Multi-file Analysis", test_multi_file_analysis),
        ("Deep Thinking Mode", test_deep_thinking_mode),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{TestColors.GREEN}PASSED{TestColors.ENDC}" if result else f"{TestColors.FAIL}FAILED{TestColors.ENDC}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{TestColors.BOLD}Total: {passed}/{total} tests passed{TestColors.ENDC}")
    
    if passed == total:
        print_success("All tests passed! 🎉")
    else:
        print_error(f"{total - passed} test(s) failed.")
    
    # Ask if user wants to run interactive mode
    print(f"\n{TestColors.CYAN}Would you like to run the interactive mode? (y/n): {TestColors.ENDC}", end="")
    choice = input().strip().lower()
    if choice == "y":
        await run_interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
