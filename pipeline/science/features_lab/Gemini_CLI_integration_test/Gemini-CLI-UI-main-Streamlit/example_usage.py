#!/usr/bin/env python3
"""
Example Usage of Gemini CLI Agent

This script demonstrates how to use the gemini_cli_agent function to interact
with the Gemini CLI in streaming mode.
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_cli_agent import gemini_cli_agent, GeminiResponse


def simple_example():
    """Basic example of using the Gemini CLI agent"""
    print("=== Simple Example ===")
    
    # Use current directory as the project folder
    current_dir = os.getcwd()
    query = "What files are in this directory? Provide a brief overview of the project structure."
    
    print(f"Working directory: {current_dir}")
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        for response in gemini_cli_agent(current_dir, query, timeout=30.0):
            if response.error:
                print(f"❌ ERROR: {response.error}")
            elif response.content:
                print(f"📝 {response.content}")
            elif response.exit_code is not None:
                print(f"✅ COMPLETED (exit code: {response.exit_code})")
                if response.session_id:
                    print(f"📋 Session ID: {response.session_id}")
                
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")


def streaming_example():
    """Example showing streaming responses with timing"""
    print("\n=== Streaming Example with Timing ===")
    
    current_dir = os.getcwd()
    query = "Analyze this Python project and explain what it does. Include details about the main components and their purpose."
    
    print(f"Working directory: {current_dir}")
    print(f"Query: {query}")
    print("-" * 50)
    
    start_time = time.time()
    chunk_count = 0
    
    try:
        for response in gemini_cli_agent(
            current_dir, 
            query, 
            model="gemini-2.5-pro",
            skip_permissions=True,  # Skip permission prompts for smoother experience
            timeout=45.0
        ):
            elapsed = time.time() - start_time
            
            if response.error:
                print(f"❌ ERROR [{elapsed:.1f}s]: {response.error}")
            elif response.content:
                chunk_count += 1
                partial_indicator = " (partial)" if response.is_partial else ""
                print(f"📝 CHUNK {chunk_count} [{elapsed:.1f}s]{partial_indicator}:")
                print(f"   {response.content}")
                print()
            elif response.exit_code is not None:
                print(f"✅ COMPLETED [{elapsed:.1f}s] (exit code: {response.exit_code})")
                print(f"📊 Total chunks received: {chunk_count}")
                if response.session_id:
                    print(f"📋 Session ID: {response.session_id}")
                
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")


def code_analysis_example():
    """Example of using Gemini to analyze code in a specific directory"""
    print("\n=== Code Analysis Example ===")
    
    # Look for Python files in current directory
    current_dir = os.getcwd()
    python_files = list(Path(current_dir).glob("*.py"))
    
    if not python_files:
        print("No Python files found in current directory. Skipping this example.")
        return
    
    query = f"""
    Please analyze the Python code in this directory. I'm particularly interested in:
    1. What is the main purpose of this code?
    2. What are the key functions and classes?
    3. Are there any potential improvements or issues you notice?
    4. How would you rate the code quality and documentation?
    
    Focus on these files: {[f.name for f in python_files[:3]]}  # Limit to first 3 files
    """
    
    print(f"Working directory: {current_dir}")
    print(f"Python files found: {[f.name for f in python_files]}")
    print(f"Query: {query.strip()}")
    print("-" * 50)
    
    try:
        accumulated_response = ""
        
        for response in gemini_cli_agent(
            current_dir, 
            query.strip(),
            model="gemini-2.5-pro",
            skip_permissions=True,
            debug=False,
            timeout=60.0
        ):
            if response.error:
                print(f"❌ ERROR: {response.error}")
            elif response.content:
                accumulated_response += response.content + "\n"
                # Show a progress indicator
                print(".", end="", flush=True)
            elif response.exit_code is not None:
                print(f"\n✅ ANALYSIS COMPLETE (exit code: {response.exit_code})")
                print("=" * 60)
                print(accumulated_response)
                print("=" * 60)
                
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")


def interactive_example():
    """Interactive example where user can input queries"""
    print("\n=== Interactive Example ===")
    print("Type your queries for Gemini CLI. Type 'quit' to exit.")
    
    current_dir = os.getcwd()
    session_id = None  # Will be set after first response
    
    while True:
        try:
            query = input(f"\n📁 [{os.path.basename(current_dir)}] Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("-" * 50)
            
            for response in gemini_cli_agent(
                current_dir, 
                query,
                session_id=session_id,  # Continue conversation
                model="gemini-2.5-pro",
                skip_permissions=True,
                timeout=30.0
            ):
                if response.error:
                    print(f"❌ ERROR: {response.error}")
                elif response.content:
                    print(response.content)
                elif response.exit_code is not None:
                    if response.session_id and not session_id:
                        session_id = response.session_id
                        print(f"📋 Session started: {session_id}")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"💥 EXCEPTION: {e}")


def main():
    """Run all examples"""
    print("🚀 Gemini CLI Agent Examples")
    print("=" * 50)
    
    # Check if Gemini CLI is available
    import subprocess
    try:
        subprocess.run(['gemini', '--version'], capture_output=True, check=True, timeout=5)
        print("✅ Gemini CLI found and accessible")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Gemini CLI not found. Please install it with:")
        print("   npm install -g @google/generative-ai-cli")
        print("\nOr set the GEMINI_PATH environment variable to point to the executable.")
        return
    
    # Run examples
    try:
        simple_example()
        streaming_example()
        code_analysis_example()
        
        # Ask user if they want to try interactive mode
        try_interactive = input("\n🤔 Would you like to try interactive mode? (y/N): ").strip().lower()
        if try_interactive in ['y', 'yes']:
            interactive_example()
            
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user.")


if __name__ == "__main__":
    main() 