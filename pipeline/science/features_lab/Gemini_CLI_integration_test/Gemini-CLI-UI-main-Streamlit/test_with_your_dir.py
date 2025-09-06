#!/usr/bin/env python3
"""
Test script for Gemini CLI Agent using the user's specific directory
"""

import os
import sys
import time

# Add the current directory to Python path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_cli_agent import gemini_cli_agent, GeminiResponse


def test_with_user_directory():
    """Test the Gemini CLI agent with the user's specific directory"""
    print("🚀 Testing Gemini CLI Agent with your directory")
    print("=" * 60)
    
    # Use the user's specified directory
    target_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files"
    
    # First, let's ask about the files in the directory
    query = "What files are in this directory? Please provide a brief overview of each file and what they might contain based on their names and any content you can see."
    
    print(f"📁 Working directory: {target_dir}")
    print(f"❓ Query: {query}")
    print("-" * 60)
    
    start_time = time.time()
    accumulated_response = ""
    chunk_count = 0
    
    try:
        for response in gemini_cli_agent(
            target_dir, 
            query,
            model="gemini-2.5-pro",
            skip_permissions=True,
            timeout=45.0
        ):
            elapsed = time.time() - start_time
            
            if response.error:
                print(f"❌ ERROR [{elapsed:.1f}s]: {response.error}")
                break
            elif response.content:
                chunk_count += 1
                accumulated_response += response.content + "\n"
                partial_indicator = " (partial)" if response.is_partial else ""
                print(f"📝 CHUNK {chunk_count} [{elapsed:.1f}s]{partial_indicator}:")
                print(response.content)
                print()
            elif response.exit_code is not None:
                print(f"✅ COMPLETED [{elapsed:.1f}s] (exit code: {response.exit_code})")
                print(f"📊 Total chunks received: {chunk_count}")
                if response.session_id:
                    print(f"📋 Session ID: {response.session_id}")
                break
                
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
    
    print("\n" + "=" * 60)
    print("📄 FULL ACCUMULATED RESPONSE:")
    print("=" * 60)
    print(accumulated_response)


def test_research_analysis():
    """Test analyzing the research papers in the directory"""
    print("\n🔬 Testing Research Paper Analysis")
    print("=" * 60)
    
    target_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files"
    
    query = """I see there are several research papers and text files here. Could you:
1. Identify what research topics these papers cover
2. Explain the main themes or subjects
3. Tell me if there are any connections between these papers
4. Suggest what kind of research project this might be for

Focus particularly on the PDF files and text files that seem to be research papers."""
    
    print(f"📁 Working directory: {target_dir}")
    print(f"❓ Query: {query}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        for response in gemini_cli_agent(
            target_dir, 
            query,
            model="gemini-2.5-pro",
            skip_permissions=True,
            timeout=60.0
        ):
            elapsed = time.time() - start_time
            
            if response.error:
                print(f"❌ ERROR [{elapsed:.1f}s]: {response.error}")
                break
            elif response.content:
                print(response.content, end="", flush=True)
            elif response.exit_code is not None:
                print(f"\n\n✅ ANALYSIS COMPLETED [{elapsed:.1f}s] (exit code: {response.exit_code})")
                break
                
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")


def main():
    """Run the tests"""
    print("🧪 Gemini CLI Agent - Testing with Your Directory")
    print("=" * 60)
    
    # Check if Gemini CLI is available
    import subprocess
    try:
        subprocess.run(['gemini', '--version'], capture_output=True, check=True, timeout=5)
        print("✅ Gemini CLI found and accessible")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Gemini CLI not found. Please install it with:")
        print("   npm install -g @google/generative-ai-cli")
        return
    
    # Run tests
    try:
        test_with_user_directory()
        
        # Ask if user wants to run the research analysis
        print("\n" + "=" * 60)
        try_research = input("🤔 Would you like to run the research paper analysis? (y/N): ").strip().lower()
        if try_research in ['y', 'yes']:
            test_research_analysis()
            
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user.")


if __name__ == "__main__":
    main() 