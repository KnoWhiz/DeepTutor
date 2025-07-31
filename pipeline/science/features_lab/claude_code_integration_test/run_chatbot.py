#!/usr/bin/env python3
"""
Run Claude Code Integration Chatbot

This script launches the Streamlit-based chatbot that uses Claude's codebase understanding
capabilities to analyze and respond to queries about text and markdown files.

Usage:
    python run_chatbot.py
    
Make sure you have:
1. Created config.py from config_template.py with your Anthropic API key
2. Activated the deeptutor conda environment
3. Files in the test_files directory to analyze
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import anthropic
        print("✅ Required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_config():
    """Check if configuration is set up."""
    config_path = Path(__file__).parent / "config.py"
    if not config_path.exists():
        print("⚠️  No config.py found. Please:")
        print("1. Copy config_template.py to config.py")
        print("2. Add your Anthropic API key to config.py")
        return False
    
    try:
        from config import ANTHROPIC_API_KEY
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
            print("⚠️  Please set your Anthropic API key in config.py")
            return False
        print("✅ Configuration found")
        return True
    except ImportError:
        print("❌ Error importing config.py")
        return False

def check_test_files():
    """Check if test files exist."""
    test_dir = Path(__file__).parent / "test_files"
    if not test_dir.exists():
        print("⚠️  test_files directory not found")
        return False
    
    files = list(test_dir.glob("*.*"))
    if not files:
        print("⚠️  No files found in test_files directory")
        return False
    
    print(f"✅ Found {len(files)} files in test_files directory:")
    for file in files[:5]:  # Show first 5 files
        print(f"   • {file.name}")
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more files")
    
    return True

def main():
    """Main function to run the chatbot."""
    print("🤖 Claude Code Integration Chatbot Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check configuration
    if not check_config():
        sys.exit(1)
    
    # Check test files
    check_test_files()
    
    print("\n🚀 Launching Streamlit application...")
    print("The chatbot will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    print("-" * 50)
    
    # Get the path to the main application file
    app_path = Path(__file__).parent / "claude_code_integration_test.py"
    
    try:
        # Run the Streamlit application
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 