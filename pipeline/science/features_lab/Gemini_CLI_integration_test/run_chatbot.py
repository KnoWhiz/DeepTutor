#!/usr/bin/env python3
"""
Launcher script for the Gemini CLI Chatbot.

This script sets up the environment and launches the Streamlit chatbot application.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def check_environment():
    """Check if the required environment is set up correctly."""
    print("🔍 Checking environment setup...")
    
    # Check if Gemini CLI is installed
    try:
        result = subprocess.run(["gemini", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Gemini CLI is installed")
        else:
            print("❌ Gemini CLI not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Gemini CLI not found. Please install it with:")
        print("   npm install -g @google/gemini-cli@latest")
        return False
    
    # Check for GEMINI_API_KEY
    env_file = current_dir / ".env"
    if env_file.exists():
        print("✅ .env file found")
        
        # Try to load the API key
        with open(env_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    if key == "GEMINI_API_KEY":
                        if value.strip('"\'') != "your_gemini_api_key_here":
                            print("✅ GEMINI_API_KEY is configured")
                            return True
                        else:
                            print("❌ GEMINI_API_KEY is not set in .env file")
                            return False
        
        print("❌ GEMINI_API_KEY not found in .env file")
        return False
    else:
        print("❌ .env file not found")
        print("Please create a .env file with your GEMINI_API_KEY:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        return False


def setup_working_directory():
    """Create the working directory for file processing."""
    working_dir = Path("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files")
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Working directory ready: {working_dir}")


def main():
    """Main launcher function."""
    print("🚀 Starting Gemini CLI Chatbot...")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Setup working directory
    setup_working_directory()
    
    print("\n✅ All checks passed!")
    print("🌟 Launching Streamlit chatbot...")
    print("=" * 50)
    
    # Launch Streamlit app
    try:
        streamlit_app = current_dir / "streamlit_chatbot.py"
        subprocess.run([
            "python", "-m", "streamlit", "run", str(streamlit_app),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 