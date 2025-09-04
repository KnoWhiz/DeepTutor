#!/usr/bin/env python3
"""
Simple launcher script for the Gemini CLI Chatbot.

This script launches the Streamlit chatbot with better error handling and feedback.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit chatbot."""
    print("🚀 Launching Gemini CLI Chatbot...")
    print("=" * 50)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    streamlit_app = current_dir / "streamlit_chatbot.py"
    
    # Check if the app file exists
    if not streamlit_app.exists():
        print(f"❌ Error: {streamlit_app} not found!")
        sys.exit(1)
    
    print(f"📁 App location: {streamlit_app}")
    print("🌐 Starting server at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Change to the app directory
        os.chdir(current_dir)
        
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_chatbot.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print("=" * 50)
        
        # Run the command in the foreground
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ Streamlit exited with code: {result.returncode}")
        else:
            print(f"\n✅ Streamlit stopped normally")
            
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user (Ctrl+C)")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you're in the correct conda environment:")
        print("   conda activate deeptutor")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 