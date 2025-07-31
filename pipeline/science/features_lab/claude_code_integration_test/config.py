"""
Configuration Template for Claude Code Integration Test

Copy this file to 'config.py' and fill in your actual API keys and settings.
DO NOT commit config.py to version control - add it to .gitignore
"""

# Anthropic API Key for Claude access
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# Optional: Default directory for codebase files
DEFAULT_CODEBASE_DIR = "./test_files"

# Claude model configuration
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 4000
TEMPERATURE = 0.3

# File processing settings
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"}
MAX_FILE_SIZE_MB = 10
MAX_CONTEXT_CHARS = 50000 