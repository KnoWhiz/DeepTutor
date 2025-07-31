# Claude Code Integration Chatbot

A sophisticated Streamlit-based chatbot that leverages Claude's advanced understanding capabilities to analyze and interact with your document collections. This application treats text and markdown files as a unified "codebase" for comprehensive analysis and intelligent responses.

## Features

🤖 **AI-Powered Analysis**: Uses Claude-3.5-Sonnet for deep document understanding
📁 **Multi-File Support**: Analyzes entire directories of text/markdown files
💬 **Interactive Chat**: Natural conversation interface for querying your documents
🔍 **Contextual Understanding**: Maintains conversation context and file relationships
📊 **Comprehensive Responses**: Provides detailed analysis with citations and connections
🎯 **Sample Queries**: Pre-built questions to demonstrate capabilities

## Supported File Types

- `.txt` - Text files
- `.md` - Markdown files
- `.py` - Python files (analyzed as documentation)
- `.js` - JavaScript files (analyzed as documentation)
- `.json` - JSON configuration files
- `.yaml`, `.yml` - YAML configuration files

## Prerequisites

1. **Python Environment**: Python 3.8+ (recommend using conda)
2. **API Access**: Anthropic API key with Claude access
3. **Dependencies**: See `requirements.txt`

## Quick Start

### 1. Environment Setup

```bash
# Activate your conda environment
conda activate deeptutor

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the configuration template
cp config_template.py config.py

# Edit config.py and add your Anthropic API key
# ANTHROPIC_API_KEY = "your_actual_api_key_here"
```

### 3. Add Your Documents

Place your text/markdown files in the `test_files` directory:

```bash
# Files are already provided for testing:
# - paper1.txt (ReAct: Synergizing Reasoning and Acting in Language Models)
# - paper2.txt (Hierarchical Taxonomy of Psychopathology)

# Add your own files:
cp /path/to/your/documents/* test_files/
```

### 4. Run the Application

```bash
# Option 1: Use the launcher script
python run_chatbot.py

# Option 2: Direct streamlit command
streamlit run claude_code_integration_test.py
```

The application will open in your web browser at `http://localhost:8501`

## Usage Instructions

### Loading Your Codebase

1. **Enter API Key**: Input your Anthropic API key in the sidebar
2. **Set Directory**: Specify the path to your documents (defaults to `test_files`)
3. **Load Files**: Click "🔄 Load/Reload Codebase" to process your documents
4. **Verify Loading**: Check the "📁 Codebase Overview" section for loaded files

### Interacting with the Chatbot

Once your codebase is loaded, you can:

- **Ask Questions**: Type any question about your documents in the chat input
- **Use Sample Queries**: Click on pre-built questions in the sidebar
- **Follow-up**: Ask follow-up questions to dive deeper into topics
- **Cross-Reference**: Ask about relationships between different documents

### Sample Queries

- "What are the main topics covered in these documents?"
- "Can you summarize the key methodologies described?"
- "What are the relationships between different concepts?"
- "Explain the ReAct framework mentioned in the papers"
- "What are the main findings and conclusions?"
- "How do these documents relate to each other?"

## Architecture

### Core Components

1. **CodebaseAnalyzer**: Handles file loading and Claude API integration
2. **StreamlitChatInterface**: Manages the web UI and user interactions
3. **Context Management**: Handles token limits and conversation history

### Key Features

- **Token Management**: Automatically handles Claude's context limits
- **File Processing**: Supports multiple file types with error handling
- **Conversation History**: Maintains context across multiple exchanges
- **Error Handling**: Graceful error handling with user-friendly messages

## Configuration Options

Edit `config.py` to customize:

```python
# Model settings
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 4000
TEMPERATURE = 0.3

# File processing
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"}
MAX_CONTEXT_CHARS = 50000
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Anthropic API key is valid and has sufficient credits
2. **File Loading Issues**: Check file permissions and encoding (UTF-8 recommended)
3. **Memory Issues**: Large files may be truncated to fit context limits
4. **Connection Issues**: Verify internet connection for API access

### Debug Steps

1. Check the console output for detailed error messages
2. Verify all dependencies are installed: `pip list | grep -E "(streamlit|anthropic)"`
3. Test API key separately: `python -c "import anthropic; print('OK')"`
4. Check file permissions in the test_files directory

## Limitations

- **File Size**: Large files are automatically truncated to fit context limits
- **Token Limits**: Very long conversations may need to be reset
- **File Types**: Only processes text-based files (no binary files)
- **API Costs**: Each query consumes API tokens (monitor usage)

## Development

### Project Structure

```
claude_code_integration_test/
├── claude_code_integration_test.py  # Main application
├── run_chatbot.py                   # Launcher script
├── config_template.py               # Configuration template
├── config.py                        # Your configuration (create from template)
├── README.md                        # This file
└── test_files/                      # Document directory
    ├── paper1.txt
    └── paper2.txt
```

### Extending the Application

- **Add File Types**: Modify `SUPPORTED_EXTENSIONS` in config
- **Custom Prompts**: Edit the system message in `CodebaseAnalyzer.get_contextual_response()`
- **UI Enhancements**: Modify the Streamlit interface in `StreamlitChatInterface`
- **New Features**: Add methods to `CodebaseAnalyzer` class

## License

This project follows the same license as the parent DeepTutor project.

## Support

For issues and questions:
1. Check this README for common solutions
2. Review the console output for error details
3. Verify your API key and network connection
4. Check file permissions and formats 