# Anthropic File API Test Implementation

This module provides a comprehensive implementation for processing files using the Anthropic API with support for chat history and question answering.

## Features

- **File Processing**: Upload and process PDF files (and other document types) using Anthropic's Claude models
- **Chat History Support**: Maintain conversation context across multiple interactions
- **Error Handling**: Robust error handling with detailed error messages
- **Type Safety**: Full TypeScript-style type annotations for Python
- **Comprehensive Testing**: Complete test suite with pytest
- **Token Usage Tracking**: Monitor API usage and costs

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
   - Copy the example environment file: `cp .env.example .env`
   - Edit `.env` and replace `sk-ant-api03-your-api-key-here` with your actual Anthropic API key
   
   Alternatively, you can set the environment variable directly:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

**Important**: The `.env` file is automatically ignored by git to keep your API key secure.

## Core Function

The main function `process_file_with_question` takes three parameters:

- **file_path** (str): Path to the file you want to process
- **question** (str): The question you want to ask about the file
- **chat_history** (Optional[List[Dict[str, str]]]): Previous conversation history

### Function Signature
```python
def process_file_with_question(
    self, 
    file_path: str, 
    question: str, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
```

### Return Format
```python
{
    "response": "Claude's response to your question",
    "model_used": "claude-3-5-sonnet-20241022",
    "tokens_used": {
        "input": 1000,
        "output": 200
    },
    "file_processed": "filename.pdf",
    "success": True
}
```

## Usage Examples

### Basic Usage
```python
from file_api_test import AnthropicFileHandler

# Initialize handler
handler = AnthropicFileHandler()

# Process a file with a simple question
result = handler.process_file_with_question(
    "path/to/your/document.pdf",
    "What is the main topic of this document?"
)

if result["success"]:
    print(f"Response: {result['response']}")
    print(f"Tokens used: {result['tokens_used']}")
else:
    print(f"Error: {result['error']}")
```

### With Chat History
```python
# Define conversation history
chat_history = [
    {"role": "user", "content": "I'm studying machine learning."},
    {"role": "assistant", "content": "Great! I'd love to help you learn ML concepts."}
]

# Ask a question in context
result = handler.process_file_with_question(
    "research_paper.pdf",
    "Can you explain the ML concepts in this paper in simple terms?",
    chat_history
)
```

### Building a Conversation
```python
# Start with empty history
conversation = []

# First interaction
result1 = handler.process_file_with_question(
    "document.pdf",
    "What are the main findings?",
    conversation
)

# Add to conversation history
if result1["success"]:
    conversation.extend([
        {"role": "user", "content": "What are the main findings?"},
        {"role": "assistant", "content": result1["response"]}
    ])

# Continue conversation
result2 = handler.process_file_with_question(
    "document.pdf",
    "Can you elaborate on the first finding?",
    conversation
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest file_api_test.py -v

# Run specific test categories
pytest file_api_test.py::TestAnthropicFileHandler::test_handler_initialization_success -v

# Run tests that don't require API calls
pytest file_api_test.py -v -k "not test_process_file_with_question_basic and not test_process_file_with_chat_history"
```

## Demo

Run the included demo to see the implementation in action:

```bash
python file_api_test.py
```

This will demonstrate:
1. Simple question answering
2. Question answering with chat history
3. Follow-up questions in context

## Supported File Types

The implementation automatically detects file MIME types and supports various document formats that Anthropic's API can process:

- PDF files (`.pdf`)
- Text files (`.txt`)
- And other document formats supported by Claude

## Error Handling

The implementation includes comprehensive error handling for:

- **File Not Found**: When specified files don't exist
- **Empty Questions**: When no question is provided
- **Invalid Chat History**: When chat history format is incorrect
- **API Errors**: Anthropic API-specific errors
- **Network Issues**: Connection and timeout errors

## API Key Security

**Important**: Never commit your API key to version control. This implementation uses secure environment variable loading:

1. **Recommended**: Use the `.env` file (automatically loaded and git-ignored)
2. **Alternative**: Set the `ANTHROPIC_API_KEY` environment variable directly
3. **For testing only**: Pass the API key directly to the constructor

The `.env` file is automatically excluded from git commits via `.gitignore` to protect your API key.

## Token Usage and Costs

The implementation tracks token usage for both input and output, helping you monitor API costs. Large PDF files can consume significant tokens, so consider:

- File size limits
- Token budget management
- Using smaller files for testing

## Class Structure

### AnthropicFileHandler
- Main handler class with all functionality
- Handles API client initialization
- Manages file processing and chat history

### TestAnthropicFileHandler
- Comprehensive test suite
- Tests all functionality including edge cases
- Includes both unit tests and integration tests

## Technical Requirements

- Python 3.7+
- anthropic>=0.25.0
- pytest>=7.0.0 (for testing)
- python-dotenv>=1.0.0 (for environment management)

## Architecture Notes

The implementation follows these design principles:

1. **Type Safety**: Complete type annotations for all functions
2. **Error Handling**: Graceful error handling with detailed messages
3. **Modularity**: Separate methods for different concerns
4. **Testability**: Comprehensive test coverage
5. **Documentation**: Extensive docstrings and comments

## Performance Considerations

- Large files may take longer to process and consume more tokens
- Chat history length affects processing time and token usage
- Consider implementing caching for repeated file processing
- Monitor token usage to manage costs effectively 