# Claude Code SDK Chatbot Implementation

This implementation provides a Claude Code SDK-based chatbot that follows the same interface as the original `get_response` function but uses Claude's code analysis capabilities.

## Features

- **Same Interface**: Uses the exact same function signature as `get_response`
- **Code Analysis**: Leverages Claude Code SDK for enhanced code understanding
- **Streaming Response**: Returns a generator for real-time streaming
- **File Context**: Analyzes uploaded files and codebase directories
- **Chat History**: Maintains conversation context
- **Error Handling**: Robust error handling with graceful fallbacks

## Dependencies

Install the required dependencies:

```bash
pip install anthropic pymongo
```

## Environment Setup

Make sure you have your Anthropic API key set in your `.env` file:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from claude_code_sdk import get_claude_code_response, ChatSession, Question, ChatMode

# Create a session
session = ChatSession()
session.initialize()

# Create a question
question = Question(
    text="Can you analyze this code and suggest improvements?",
    language="English",
    question_type="global"
)

# Define files to analyze
file_paths = ["path/to/your/file.py"]

# Define codebase directory
codebase_dir = "path/to/your/codebase"

# Get streaming response
response_generator = get_claude_code_response(
    chat_session=session,
    file_path_list=file_paths,
    question=question,
    chat_history=[],
    codebase_folder_dir=codebase_dir,
    deep_thinking=True,
    stream=True
)

# Process the streaming response
for chunk in response_generator:
    print(chunk, end="", flush=True)
```

### Function Signature

```python
def get_claude_code_response(
    chat_session: ChatSession, 
    file_path_list: List[str], 
    question: Question, 
    chat_history: List[Dict], 
    codebase_folder_dir: str,
    deep_thinking: bool = True, 
    stream: bool = True
) -> Generator[str, None, None]:
```

**Parameters:**
- `chat_session`: ChatSession object containing session information
- `file_path_list`: List of file paths to analyze
- `question`: Question object containing the user's question
- `chat_history`: List of previous chat messages
- `codebase_folder_dir`: Path to the codebase folder for context
- `deep_thinking`: Whether to use deep thinking mode (not used in this implementation)
- `stream`: Whether to return a streaming generator (always True)

**Returns:**
- Generator[str, None, None]: Streaming response chunks

## Testing

### Method 1: Run the test script

```bash
cd /Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/
python test_claude_sdk.py
```

### Method 2: Run the built-in test

```bash
cd /Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/
python claude_code_sdk.py
```

### Method 3: Interactive testing

```python
# In a Python REPL or Jupyter notebook
from claude_code_sdk import get_claude_code_response, ChatSession, Question, ChatMode

# Set up your test
session = ChatSession()
session.initialize()

question = Question(
    text="What does this code do?",
    language="English",
    question_type="global"
)

file_paths = ["/path/to/your/test/file.py"]
codebase_dir = "/path/to/your/codebase"

# Get response
response = get_claude_code_response(
    chat_session=session,
    file_path_list=file_paths,
    question=question,
    chat_history=[],
    codebase_folder_dir=codebase_dir
)

# Print response
for chunk in response:
    print(chunk, end="")
```

## Customization

### Modifying the System Prompt

Edit the `system_prompt` variable in the `get_claude_code_response` function to customize the AI's behavior:

```python
system_prompt = """Your custom system prompt here..."""
```

### Adjusting File Processing

Modify the file processing logic to:
- Change file extensions to analyze (currently `.py` files)
- Adjust content length limits
- Add file filtering logic

### Error Handling

The implementation includes comprehensive error handling:
- Missing dependencies
- File read errors
- API errors
- Network issues

## Integration with Existing Code

To integrate with your existing codebase, simply replace calls to `get_response` with `get_claude_code_response`:

```python
# Before
response = await get_response(
    chat_session=session,
    file_path_list=files,
    question=question,
    chat_history=history,
    embedding_folder_list=embeddings,
    deep_thinking=True,
    stream=True
)

# After
response = get_claude_code_response(
    chat_session=session,
    file_path_list=files,
    question=question,
    chat_history=history,
    codebase_folder_dir="/path/to/codebase",
    deep_thinking=True,
    stream=True
)
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure to install dependencies with `pip install anthropic pymongo`
2. **API Key Error**: Verify your `ANTHROPIC_API_KEY` is set correctly
3. **File Not Found**: Check that file paths and codebase directory exist
4. **Permission Errors**: Ensure read permissions for files and directories

### Debug Mode

Enable debug logging by modifying the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- The implementation limits context to the first 10 files to avoid token limits
- File content is truncated to 2000 characters per file
- Consider the total context size when processing large codebases
- Monitor API usage and costs through the session's accumulated cost tracking
