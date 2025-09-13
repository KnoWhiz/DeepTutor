# Claude Code SDK Integration for DeepTutor

This implementation provides a Claude-based chatbot with code analysis capabilities that maintains compatibility with the existing DeepTutor `get_response` function interface.

## Features

- **Compatible Interface**: Maintains the same function signature as the original `get_response` function
- **Streaming Support**: Supports both streaming and non-streaming responses
- **Code Analysis**: Can analyze code files and provide detailed explanations
- **Chat History**: Maintains conversation context across multiple interactions
- **Deep Thinking Mode**: Provides more detailed analysis when enabled
- **Multi-file Support**: Can analyze multiple files simultaneously

## Installation

1. Ensure you have the `deeptutor` conda environment activated:
```bash
conda activate deeptutor
```

2. Install required dependencies (already installed in deeptutor env):
```bash
pip install anthropic python-dotenv
```

3. Ensure your `.env` file contains the `ANTHROPIC_API_KEY`:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Files

- **claude_code_sdk.py** - Main implementation with the core chatbot logic
- **test_claude_sdk.py** - Comprehensive test suite with interactive mode
- **test_files/** - Sample codebase folder for testing

## Usage

### Basic Usage

```python
import asyncio
from claude_code_sdk import ChatSession, Question, get_response

async def main():
    # Create a chat session
    session = ChatSession(
        session_id="unique_id",
        mode="ADVANCED",
        model="claude-3-5-sonnet-20241022"
    )
    
    # Create a question
    question = Question(
        text="Explain the Calculator class",
        special_context="Focus on error handling"
    )
    
    # Get streaming response
    async for chunk in await get_response(
        chat_session=session,
        file_path_list=["sample_code.py"],
        question=question,
        chat_history=[],
        embedding_folder_list=[],
        deep_thinking=True,
        stream=True
    ):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Simple Query Helper

```python
import asyncio
from claude_code_sdk import simple_query

async def main():
    async for chunk in simple_query("What does this code do?", ["sample_code.py"]):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Running Tests

Run the comprehensive test suite:

```bash
python test_claude_sdk.py
```

The test suite includes:
1. Basic functionality tests
2. Simple query tests
3. Streaming response tests
4. Chat history tests
5. Multi-file analysis tests
6. Deep thinking mode tests
7. Interactive chat mode

## Interactive Mode

After running the tests, you can enter interactive mode to chat with the bot about the codebase:

```bash
python test_claude_sdk.py
# Choose 'y' when prompted for interactive mode
```

## Configuration

The SDK uses the following default configuration:
- **Codebase Path**: `/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/test_files`
- **Model**: `claude-3-5-sonnet-20241022`
- **Max Tokens**: 4096
- **Temperature**: 0.7

These can be modified in the `claude_code_sdk.py` file.

## Response Format

Responses follow these guidelines:
1. Start with a TL;DR summary
2. Use **bold** for key concepts
3. Use `inline code` for code references
4. Include code blocks with syntax highlighting
5. Support LaTeX math with `$...$` or `$$...$$`
6. Break down complex topics into logical segments

## Limitations

- Requires valid ANTHROPIC_API_KEY
- Subject to Claude API rate limits
- Currently configured for a specific codebase path (can be modified)

## Integration with DeepTutor

This implementation maintains full compatibility with the existing DeepTutor interface, allowing it to be used as a drop-in replacement for the original `get_response` function while leveraging Claude's capabilities for code analysis and understanding.
