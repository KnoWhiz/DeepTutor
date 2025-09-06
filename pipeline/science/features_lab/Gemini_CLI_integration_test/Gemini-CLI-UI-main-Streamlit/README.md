# Gemini CLI Agent - Python Implementation

A Python interface to the Gemini CLI that provides streaming responses and intelligent buffering. This implementation is based on the [Gemini-CLI-UI-main](../Gemini-CLI-UI-main) project and provides a clean Python API for interacting with Google's Gemini CLI tool.

## Features

- **Streaming Responses**: Real-time streaming of Gemini CLI output
- **Intelligent Buffering**: Smart message chunking similar to the original JavaScript implementation
- **Session Management**: Support for continuing conversations with session IDs
- **Image Support**: Ability to include images in queries
- **Error Handling**: Comprehensive error handling and timeout management
- **Type Safety**: Full TypeScript-style type hints for better IDE support
- **Clean API**: Simple, Pythonic interface

## Prerequisites

1. **Gemini CLI**: Install the official Gemini CLI tool
   ```bash
   npm install -g @google/generative-ai-cli
   ```

2. **Python**: Python 3.7+ (uses only standard library modules)

3. **API Key**: Set up your Google AI API key for Gemini CLI

## Installation

No external dependencies required! The implementation uses only Python standard library modules.

```bash
# Clone or download the files
# No pip install needed - uses standard library only
```

## Quick Start

```python
from gemini_cli_agent import gemini_cli_agent, GeminiResponse

# Basic usage
folder_dir = "/path/to/your/project"
query = "What files are in this directory?"

for response in gemini_cli_agent(folder_dir, query):
    if response.error:
        print(f"Error: {response.error}")
    elif response.content:
        print(response.content)
    elif response.exit_code is not None:
        print(f"Completed with exit code: {response.exit_code}")
```

## API Reference

### `gemini_cli_agent()`

Main function to interact with Gemini CLI.

```python
def gemini_cli_agent(
    input_folder_dir: str,
    query: str,
    session_id: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    gemini_path: Optional[str] = None,
    skip_permissions: bool = False,
    debug: bool = False,
    images: Optional[List[str]] = None,
    timeout: float = 30.0
) -> Generator[GeminiResponse, None, None]:
```

#### Parameters

- **`input_folder_dir`** (str): Directory path where Gemini CLI should operate
- **`query`** (str): Command/query to send to Gemini CLI
- **`session_id`** (Optional[str]): Session ID for continuing conversations
- **`model`** (str): Gemini model to use (default: "gemini-2.5-flash")
- **`gemini_path`** (Optional[str]): Path to gemini CLI executable (uses PATH if None)
- **`skip_permissions`** (bool): Skip permission prompts (--yolo flag)
- **`debug`** (bool): Enable debug output
- **`images`** (Optional[List[str]]): List of image file paths to include
- **`timeout`** (float): Timeout in seconds for CLI response

#### Returns

Generator yielding `GeminiResponse` objects with:
- **`content`** (str): Response content from Gemini
- **`is_partial`** (bool): Whether this is a partial response
- **`error`** (Optional[str]): Error message if any
- **`exit_code`** (Optional[int]): Process exit code when completed
- **`session_id`** (Optional[str]): Session ID for the conversation

## Examples

### Basic Query

```python
from gemini_cli_agent import gemini_cli_agent

# Simple question about project structure
for response in gemini_cli_agent("/path/to/project", "What does this code do?"):
    if response.content:
        print(response.content)
```

### Streaming with Progress

```python
import time

start_time = time.time()
accumulated_response = ""

for response in gemini_cli_agent(
    "/path/to/project", 
    "Analyze this codebase and suggest improvements",
    model="gemini-2.5-flash",
    skip_permissions=True,
    timeout=60.0
):
    elapsed = time.time() - start_time
    
    if response.error:
        print(f"❌ ERROR [{elapsed:.1f}s]: {response.error}")
    elif response.content:
        accumulated_response += response.content
        print(".", end="", flush=True)  # Progress indicator
    elif response.exit_code is not None:
        print(f"\n✅ COMPLETED [{elapsed:.1f}s]")
        print(accumulated_response)
```

### Session Continuation

```python
# First query
session_id = None
for response in gemini_cli_agent("/path/to/project", "What is this project about?"):
    if response.session_id:
        session_id = response.session_id
    if response.content:
        print(response.content)

# Follow-up query in same session
for response in gemini_cli_agent(
    "/path/to/project", 
    "Can you explain the main function in more detail?",
    session_id=session_id
):
    if response.content:
        print(response.content)
```

### With Images

```python
# Include images in the query
image_paths = ["/path/to/screenshot.png", "/path/to/diagram.jpg"]

for response in gemini_cli_agent(
    "/path/to/project",
    "Analyze these UI screenshots and suggest improvements",
    images=image_paths
):
    if response.content:
        print(response.content)
```

## Advanced Features

### Custom Response Buffering

The implementation includes intelligent response buffering that:
- Waits for complete sentences before yielding responses
- Never splits code blocks in the middle
- Handles proper formatting and spacing
- Provides configurable timing parameters

### Error Handling

Comprehensive error handling for:
- Missing Gemini CLI installation
- Invalid directory paths
- Process timeouts
- Network issues
- Permission errors

### Thread-Safe Operation

Uses threading for non-blocking I/O operations while maintaining thread safety for the main API.

## Running Examples

```bash
# Run the example script
python example_usage.py

# Or run individual examples
python -c "from example_usage import simple_example; simple_example()"
```

## Environment Variables

- **`GEMINI_PATH`**: Custom path to Gemini CLI executable
- **`GOOGLE_AI_API_KEY`**: Your Google AI API key (required by Gemini CLI)

## Troubleshooting

### "Gemini CLI not found"
```bash
# Install Gemini CLI
npm install -g @google/generative-ai-cli

# Or set custom path
export GEMINI_PATH="/path/to/gemini"
```

### "No response received"
- Check your internet connection
- Verify your API key is set correctly
- Increase the timeout parameter
- Try with `debug=True` to see detailed output

### Permission Issues
- Use `skip_permissions=True` for automated scripts
- Ensure the working directory is accessible
- Check file permissions in the target directory

## Implementation Details

This Python implementation closely follows the original JavaScript version:

1. **Process Management**: Uses `subprocess.Popen` with threading for non-blocking I/O
2. **Response Buffering**: Implements the same intelligent buffering logic as the JavaScript version
3. **Message Filtering**: Filters out debug messages and system notifications
4. **Session Handling**: Supports session continuation for multi-turn conversations
5. **Image Processing**: Handles image inclusion with temporary file management

## Comparison with Original

| Feature | JavaScript Version | Python Version |
|---------|-------------------|----------------|
| Streaming | ✅ WebSocket | ✅ Generator |
| Buffering | ✅ Intelligent | ✅ Intelligent |
| Sessions | ✅ Session Manager | ✅ Session ID |
| Images | ✅ Base64 Upload | ✅ File Paths |
| Error Handling | ✅ Comprehensive | ✅ Comprehensive |
| Dependencies | ❌ Many Node.js deps | ✅ Standard Library Only |

## License

MIT License - Same as the original Gemini-CLI-UI-main project.

## Contributing

Feel free to submit issues and enhancement requests! 