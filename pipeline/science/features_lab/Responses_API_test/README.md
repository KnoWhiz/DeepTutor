# OpenAI Responses API Chatbot (with Chat Completions Fallback)

This module implements a comprehensive chatbot originally designed for OpenAI's Responses API, but currently using Chat Completions API as a fallback since the Responses API is not yet available in the standard OpenAI Python library.

## Features

- **Context-aware response generation**: The chatbot can use provided context to generate more relevant responses
- **Multi-round web search capabilities**: Automatically performs web searches when needed to gather current information (via function calling)
- **Streaming response generator**: Provides real-time streaming responses for better user experience
- **Conversation history management**: Maintains conversation continuity across multiple interactions
- **Error handling and type validation**: Robust error handling with proper TypeScript-style type hints
- **Automatic model fallback**: Falls back from gpt-5-thinking to gpt-4o if the newer model is not available

## Setup

1. **Activate the conda environment**:
   ```bash
   conda activate deeptutor
   ```

2. **Ensure dependencies are installed**:
   The required packages should already be in your `requirements.txt`:
   - `openai>=1.65.1`
   - `python-dotenv>=1.0.1`

3. **Set up environment variables**:
   Make sure your `.env` file contains:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Basic Usage

```python
from responsee_api_test import create_chatbot_instance

# Create a chatbot instance (uses o3-mini by default)
chatbot = create_chatbot_instance(model="o3-mini")

# Simple query with context
context = "You are a helpful AI tutor specializing in quantum physics."
query = "Explain quantum entanglement in simple terms"

# Stream the response
for chunk in chatbot.chatbot_with_web_search(query=query, context=context, stream=True):
    if chunk["type"] == "chunk" and "content" in chunk:
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "web_search":
        print(f"\\n{chunk['content']}")
    elif chunk["type"] == "finish":
        print(f"\\n\\nResponse completed!")
```

### Advanced Usage with Web Search

```python
# Query that may require web search
query = "What are the latest developments in quantum computing in 2024?"

for chunk in chatbot.chatbot_with_web_search(
    query=query, 
    enable_web_search=True,
    stream=True
):
    if chunk["type"] == "chunk" and "content" in chunk:
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "web_search":
        print(f"\\n🔍 {chunk['content']}")
    elif chunk["type"] == "finish":
        print(f"\\n\\nResponse completed with {len(chunk.get('search_calls', []))} web searches")
```

### Continuing Conversations

```python
# Get response ID from previous interaction
previous_response_id = "your_previous_response_id"

# Continue the conversation
follow_up = "Can you provide more details about the commercial applications?"

for chunk in chatbot.continue_conversation(
    new_query=follow_up,
    previous_response_id=previous_response_id
):
    if chunk["type"] == "chunk" and "content" in chunk:
        print(chunk["content"], end="", flush=True)
```

## Running the Demo

To run the built-in demonstration:

```bash
conda activate deeptutor
cd pipeline/science/features_lab/Responses_API_test/
python responsee_api_test.py
```

This will run both functionality tests and demonstration examples.

## API Reference

### `ResponsesAPIChatbot` Class

#### Constructor
```python
ResponsesAPIChatbot(model: str = "o3-mini", max_search_rounds: int = 3)
```

**Note**: The chatbot uses `o3-mini` model by default. o3 models have fixed behavior for consistent reasoning (temperature adjustment not supported).

#### Main Methods

- **`chatbot_with_web_search()`**: Main chatbot function with web search capabilities
- **`continue_conversation()`**: Continue a conversation from a previous response
- **`get_response_by_id()`**: Retrieve a previous response by its ID

#### Response Types

The generator yields different types of chunks:

- **`"chunk"`**: Text content chunks during streaming
- **`"web_search"`**: Indicates a web search is being performed
- **`"finish"`**: Indicates the response is complete (streaming mode)
- **`"complete_response"`**: Full response data (non-streaming mode)
- **`"error"`**: Error information

## Key Features of OpenAI Responses API

1. **Stateful Conversations**: The API maintains conversation state automatically
2. **Hosted Tools**: Built-in web search without manual tool management
3. **Streaming Support**: Real-time response streaming
4. **Multi-turn Interactions**: Handle complex conversations in a single API call
5. **Tool Integration**: Seamless integration with web search and other tools

## Model Configuration: o3-mini (with gpt-4o Fallback)

This implementation uses `o3-mini` for optimal reasoning performance:

- **o3-mini**: OpenAI's latest reasoning model that provides enhanced analytical capabilities and step-by-step thinking
- **gpt-4o Fallback**: Automatically falls back to gpt-4o (with temperature=0) if o3-mini is not available
- **Fixed Behavior**: o3 models have fixed parameters and don't support temperature adjustment - they're designed for consistent reasoning
- **Benefits**:
  - Enhanced reasoning capabilities with o3-mini's advanced problem-solving
  - Built-in consistency due to fixed model behavior
  - Better analytical thinking through complex problems
  - Reliable fallback ensures the chatbot always works
  - Ideal for tutoring and educational contexts

**Available Models**: You can also use `o3`, `gpt-5`, `gpt-5-mini`, or `gpt-5-nano` depending on your access level and requirements.

**Note**: o3 models use different parameters (`max_completion_tokens` instead of `max_tokens`) and don't support temperature adjustment.

## Error Handling

The implementation includes comprehensive error handling:

- Import errors for missing dependencies
- API key validation
- Network and API errors
- Streaming errors
- Response parsing errors

## Performance Considerations

- Uses streaming by default for better user experience
- Implements proper error recovery
- Manages conversation history efficiently
- Handles rate limiting gracefully

## Troubleshooting

1. **Import Errors**: Make sure you're in the correct conda environment and dependencies are installed
2. **API Key Issues**: Verify your OpenAI API key is correctly set in the `.env` file
3. **Network Issues**: Check your internet connection for web search functionality
4. **Rate Limiting**: The API includes built-in rate limiting handling
5. **Model Availability**: The implementation automatically falls back to gpt-4o if o3-mini is not available
6. **Responses API**: This implementation uses Chat Completions API as a fallback since Responses API is not yet available in the standard OpenAI library

## Future Migration

When the OpenAI Responses API becomes available in the standard Python library, this implementation can be easily migrated by:
1. Updating the `_create_response` method to use `client.responses.create()`
2. Updating response handling methods to work with the Responses API format
3. The interface and functionality will remain the same

For more information, refer to the [OpenAI Chat Completions documentation](https://platform.openai.com/docs/api-reference/chat) and [OpenAI Responses API documentation](https://platform.openai.com/docs/api-reference/responses) (when available).
