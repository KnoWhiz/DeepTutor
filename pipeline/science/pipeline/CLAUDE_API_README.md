# Claude API Format Integration

**Your Goal**: Use Claude API format directly in your code (not CLI), with the proxy converting requests to Azure OpenAI format.

**Correct Architecture**:
```
Your Code (Claude API format) → Claude Proxy Server → Azure OpenAI Server
```

## 🎯 What You Want

You want to:
1. **Write code using Claude API format** (like the test examples show)
2. **Send requests to the proxy server** 
3. **Proxy converts to Azure OpenAI format** and sends to your Azure server
4. **Get Claude API features** while using your Azure infrastructure

## 🚀 Quick Start

### 1. Set up the proxy for Azure OpenAI

```bash
# Run the setup script
python claude_proxy_setup_correct.py
```

### 2. Start the proxy server

```bash
cd pipeline/science/features_lab/claude-code-proxy
python start_proxy.py
```

### 3. Use Claude API format in your code

```python
from claude_api_wrapper import ClaudeAPIWrapper, create_claude_message, create_claude_request

async def example():
    async with ClaudeAPIWrapper() as client:
        # Create Claude API format messages
        messages = [
            create_claude_message("user", "Hello, how are you?")
        ]
        
        # Create Claude API format request
        request = create_claude_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            max_tokens=100
        )
        
        # Send request (proxy converts to Azure OpenAI format)
        response = await client.chat_completion(request)
        print(f"Response: {response.text}")
```

## 📝 Usage Examples

### Basic Chat (Replaces API Handler)

```python
from claude_api_integration_example import ClaudeAPIService

async def basic_chat():
    claude_service = ClaudeAPIService()
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    response = await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-sonnet-20241022",  # Maps to your Azure deployment
        max_tokens=100
    )
    print(f"Response: {response}")
```

### Advanced Chat with System Message

```python
async def advanced_chat():
    claude_service = ClaudeAPIService()
    
    messages = [
        {"role": "user", "content": "Explain quantum computing"}
    ]
    
    response = await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-opus-20241022",  # Maps to your Azure deployment
        max_tokens=300,
        system="You are a helpful physics professor. Explain concepts clearly."
    )
    print(f"Response: {response}")
```

### Streaming

```python
async def streaming_chat():
    claude_service = ClaudeAPIService()
    
    messages = [
        {"role": "user", "content": "Tell me a story"}
    ]
    
    async for chunk in await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        stream=True
    ):
        if chunk.get("type") == "content_block_delta":
            content = chunk.get("delta", {}).get("text", "")
            if content:
                print(content, end="", flush=True)
```

### Function Calling

```python
async def function_calling():
    async with ClaudeAPIWrapper() as client:
        messages = [
            create_claude_message("user", "What's the weather like in New York?")
        ]
        
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        request = create_claude_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            max_tokens=200
        )
        
        response = await client.chat_completion(request)
        print(f"Response: {response.text}")
        print(f"Tool calls: {response.tool_calls}")
```

## 🔄 Replace Your Existing API Handler

### Before (API Handler)

```python
from pipeline.science.pipeline.api_handler import ApiHandler

config = {'llm_source': 'azure', 'temperature': 0.7}
handler = ApiHandler(config)
response = handler.models['basic']['instance'].invoke("Hello")
```

### After (Claude API Format)

```python
from claude_api_integration_example import ClaudeAPIHandler

handler = ClaudeAPIHandler()
response = await handler.basic_model_invoke("Hello")
```

## 🎯 Model Mapping

| Claude Model | Maps to Azure Deployment | Use Case |
|--------------|-------------------------|----------|
| `claude-3-5-haiku` | `SMALL_MODEL` (gpt-35-turbo) | Fast responses |
| `claude-3-5-sonnet` | `MIDDLE_MODEL` (your deployment) | Standard tasks |
| `claude-3-5-opus` | `BIG_MODEL` (your deployment) | Advanced tasks |

## 📁 Integration with Your Existing Files

### Replace calls in these files:

1. **`get_response.py`** - Replace API handler calls
2. **`get_rag_response.py`** - Replace API handler calls  
3. **`inference.py`** - Replace API handler calls
4. **`images_understanding.py`** - Replace OpenAI calls
5. **Any other file using OpenAI API**

### Example: Replace in `get_response.py`

```python
# Old way
from pipeline.science.pipeline.api_handler import ApiHandler
handler = ApiHandler(config)
response = handler.models['basic']['instance'].invoke(prompt)

# New way
from claude_api_integration_example import ClaudeAPIHandler
handler = ClaudeAPIHandler()
response = await handler.basic_model_invoke(prompt)
```

## 🔧 Configuration

### Proxy Configuration (Auto-generated)

The setup script creates this in `pipeline/science/features_lab/claude-code-proxy/.env`:

```bash
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-35-turbo
```

### Environment Variables

```bash
# Set these in your .env file
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
CLAUDE_PROXY_URL=http://localhost:8082
CLAUDE_PROXY_KEY=your-claude-key
```

## 🧪 Testing

### Test the integration:

```bash
# Test basic functionality
python claude_api_wrapper.py

# Test integration examples
python claude_api_integration_example.py
```

### Test specific features:

```python
# Test basic chat
await example_basic_chat()

# Test streaming
await example_streaming_chat()

# Test function calling
await example_function_calling()
```

## 🎯 Benefits

1. **Claude API Format**: Use Claude API format directly in your code
2. **Azure Infrastructure**: Keep using your existing Azure OpenAI server
3. **Easy Migration**: Simple replacement of API handler calls
4. **Full Features**: Get all Claude API features (function calling, streaming, etc.)
5. **Cost Control**: Still using your Azure OpenAI pricing

## 🔍 Troubleshooting

### Proxy Not Running
```bash
# Check if proxy is running
curl http://localhost:8082/health

# Start proxy if needed
cd pipeline/science/features_lab/claude-code-proxy
python start_proxy.py
```

### Wrong Azure Configuration
```bash
# Check your Azure OpenAI settings
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT

# Update proxy config if needed
python claude_proxy_setup_correct.py
```

### Import Errors
```bash
# Make sure you're in the right directory
cd pipeline/science/pipeline

# Install dependencies
pip install httpx
```

## 📊 Comparison

| Approach | What You Get | What You Use |
|----------|--------------|--------------|
| **Direct Azure OpenAI** | Azure features | Azure OpenAI |
| **Claude API + Proxy** | Claude API features | Azure OpenAI |
| **Direct Claude API** | Claude features | Claude API |

The **Claude API + Proxy** approach gives you Claude API features with Azure infrastructure!

## 🎉 Summary

Your goal is achieved by:
1. ✅ Using Claude API format in your code
2. ✅ Converting requests via proxy (Claude → OpenAI format)
3. ✅ Using your Azure OpenAI server (existing infrastructure)

This gives you the best of both worlds: Claude API features + Azure infrastructure! 