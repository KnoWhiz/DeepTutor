# Claude Client Migration Guide

**Your Goal**: Replace existing Claude client requests with proxy-enabled requests that convert to Azure OpenAI format.

## 🎯 Migration Strategy

Replace your existing `client.messages.create()` calls with the same format, but using a wrapper that sends requests through the proxy to Azure OpenAI.

## 🔄 Migration Examples

### Example 1: Basic Chat Request

#### Before (Direct Claude API)
```python
from anthropic import Anthropic

client = Anthropic(api_key="your-key")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    temperature=0.3,
    system=system_message,
    messages=messages
)
return response.content[0].text
```

#### After (Proxy-enabled, converts to Azure OpenAI)
```python
from claude_client_wrapper import create_claude_client

async with create_claude_client() as client:
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.3,
        system=system_message,
        messages=messages
    )
    return response.text
```

### Example 2: Your Existing Codebase Analyzer

#### Before (from your claude_code_integration_test.py)
```python
from anthropic import Anthropic

class CodebaseAnalyzer:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def get_contextual_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        # ... prepare messages ...
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            system=system_message,
            messages=messages
        )
        
        return response.content[0].text
```

#### After (Proxy-enabled)
```python
from claude_client_wrapper import create_claude_client

class CodebaseAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_contextual_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        # ... prepare messages ...
        
        async with create_claude_client(self.api_key) as client:
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.3,
                system=system_message,
                messages=messages
            )
            
            return response.text
```

### Example 3: Replace OpenAI API Calls

#### Before (OpenAI API calls in your pipeline)
```python
# In get_response.py, inference.py, etc.
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=deployment_name,
    temperature=temperature,
    streaming=stream
)

response = llm.invoke(prompt)
```

#### After (Claude client format with proxy)
```python
# In get_response.py, inference.py, etc.
from claude_client_wrapper import create_claude_client

async def get_response(prompt: str, system_message: str = None):
    async with create_claude_client() as client:
        messages = [{"role": "user", "content": prompt}]
        
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            system=system_message,
            messages=messages
        )
        
        return response.text
```

## 📁 File-by-File Migration

### 1. Replace in `claude_code_integration_test.py`

```python
# Replace this:
from anthropic import Anthropic
self.client = Anthropic(api_key=api_key)

# With this:
from claude_client_wrapper import create_claude_client
self.api_key = api_key

# Replace this:
response = self.client.messages.create(...)
return response.content[0].text

# With this:
async with create_claude_client(self.api_key) as client:
    response = await client.messages.create(...)
    return response.text
```

### 2. Replace in `get_response.py`

```python
# Replace API handler calls:
from pipeline.science.pipeline.api_handler import ApiHandler
handler = ApiHandler(config)
response = handler.models['basic']['instance'].invoke(prompt)

# With Claude client calls:
from claude_client_wrapper import create_claude_client
async with create_claude_client() as client:
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.text
```

### 3. Replace in `inference.py`

```python
# Replace Azure OpenAI calls:
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(...)
response = llm.invoke(prompt)

# With Claude client calls:
from claude_client_wrapper import create_claude_client
async with create_claude_client() as client:
    response = await client.messages.create(
        model="claude-3-5-opus-20241022",  # For advanced tasks
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.text
```

### 4. Replace in `images_understanding.py`

```python
# Replace OpenAI calls:
import openai
client = openai.OpenAI(...)
response = client.chat.completions.create(...)

# With Claude client calls:
from claude_client_wrapper import create_claude_client
async with create_claude_client() as client:
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.text
```

## 🔧 Configuration

### 1. Set up the proxy

```bash
# Run the setup script
python claude_proxy_setup_correct.py

# Start the proxy server
cd pipeline/science/features_lab/claude-code-proxy
python start_proxy.py
```

### 2. Environment variables

```bash
# In your .env file
ANTHROPIC_API_KEY=your-claude-key
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
CLAUDE_PROXY_URL=http://localhost:8082
```

## 🎯 Model Mapping

| Claude Model | Maps to Azure Deployment | Use Case |
|--------------|-------------------------|----------|
| `claude-3-5-haiku` | `SMALL_MODEL` (gpt-35-turbo) | Fast responses |
| `claude-3-5-sonnet` | `MIDDLE_MODEL` (your deployment) | Standard tasks |
| `claude-3-5-opus` | `BIG_MODEL` (your deployment) | Advanced tasks |

## 📝 Migration Checklist

- [ ] Set up proxy server pointing to Azure OpenAI
- [ ] Replace `from anthropic import Anthropic` with `from claude_client_wrapper import create_claude_client`
- [ ] Replace `client = Anthropic(api_key)` with `create_claude_client(api_key)`
- [ ] Replace `response = client.messages.create(...)` with `response = await client.messages.create(...)`
- [ ] Replace `response.content[0].text` with `response.text`
- [ ] Make functions async where needed
- [ ] Test all migrated functions

## 🧪 Testing Migration

### Test the migration:

```bash
# Test the wrapper
python claude_client_wrapper.py

# Test your migrated code
python your_migrated_file.py
```

### Verify proxy conversion:

```bash
# Check proxy logs to see conversion
cd pipeline/science/features_lab/claude-code-proxy
python start_proxy.py
# Watch the logs to see Claude → Azure OpenAI conversion
```

## 🎯 Benefits

1. **Same API**: Use the exact same `client.messages.create()` format
2. **Azure Infrastructure**: Keep using your existing Azure OpenAI server
3. **Easy Migration**: Minimal code changes required
4. **Full Features**: Get all Claude API features
5. **Cost Control**: Still using your Azure OpenAI pricing

## 🚨 Important Notes

1. **Async Required**: The wrapper uses async/await, so your functions need to be async
2. **Proxy Must Be Running**: Make sure the proxy server is running before using the wrapper
3. **Same Response Format**: The wrapper mimics Anthropic's response format
4. **Error Handling**: The wrapper includes proper error handling and logging

## 🎉 Summary

Your migration is complete when:
1. ✅ All `client.messages.create()` calls use the wrapper
2. ✅ Proxy server is running and configured for Azure OpenAI
3. ✅ All functions are async where needed
4. ✅ All tests pass

You now have Claude API features with Azure OpenAI infrastructure! 