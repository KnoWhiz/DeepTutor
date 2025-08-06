# Claude Code Proxy Integration

This integration allows you to use Claude Code through a proxy server instead of direct Azure OpenAI API calls. The proxy converts Claude API requests to OpenAI-compatible requests, enabling you to use Claude Code with your existing infrastructure.

## 🚀 Quick Start

### 1. Start the Claude Proxy Server

```bash
# Navigate to the proxy directory
cd pipeline/science/features_lab/claude-code-proxy

# Install dependencies
pip install -r requirements.txt

# Start the proxy server
python start_proxy.py
```

The proxy server will run on `http://localhost:8082` by default.

### 2. Configure Environment Variables

Copy the example configuration and update it with your values:

```bash
# Copy the example config
cp claude_proxy_config_example.env .env

# Edit the .env file with your actual API keys
```

Required environment variables:
- `CLAUDE_PROXY_URL`: URL of the proxy server (default: http://localhost:8082)
- `CLAUDE_PROXY_KEY`: API key for the proxy (default: your-key)
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key (for fallback)
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint

### 3. Update Your Configuration

In your `config.json`, change the `llm_source` to `claude_proxy`:

```json
{
  "llm_source": "claude_proxy",
  "temperature": 0.7,
  "creative_temperature": 0.9,
  "stream": false,
  "openai_key_dir": "."
}
```

### 4. Test the Integration

```bash
# Run the setup script to test everything
python setup_claude_proxy.py
```

## 🔧 How It Works

### Architecture

```
Your Application → API Handler → Claude Proxy Wrapper → Claude Proxy Server → OpenAI API
```

1. **Your Application**: Uses the existing API handler as before
2. **API Handler**: Routes requests to Claude proxy when `llm_source` is `claude_proxy`
3. **Claude Proxy Wrapper**: Converts LangChain requests to Claude API format
4. **Claude Proxy Server**: Converts Claude API requests to OpenAI format
5. **OpenAI API**: Processes the request and returns the response

### Model Mapping

The integration automatically maps your existing model names to Claude models:

| Your Model Name | Claude Model | Use Case |
|----------------|--------------|----------|
| `gpt-4.1` | `claude-3-5-opus-20241022` | Advanced tasks |
| `gpt-4.1-mini` | `claude-3-5-sonnet-20241022` | Basic tasks |
| `o3-mini` | `claude-3-5-haiku-20241022` | Fast responses |

## 📝 Usage Examples

### Basic Usage

```python
from pipeline.science.pipeline.api_handler import ApiHandler

# Configure for Claude proxy
config = {
    'llm_source': 'claude_proxy',
    'temperature': 0.7,
    'stream': False,
    'openai_key_dir': '.'
}

# Initialize API handler
handler = ApiHandler(config)

# Use as before - no code changes needed!
basic_model = handler.models['basic']['instance']
response = basic_model.invoke("Hello, how are you?")
```

### Streaming Usage

```python
# Enable streaming
config['stream'] = True
handler = ApiHandler(config)

# Get streaming response
streaming_model = handler.models['advanced']['instance']
for chunk in streaming_model.stream("Tell me a story"):
    print(chunk.content, end='', flush=True)
```

### Fallback Configuration

The integration includes automatic fallback to Azure OpenAI if the Claude proxy fails:

```python
# If Claude proxy is unavailable, it will automatically fall back to Azure OpenAI
try:
    response = claude_model.invoke("Hello")
except Exception as e:
    # Fallback to Azure OpenAI happens automatically
    print("Using fallback model")
```

## 🔍 Troubleshooting

### Common Issues

1. **Proxy server not running**
   ```
   Error: Connection refused
   Solution: Start the proxy server with `python start_proxy.py`
   ```

2. **Wrong proxy URL**
   ```
   Error: 404 Not Found
   Solution: Check CLAUDE_PROXY_URL in your .env file
   ```

3. **Missing environment variables**
   ```
   Error: Missing environment variables
   Solution: Copy claude_proxy_config_example.env to .env and update values
   ```

### Testing Connection

```bash
# Test the proxy connection
python setup_claude_proxy.py
```

### Debug Mode

Enable debug logging to see detailed request/response information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Migration from Azure OpenAI

### Before (Azure OpenAI)
```python
# Old configuration
config = {
    'llm_source': 'azure',
    'temperature': 0.7,
    'stream': False
}
```

### After (Claude Proxy)
```python
# New configuration
config = {
    'llm_source': 'claude_proxy',  # Only this line changes!
    'temperature': 0.7,
    'stream': False
}
```

**No other code changes are required!** All your existing API calls will work exactly the same.

## 📊 Performance Comparison

| Feature | Azure OpenAI | Claude Proxy |
|---------|-------------|--------------|
| Response Time | ~1-3s | ~1-3s |
| Streaming | ✅ | ✅ |
| Function Calling | ✅ | ✅ |
| Image Support | ✅ | ✅ |
| Cost | Standard | Depends on proxy config |
| Fallback | ❌ | ✅ (to Azure OpenAI) |

## 🛠️ Advanced Configuration

### Custom Model Mapping

You can customize the model mapping in `api_handler.py`:

```python
claude_model_map = {
    'gpt-4.1': 'claude-3-5-opus-20241022',
    'gpt-4.1-mini': 'claude-3-5-sonnet-20241022',
    'o3-mini': 'claude-3-5-haiku-20241022',
    'custom-model': 'claude-3-5-sonnet-20241022'
}
```

### Custom Proxy Settings

```python
# In your .env file
CLAUDE_PROXY_URL=http://your-custom-proxy:8082
CLAUDE_PROXY_KEY=your-custom-key
```

### Multiple Proxy Instances

You can run multiple proxy instances for load balancing:

```python
# Round-robin between proxies
proxy_urls = [
    "http://localhost:8082",
    "http://localhost:8083",
    "http://localhost:8084"
]
```

## 🔐 Security Considerations

1. **API Key Management**: Store API keys securely in environment variables
2. **Proxy Authentication**: Use the `CLAUDE_PROXY_KEY` for proxy authentication
3. **Network Security**: Ensure the proxy server is only accessible from trusted sources
4. **Request Logging**: Be aware that proxy requests may be logged

## 📚 Additional Resources

- [Claude Code Proxy Documentation](../features_lab/claude-code-proxy/README.md)
- [Integration Guide](../features_lab/claude-code-proxy/INTEGRATION_GUIDE.md)
- [API Handler Documentation](api_handler.py)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python setup_claude_proxy.py` to test your setup
3. Check the proxy server logs for detailed error information
4. Verify your environment variables are correctly set

## 🎯 Benefits

- **No Code Changes**: Existing code works without modification
- **Automatic Fallback**: Falls back to Azure OpenAI if Claude proxy fails
- **Cost Optimization**: Use Claude Code for better pricing
- **Model Flexibility**: Easy to switch between different models
- **Streaming Support**: Full streaming capability maintained
- **Function Calling**: Complete tool use support 