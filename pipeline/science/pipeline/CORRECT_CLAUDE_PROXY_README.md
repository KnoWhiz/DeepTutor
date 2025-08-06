# Correct Claude Code Proxy Integration

**Your Goal**: Use Claude Code CLI's advanced services while still using your Azure OpenAI server.

**Correct Architecture**:
```
Claude Code CLI → Claude Proxy Server → Azure OpenAI Server
```

## 🎯 What You Actually Want

You want to:
1. Use **Claude Code CLI** directly (with its advanced features)
2. Have the **proxy server** convert Claude requests to OpenAI format
3. Send those requests to your **existing Azure OpenAI server**

This gives you the best of both worlds: Claude Code's advanced features + your Azure OpenAI infrastructure.

## 🚀 Correct Setup

### 1. Configure the Proxy for Azure OpenAI

```bash
# Run the correct setup script
python claude_proxy_setup_correct.py
```

This creates the proper proxy configuration that points to your Azure OpenAI server.

### 2. Start the Proxy Server

```bash
cd pipeline/science/features_lab/claude-code-proxy
python start_proxy.py
```

The proxy will run on `http://localhost:8082` and convert Claude requests to Azure OpenAI format.

### 3. Use Claude Code CLI

```bash
# Set environment variables
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=your-claude-key

# Use Claude Code CLI
claude
```

Now when you use `claude`, it will:
- Send requests to the proxy server
- Proxy converts them to Azure OpenAI format
- Requests go to your Azure OpenAI server
- You get Claude Code's advanced features with your Azure infrastructure

## 🔧 Configuration Details

### Proxy Configuration (Auto-generated)

The setup script creates this configuration:

```bash
# In pipeline/science/features_lab/claude-code-proxy/.env
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-35-turbo
```

### Model Mapping

| Claude Model | Maps to Azure Deployment | Use Case |
|--------------|-------------------------|----------|
| `claude-3-5-haiku` | `SMALL_MODEL` (gpt-35-turbo) | Fast responses |
| `claude-3-5-sonnet` | `MIDDLE_MODEL` (your deployment) | Standard tasks |
| `claude-3-5-opus` | `BIG_MODEL` (your deployment) | Advanced tasks |

## 📝 Usage Examples

### Basic Usage

```bash
# Start interactive chat
claude

# Ask a question
claude "What is machine learning?"
```

### File Processing

```bash
# Process a file with Claude Code features
claude --file your_document.pdf

# The proxy converts this to Azure OpenAI format
```

### Streaming

```bash
# Get streaming responses
claude --stream "Explain quantum computing"
```

### Model Selection

```bash
# Use specific Claude model (maps to Azure deployment)
claude --model claude-3-5-opus-20241022 "Complex analysis task"
```

## 🔄 Integration with Your Existing Code

### Option 1: Replace API Handler Calls

Instead of using your API handler, use Claude Code CLI directly:

```python
# Old way (API handler)
from pipeline.science.pipeline.api_handler import ApiHandler
handler = ApiHandler(config)
response = handler.models['basic']['instance'].invoke("Hello")

# New way (Claude Code CLI)
import subprocess
result = subprocess.run(['claude', 'Hello'], capture_output=True, text=True)
response = result.stdout
```

### Option 2: Use Claude Code as a Service

```python
import subprocess
import json

def claude_request(prompt, model="claude-3-5-sonnet-20241022"):
    """Send request to Claude Code CLI through proxy."""
    cmd = ['claude', '--model', model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# Use in your existing code
response = claude_request("Analyze this paper", model="claude-3-5-opus-20241022")
```

## 🎯 Benefits of This Approach

1. **Claude Code Advanced Features**: Get all Claude Code CLI features (file processing, streaming, etc.)
2. **Azure OpenAI Infrastructure**: Keep using your existing Azure OpenAI server
3. **No Code Changes**: Minimal changes to your existing workflow
4. **Cost Control**: Still using your Azure OpenAI pricing
5. **Model Flexibility**: Easy to switch between Claude models

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

### Claude Code CLI Issues
```bash
# Install Claude Code CLI
pip install anthropic

# Check configuration
claude --help
```

## 🚨 Important Notes

1. **This is NOT the API handler approach**: You're using Claude Code CLI directly, not through the API handler
2. **Proxy converts requests**: The proxy server converts Claude API format to OpenAI format
3. **Azure OpenAI receives requests**: Your Azure OpenAI server receives the converted requests
4. **Claude Code features work**: You get all Claude Code CLI features while using Azure infrastructure

## 📊 Comparison

| Approach | What You Get | What You Use |
|----------|--------------|--------------|
| **Direct Azure OpenAI** | Azure features | Azure OpenAI |
| **Claude Code + Proxy** | Claude Code features | Azure OpenAI |
| **Direct Claude API** | Claude features | Claude API |

The **Claude Code + Proxy** approach gives you the best of both worlds!

## 🎉 Summary

Your goal is achieved by:
1. ✅ Using Claude Code CLI (advanced features)
2. ✅ Converting requests via proxy (Claude → OpenAI format)
3. ✅ Using your Azure OpenAI server (existing infrastructure)

This is the **correct direction** for your use case! 