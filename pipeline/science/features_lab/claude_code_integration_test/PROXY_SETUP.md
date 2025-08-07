# Claude Code Proxy Integration Guide

This guide explains how to use the Claude Code proxy server with your DeepTutor pipeline to route Claude Code requests through the proxy to OpenAI or other providers.

## Overview

The proxy server allows you to:
- Use Claude Code API format but route to OpenAI/other providers
- Reduce costs by using cheaper models
- Maintain the same Claude Code interface
- Switch between different providers easily

## Setup Steps

### 1. Start the Proxy Server

First, start the proxy server from the proxy directory:

```bash
cd pipeline/science/features_lab/proxy/

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Example for OpenAI:
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
HOST="0.0.0.0"
PORT="8082"

# Start the server
python start_proxy.py
```

### 2. Configure Your Pipeline

Set the environment variable to point to the proxy:

```bash
# Set the proxy URL
export ANTHROPIC_BASE_URL="http://localhost:8082"

# Set any API key (proxy will validate if ANTHROPIC_API_KEY is set in proxy)
export ANTHROPIC_API_KEY="any-value-or-your-actual-key"
```

### 3. Verify Configuration

Run the verification script to confirm the proxy is being used:

```bash
cd pipeline/science/features_lab/claude_code_integration_test/
python verify_claude_code_usage.py
```

You should see output indicating that requests are going through the proxy.

## Configuration Examples

### OpenAI Provider

```bash
# Proxy .env configuration
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"

# Your pipeline environment
export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_API_KEY="any-value"
```

### Azure OpenAI Provider

```bash
# Proxy .env configuration
OPENAI_API_KEY="your-azure-key"
OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
BIG_MODEL="gpt-4"
MIDDLE_MODEL="gpt-4"
SMALL_MODEL="gpt-35-turbo"

# Your pipeline environment
export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_API_KEY="any-value"
```

### Local Models (Ollama)

```bash
# Proxy .env configuration
OPENAI_API_KEY="dummy-key"
OPENAI_BASE_URL="http://localhost:11434/v1"
BIG_MODEL="llama3.1:70b"
MIDDLE_MODEL="llama3.1:70b"
SMALL_MODEL="llama3.1:8b"

# Your pipeline environment
export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_API_KEY="any-value"
```

## Model Mapping

The proxy maps Claude Code requests to your configured models:

| Claude Code Request | Mapped To | Environment Variable |
|-------------------|-----------|-------------------|
| `claude-3-5-sonnet-20241022` | `MIDDLE_MODEL` | Default: `gpt-4o` |
| `claude-3-5-haiku-20241022` | `SMALL_MODEL` | Default: `gpt-4o-mini` |
| `claude-3-5-opus-20241022` | `BIG_MODEL` | Default: `gpt-4o` |

## Testing the Integration

### 1. Test Proxy Connection

```bash
# Test the proxy directly
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any-value" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 2. Test Pipeline Integration

```bash
# Run the verification script
python verify_claude_code_usage.py

# Run the get_response test
python test_claude_code_get_response.py
```

## Environment Variables

### Required for Proxy

- `OPENAI_API_KEY` - Your target provider API key
- `OPENAI_BASE_URL` - Target provider base URL

### Optional for Proxy

- `ANTHROPIC_API_KEY` - For client validation
- `BIG_MODEL`, `MIDDLE_MODEL`, `SMALL_MODEL` - Model mapping
- `HOST`, `PORT` - Server settings

### Required for Pipeline

- `ANTHROPIC_BASE_URL` - Set to proxy URL (e.g., `http://localhost:8082`)
- `ANTHROPIC_API_KEY` - Any value or your actual key

## Troubleshooting

### Common Issues

1. **Proxy not starting**
   ```bash
   # Check if port is in use
   lsof -i :8082
   
   # Kill existing process
   kill -9 <PID>
   ```

2. **Connection refused**
   ```bash
   # Check if proxy is running
   curl http://localhost:8082/health
   
   # Check firewall settings
   sudo ufw status
   ```

3. **Authentication errors**
   ```bash
   # Verify API key in proxy .env
   cat .env | grep OPENAI_API_KEY
   
   # Check proxy logs
   tail -f proxy.log
   ```

### Debug Mode

Enable debug logging:

```bash
# In proxy .env
LOG_LEVEL=DEBUG

# In your pipeline
export LOG_LEVEL=DEBUG
```

## Performance Considerations

- **Latency**: Proxy adds ~10-50ms overhead
- **Throughput**: Proxy handles multiple concurrent requests
- **Caching**: Consider adding caching layer for repeated requests
- **Monitoring**: Monitor proxy logs for performance issues

## Security

- **API Key Protection**: Store keys securely in `.env` files
- **Network Security**: Use HTTPS in production
- **Access Control**: Set `ANTHROPIC_API_KEY` in proxy for validation
- **Rate Limiting**: Monitor API usage to avoid rate limits

## Production Deployment

For production use:

1. **Use HTTPS**: Set up SSL certificates
2. **Load Balancing**: Use multiple proxy instances
3. **Monitoring**: Add health checks and metrics
4. **Backup**: Have fallback providers configured

```bash
# Production proxy URL
export ANTHROPIC_BASE_URL="https://your-proxy-domain.com"
```

## Cost Optimization

The proxy enables cost optimization by:

- Using cheaper models (e.g., GPT-4o-mini instead of Claude)
- Switching between providers based on cost
- Using local models for development
- Implementing usage-based routing

## Example Workflow

```bash
# 1. Start proxy
cd proxy/
python start_proxy.py

# 2. Set environment
export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_API_KEY="any-value"

# 3. Run your pipeline
cd ../claude_code_integration_test/
python verify_claude_code_usage.py

# 4. Use in your application
# All Claude Code requests now go through the proxy!
```
