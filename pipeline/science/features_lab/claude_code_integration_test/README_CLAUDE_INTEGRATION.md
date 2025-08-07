# Claude Code Integration with API Handler

This document explains how to use Anthropic's Claude Code model with the existing API handler infrastructure.

## Overview

The API handler has been extended to support Anthropic's Claude Code model (`claude-3-5-sonnet-20241022`) alongside the existing Azure, OpenAI, and SambaNova models.

## Setup

### 1. Install Dependencies

Add the required package to your environment:

```bash
pip install langchain-anthropic==0.2.1
```

The package is already added to `requirements.txt`.

### 2. Environment Variables

Make sure you have the following environment variable set:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Configuration

To use Claude Code, update your configuration to set the LLM source to 'anthropic':

```python
config['llm']['llm_source'] = 'anthropic'
```

## Switching the Entire Pipeline to Claude Code

To make Claude Code be used for **all responses and summaries** in the pipeline (including `get_response.py`), you have two options:

### Option 1: Update Configuration File (Recommended)

Edit `pipeline/science/pipeline/config.json`:

```json
{
    "llm": {
        "llm_source": "anthropic",  // Changed from "azure" to "anthropic"
        "level": "advanced",
        "temperature": 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env"
    },
    // ... rest of config
}
```

### Option 2: Update Configuration Programmatically

```python
from pipeline.science.pipeline.config import load_config

# Load configuration
config = load_config()

# Switch to Claude Code
config['llm']['llm_source'] = 'anthropic'

# Now all get_response.py functions will use Claude Code
```

### Verification

Run the verification script to confirm Claude Code is being used:

```bash
cd pipeline/science/features_lab/claude_code_integration_test/
python verify_claude_code_usage.py
```

This will show you which models are being used for each model type (basic, advanced, creative, backup).

## Usage

### Basic Usage

```python
from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.config import load_config

# Load configuration
config = load_config()
config['llm']['llm_source'] = 'anthropic'

# Create API handler
api_handler = ApiHandler(config['llm'], stream=True)

# Get Claude Code model instances
basic_model = api_handler.models['basic']['instance']
advanced_model = api_handler.models['advanced']['instance']
creative_model = api_handler.models['creative']['instance']
backup_model = api_handler.models['backup']['instance']
```

### Model Types

When using `llm_source='anthropic'`:

- **Basic**: Uses Claude Code (`claude-3-5-sonnet-20241022`)
- **Advanced**: Uses Claude Code (`claude-3-5-sonnet-20241022`)
- **Creative**: Uses Claude Code (`claude-3-5-sonnet-20241022`)
- **Backup**: Uses Azure GPT-4.1-mini (fallback)

### Context Window

Claude Code models have a context window of **200,000 tokens**, which is larger than the Azure models (128,000 tokens).

### Streaming Support

Claude Code supports streaming responses:

```python
# Enable streaming
api_handler = ApiHandler(config['llm'], stream=True)
model = api_handler.models['advanced']['instance']

# Use streaming
response = model.stream("Your prompt here")
for chunk in response:
    if hasattr(chunk, 'content'):
        print(chunk.content, end='', flush=True)
```

## Integration with Existing Pipeline

The Claude Code integration works seamlessly with the existing pipeline:

```python
from pipeline.science.pipeline.utils import get_llm

# Get Claude Code model through the existing interface
llm = get_llm('advanced', config['llm'], stream=True)

# Use with existing pipeline functions
response = llm.stream(prompt)
```

## Testing

Run the test script to verify the integration:

```bash
cd pipeline/science/features_lab/claude_code_integration_test/
python claude_api_handler_test.py
```

For testing with get_response.py functions:

```bash
python test_claude_code_get_response.py
```

## Key Features

1. **Code Understanding**: Claude Code is specifically designed for understanding and analyzing codebases
2. **Large Context Window**: 200K token context window for handling large codebases
3. **Streaming Support**: Real-time response streaming
4. **Seamless Integration**: Works with existing pipeline infrastructure
5. **Fallback Support**: Uses Azure models as backup

## Configuration Options

The Claude Code model uses these default settings:

- **Model**: `claude-3-5-sonnet-20241022`
- **Temperature**: Configurable via `config['llm']['temperature']`
- **Max Tokens**: 4000 (configurable)
- **Streaming**: Supported
- **Context Window**: 200,000 tokens

## Error Handling

The integration includes proper error handling:

- Missing API key validation
- Network error handling
- Fallback to Azure models if Claude Code fails
- Graceful degradation for streaming issues

## Migration from Other Models

To migrate from Azure/OpenAI to Claude Code:

1. Set `config['llm']['llm_source'] = 'anthropic'`
2. Ensure `ANTHROPIC_API_KEY` is set
3. Update any model-specific prompts if needed
4. Test with your specific use cases

## Performance Considerations

- Claude Code may have different response times compared to Azure models
- The 200K context window allows for larger inputs but may increase latency
- Consider using the backup model for critical applications
- Monitor API usage and costs

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `ANTHROPIC_API_KEY` is set
2. **Import Errors**: Install `langchain-anthropic==0.2.1`
3. **Streaming Issues**: Check network connectivity and API limits
4. **Context Limits**: Monitor token usage for large inputs

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See `claude_api_handler_test.py` for complete working examples of:

- Basic model instantiation
- Streaming responses
- Error handling
- Configuration management

## Impact on get_response.py

When you switch to Claude Code, the following functions in `get_response.py` will use Claude Code instead of Azure/OpenAI:

1. **`get_multiple_files_summary()`** - For generating summaries of multiple documents
2. **`get_response()`** - For all regular responses in different modes (LITE, ADVANCED, BASIC)
3. **`get_query_helper()`** - For query processing and question analysis
4. **`generate_follow_up_questions()`** - For generating follow-up questions

All these functions will automatically use Claude Code once you set `llm_source='anthropic'` in the configuration.
