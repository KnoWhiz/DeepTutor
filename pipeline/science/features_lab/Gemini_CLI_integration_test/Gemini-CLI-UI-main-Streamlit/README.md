# Gemini CLI Streamlit Chatbot

A web-based chatbot interface that demonstrates the `gemini_stream.py` functionality with real-time streaming responses.

## Files

- `gemini_stream.py` - Core streaming module that wraps the Gemini CLI
- `streamlit_chatbot.py` - Streamlit web UI for testing the streaming functionality
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Gemini CLI is installed:**
   ```bash
   # Verify Gemini CLI is accessible
   gemini --help
   
   # Test basic functionality
   gemini --prompt "Hello world" --model gemini-2.0-flash
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_chatbot.py
   ```

4. **Open your browser** to the URL shown (usually `http://localhost:8501`)

## Features

### 🤖 **Real-time Streaming**
- Watch responses appear character by character as they're generated
- No waiting for complete responses - see output as it happens

### 💬 **Chat Interface**
- Clean, intuitive chat UI with message history
- Timestamps for all messages
- User and assistant message differentiation

### ⚙️ **Configuration Options**
- **Model Selection**: Choose from different Gemini models
- **Timeout Control**: Set maximum wait time for responses
- **Extra CLI Arguments**: Add custom parameters like temperature, top-p, etc.

### 📊 **Chat Management**
- Clear chat history
- View conversation statistics
- Export chat history as JSON

### 🛡️ **Error Handling**
- Graceful handling of CLI errors, timeouts, and connection issues
- User-friendly error messages with troubleshooting hints

## Usage Examples

### Basic Chat
Just type your message in the chat input and press Enter. The response will stream in real-time.

### Advanced Configuration
Use the sidebar to:
- Change the Gemini model
- Adjust timeout settings
- Add extra CLI arguments like:
  ```
  --temperature 0.7
  --top-p 0.9
  --max-tokens 1000
  ```

### Export Chat History
Click the "Export Chat History" button to download your conversation as a JSON file with timestamps and metadata.

## Troubleshooting

### "Gemini CLI Not Found"
- Ensure the Gemini CLI is installed and in your PATH
- Test with: `gemini --version`

### "Request Timeout"
- Increase the timeout slider in the sidebar
- Try simpler prompts
- Check your internet connection

### "Import Error"
- Make sure you've installed the requirements: `pip install -r requirements.txt`
- Ensure both `gemini_stream.py` and `streamlit_chatbot.py` are in the same directory

## Development

The chatbot uses the `stream_gemini()` function from `gemini_stream.py` to:
1. Execute the Gemini CLI with your prompt
2. Stream stdout/stderr in real-time using threading
3. Handle errors and timeouts gracefully
4. Display results in the Streamlit UI

## Requirements

- Python 3.7+
- Streamlit 1.28.0+
- Gemini CLI (installed and configured)
- Internet connection for Gemini API calls
