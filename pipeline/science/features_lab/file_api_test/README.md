```bash
# Interactive mode
conda activate deeptutor
python openai_file_api_test.py

# Test mode
python test_chatbot.py
```

# Azure OpenAI PDF Chatbot

A terminal-based chatbot that analyzes PDF files using Azure OpenAI's Assistants API with file search capabilities. It provides responses with source citations based on the uploaded PDF content.

## Features

- 📤 Upload multiple PDF files to Azure OpenAI
- 🤖 Create an AI assistant with file search capabilities
- 💬 Interactive terminal chat interface
- 📝 Source citations for each response
- 🧹 Proper error handling and cleanup
- 📊 Logging for debugging and monitoring

## Requirements

### Environment Variables

Create a `.env` file in your project root with:

```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

### Python Dependencies

```bash
pip install openai python-dotenv
```

## Usage

### Interactive Mode

Run the main script to start an interactive chat session:

```bash
conda activate deeptutor
python openai_file_api_test.py
```

### Test Mode

Run the test script to verify functionality:

```bash
python test_chatbot.py
```

### Customizing PDF Files

Edit the `pdf_files` list in the `main()` function of `openai_file_api_test.py`:

```python
def main():
    # Add your PDF files here
    pdf_files = [
        "/path/to/your/first.pdf",
        "/path/to/your/second.pdf",
        # Add more files as needed
    ]
    # ... rest of the function
```

## How It Works

1. **File Upload**: PDFs are uploaded to Azure OpenAI with the "assistants" purpose
2. **Vector Store**: A vector store is created to enable semantic search across documents
3. **Assistant Creation**: An AI assistant is created with file search capabilities
4. **Chat Session**: Interactive terminal session where you can ask questions
5. **Citations**: Responses include source citations from the analyzed documents
6. **Cleanup**: Resources are automatically cleaned up after the session

## Example Usage

```
🤖 Azure OpenAI PDF Chatbot
================================================================================
Ask me anything about the uploaded PDF documents!
Type 'quit', 'exit', or 'bye' to end the session.
Type 'help' for available commands.
================================================================================

💬 You: What is this paper about?

🤖 Assistant: This paper discusses multiplexed single photon sources based on quantum dots. The research focuses on developing efficient methods for generating single photons using semiconductor quantum dots in optical cavities. [Source: assistant-XYZ123] The work demonstrates how to create multiple independent single photon sources that can be used for quantum communication and quantum computing applications. [Source: assistant-XYZ123]

💬 You: What are the main findings?

🤖 Assistant: The main findings include:
1. Successful demonstration of multiplexed single photon generation with high efficiency [Source: assistant-XYZ123]
2. Achievement of high collection efficiency through optimized cavity design [Source: assistant-XYZ123]
3. Demonstration of indistinguishable photons suitable for quantum interference experiments [Source: assistant-XYZ123]
...
```

## Commands

- **Ask questions**: Type any question about the PDF content
- **help**: Show available commands
- **quit/exit/bye**: End the chat session

## File Limitations

- Maximum file size: 512MB per PDF
- Supported format: PDF only
- Maximum pages: 100 pages per request
- Total content: 32MB across all files

## Troubleshooting

### Common Issues

1. **Missing environment variables**: Ensure `.env` file contains `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
2. **File not found**: Check that PDF file paths are correct and files exist
3. **API errors**: Verify your Azure OpenAI resource has the required model deployments
4. **Timeout errors**: Large PDFs may take longer to process; the script will wait up to 5 minutes

### Logs

Check `chatbot.log` for detailed logging information about uploads, processing, and any errors.

## Notes

- The script uses GPT-4o model for optimal performance with file search
- Files are automatically cleaned up after each session
- Vector stores are created fresh for each session
- The assistant uses the 2024-05-01-preview API version

## Security

- API keys are loaded from environment variables
- Files are uploaded to Microsoft-managed storage
- All resources are cleaned up after use
- No sensitive data is logged 