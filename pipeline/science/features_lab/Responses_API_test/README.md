# AI Research Chatbot with OpenAI Responses API

A sophisticated Streamlit-based chatbot that leverages OpenAI's Responses API to provide intelligent responses with web search capabilities and document analysis features. The application provides real-time status updates and step-by-step visibility into the AI's tool calling process.

## 🌟 Features

- **🔍 Web Search Integration**: Real-time web search using DuckDuckGo API
- **📄 Document Analysis**: Upload and analyze PDF and text documents
- **🔄 Streaming Responses**: Real-time response generation with live updates
- **📊 Status Tracking**: Step-by-step visibility into AI processing steps
- **🛠️ Tool Calling**: Automatic tool selection and execution
- **💬 Conversation History**: Maintains context across multiple exchanges
- **🎨 Modern UI**: Clean, responsive Streamlit interface with custom styling

## 🏗️ Architecture

The application consists of three main components:

1. **`main.py`**: Core chatbot logic with streaming generator functions
2. **`ui.py`**: Streamlit user interface with real-time updates
3. **`README.md`**: Documentation and setup instructions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda environment named `deeptutor`
- OpenAI API key

### Installation

1. **Activate the Conda environment:**
   ```bash
   conda activate deeptutor
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify environment variables:**
   The `.env` file should contain your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application:**
   ```bash
   cd pipeline/science/features_lab/Responses_API_test
   streamlit run ui.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

The application uses the following environment variables from the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key (fallback)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint (fallback)

### Supported File Types

- **Text files**: `.txt`
- **PDF documents**: `.pdf`

### Available Tools

1. **Web Search Tool**
   - Function: `web_search`
   - Description: Searches the web for current information
   - Parameters: `query` (string)

2. **Document Analysis Tool**
   - Function: `analyze_document`
   - Description: Analyzes uploaded documents
   - Parameters: `document_content` (string), `analysis_type` (enum)
   - Analysis types: `summary`, `key_points`, `research`, `detailed_analysis`

## 📱 Usage Guide

### Basic Chat

1. Type your message in the text area at the bottom
2. Click "🚀 Send Message" to submit
3. Watch real-time status updates in the right panel
4. View the AI's response as it generates

### Document Analysis

1. Upload a document using the file uploader in the sidebar
2. Ask questions about the document in your message
3. The AI will automatically analyze the document and provide insights

### Web Search

1. Ask questions that require current information
2. The AI will automatically perform web searches when needed
3. View search results and tool execution status in real-time

### Status Monitoring

- **Left Panel**: Main chat interface with conversation history
- **Right Panel**: Real-time status updates showing:
  - Current processing step
  - Tool execution status
  - Tool outputs and results
  - Timestamps for each operation

## 🔍 Status Update Types

The application provides detailed status updates for the following steps:

- **Initialization**: Setting up the request and processing files
- **File Processing**: Extracting content from uploaded documents
- **API Call**: Communicating with OpenAI's API
- **Response Processing**: Handling streaming responses
- **Tool Call**: Initiating tool functions
- **Tool Execution**: Running web search or document analysis
- **Final Response**: Generating the complete response
- **Completion**: Finishing the response generation
- **Error**: Handling any errors that occur

## 🛠️ Technical Details

### Core Components

#### ResponsesAPIChatbot Class
- Manages OpenAI API interactions
- Handles tool calling and execution
- Provides streaming response generation
- Includes error handling and status updates

#### StatusUpdate DataClass
- Tracks processing steps and status
- Includes timestamps and tool information
- Provides structured status reporting

#### Tool Functions
- `_web_search()`: Performs web searches using DuckDuckGo API
- `_analyze_document()`: Analyzes documents using OpenAI
- `_extract_text_from_file()`: Extracts text from uploaded files

### Streaming Architecture

The application uses a generator-based streaming architecture:

1. **Request Processing**: Prepares messages and processes files
2. **API Streaming**: Receives streaming responses from OpenAI
3. **Tool Execution**: Executes tools when requested by the AI
4. **Status Updates**: Yields status updates throughout the process
5. **Response Assembly**: Combines all responses into final output

### Error Handling

- Comprehensive exception handling at all levels
- Graceful degradation when tools fail
- User-friendly error messages
- Status updates for error conditions

## 🔒 Security Considerations

- API keys are loaded from environment variables
- File uploads are processed in memory when possible
- Temporary files are cleaned up after processing
- Input validation for all user inputs

## 📊 Performance

- Streaming responses for better user experience
- Efficient file processing with size limits
- Parallel tool execution when possible
- Optimized status update frequency

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file contains the correct API key
   - Verify the file is in the correct directory

2. **File upload errors**
   - Check file size (should be reasonable for processing)
   - Ensure file type is supported (.txt, .pdf)

3. **Web search failures**
   - Check internet connectivity
   - DuckDuckGo API may have rate limits

4. **Streaming issues**
   - Refresh the browser page
   - Check browser console for JavaScript errors

### Debug Mode

To enable debug mode, modify the main.py file to include more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the DeepTutor system. Please refer to the main project license.

## 🔮 Future Enhancements

- [ ] Support for more file types (DOCX, XLSX, etc.)
- [ ] Integration with more search APIs
- [ ] Voice input/output capabilities
- [ ] Multi-language support
- [ ] Advanced document parsing
- [ ] Custom tool creation interface
- [ ] Response caching and optimization
- [ ] User authentication and sessions

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs for error details
3. Ensure all dependencies are correctly installed
4. Verify environment variables are set

---

**Happy chatting! 🤖✨**
