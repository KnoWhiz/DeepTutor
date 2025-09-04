# Gemini CLI Chatbot Integration

A Streamlit-based chatbot that integrates with Google's Gemini CLI to provide intelligent document analysis with web search capabilities and streaming responses.

## 🚀 Features

- **📄 PDF Upload & Conversion**: Upload PDF files and automatically convert them to text for analysis
- **💬 Streaming Chat Interface**: Real-time streaming responses for better user experience
- **🔍 Deep Research**: Leverages Gemini CLI's web search capabilities for comprehensive analysis
- **📚 Citation & Bibliography**: Automatically generates citations and references
- **🗂️ Multi-file Support**: Upload and manage multiple documents
- **🎯 Interactive UI**: Clean, modern Streamlit interface with file management

## 📋 Prerequisites

- **Python Environment**: Conda environment with required packages
- **Node.js & npm**: For installing Gemini CLI
- **Gemini API Key**: From Google AI Studio

## 🛠️ Installation & Setup

### 1. Activate Conda Environment

```bash
conda activate deeptutor
```

### 2. Install Gemini CLI

```bash
npm install -g @google/gemini-cli@latest
```

### 3. Set up API Key

Create a `.env` file in this directory:

```bash
# In /Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/Gemini_CLI_integration_test/
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**Get your API key from**: https://makersuite.google.com/app/apikey

### 4. Verify Installation

```bash
python run_chatbot.py
```

This will check your environment setup and launch the chatbot if everything is configured correctly.

## 📁 File Structure

```
Gemini_CLI_integration_test/
├── README.md                 # This file
├── run_chatbot.py           # Main launcher script
├── streamlit_chatbot.py     # Streamlit web interface
├── gemini_cli_handler.py    # Gemini CLI integration logic
├── pdf_converter.py         # PDF to text conversion utility
├── .env                     # API key configuration (create this)
└── __pycache__/            # Python cache directory
```

## 🎮 Usage

### Starting the Chatbot

```bash
python run_chatbot.py
```

This will:
1. ✅ Check if Gemini CLI is installed
2. ✅ Verify your API key is configured
3. ✅ Set up the working directory
4. 🚀 Launch the Streamlit interface at `http://localhost:8501`

### Using the Interface

1. **Upload a Document**: 
   - Click "Browse files" in the sidebar
   - Upload a PDF or TXT file
   - The system will automatically convert PDFs to text

2. **Ask Questions**:
   - Type your question in the chat input
   - Get streaming responses with web search integration
   - Responses include citations and references

3. **Manage Files**:
   - View uploaded files in the sidebar
   - Switch between different documents
   - Remove files when no longer needed

### Example Queries

- "What are the main findings of this research paper?"
- "Compare the methodology used here with recent similar studies"
- "What are the limitations mentioned in this document?"
- "Find recent developments related to the topics discussed"

## 🔧 Technical Details

### Core Components

1. **`main_streaming_function(file_dir, query)`**: 
   - Main function that processes files and queries
   - Returns a generator yielding response chunks
   - Handles both directory paths and specific file paths

2. **`GeminiCLIHandler`**: 
   - Manages Gemini CLI interactions
   - Constructs prompts based on the provided example format
   - Handles streaming output from CLI commands

3. **`PDFConverter`**: 
   - Converts PDF files to text using PyMuPDF
   - Preserves page structure and formatting
   - Handles multi-page documents

4. **Streamlit Interface**: 
   - Provides web-based chat interface
   - Manages file uploads and conversions
   - Displays streaming responses in real-time

### Working Directory

Files are processed in: `/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files`

### CLI Command Format

The system uses the following Gemini CLI command structure:

```bash
gemini --yolo -p "read_file(path=\"/path/to/file\")
Act as a domain expert research assistant.
User question: [USER_QUERY]

Using the loaded document as a primary source, do a deep research workflow:
1) Extract the document's main topics, key points, methods, results, limitations, and important information.
2) From these, craft 3–5 precise queries and call google_web_search to find ≥8 recent, high-quality sources (peer-reviewed or official) since 2024.
3) Select the best 5 diverse sources and call web_fetch to pull details; compare against the document: methods, results, limitations, reproducibility; flag contradictions.
4) Synthesize a markdown brief that answers the question, clearly labeling what comes from the document vs. external sources, with inline numeric citations and a short bibliography; include a comparison table."
```

## 🐛 Troubleshooting

### Common Issues

1. **"Gemini CLI not found"**
   - Ensure Node.js and npm are installed
   - Run: `npm install -g @google/gemini-cli@latest`

2. **"GEMINI_API_KEY not configured"**
   - Create a `.env` file in this directory
   - Add your API key: `GEMINI_API_KEY=your_key_here`

3. **"No .txt or .pdf files found"**
   - Ensure you've uploaded a file before asking questions
   - Check that the file uploaded successfully

4. **Streaming not working**
   - Check your internet connection
   - Verify API key has sufficient quota
   - Try with a smaller document first

### Logs and Debugging

- Check the terminal/console for detailed error messages
- The launcher script provides comprehensive environment checking
- Streamlit logs appear in the terminal where you ran the script

## 📦 Dependencies

All required packages are included in the `requirements.txt` file in the project root:

- `streamlit` - Web interface
- `PyMuPDF` (fitz) - PDF processing
- `pathlib` - File path handling
- `subprocess` - CLI interaction
- Standard library modules (os, sys, logging, etc.)

## 🤝 Contributing

This is a research/testing implementation. For improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the DeepTutor research framework. Please refer to the main project license.

## 🔗 Related Links

- [Google Gemini CLI Documentation](https://github.com/google/gemini-cli)
- [Google AI Studio](https://makersuite.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)

---

**Happy Chatting! 🤖✨** 