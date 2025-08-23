# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create new environment
conda create --name deeptutor python=3.12
conda activate deeptutor

# Install core dependencies
pip install -r requirements.txt

# Install optional AgentChat extensions
pip install -U "autogen-agentchat" "autogen-ext[openai]"
pip install -U "autogenstudio"
```

### Running the Application
```bash
# Activate conda environment
conda activate deeptutor

# Set environment variable for local development
export ENVIRONMENT=local

# Run the Streamlit application
python -m streamlit run tutor.py
```

### Testing and Development
```bash
# Run specific integration tests
python pipeline/science/features_lab/api_test.py
python pipeline/science/features_lab/deepseek_test.py
python pipeline/science/features_lab/graphrag_api_test.py

# Test file processing capabilities
python pipeline/science/features_lab/file_api_test/test_chatbot.py

# Test source retrieval and indexing
python pipeline/science/features_lab/source_index_locate_test.py

# Run RAG-specific tests
python pipeline/science/features_lab/agentic_rag_test.py
```

### Common Development Tasks
```bash
# Check embedding generation
python pipeline/science/features_lab/load_embeddings_test.py

# Test streaming capabilities
python pipeline/science/features_lab/streaming_test.py

# Verify API integrations
python pipeline/science/features_lab/api_test.py
```

## Architecture Overview

### Core Structure
- **Entry Point**: `tutor.py` - Main Streamlit application
- **Frontend**: `frontend/` - UI components, state management, and authentication
- **Pipeline**: `pipeline/science/pipeline/` - AI processing, embeddings, and chat logic
- **Configuration**: `pipeline/science/pipeline/config.json` - Application settings

### Key Components

#### Frontend Architecture (`frontend/`)
- `ui.py` - Main UI components and layout
- `state.py` - Session state management and file processing
- `auth.py` - AWS Cognito authentication
- `utils.py` - UI utility functions

#### AI Pipeline (`pipeline/science/pipeline/`)
- `tutor_agent.py` - Main agent orchestration
- `tutor_agent_lite.py`, `tutor_agent_basic.py`, `tutor_agent_advanced.py` - Mode-specific implementations
- `embeddings.py`, `embeddings_agent.py` - Vector embedding generation
- `doc_processor.py` - PDF processing and document handling
- `session_manager.py` - Chat session management
- `config.py` - Configuration loading

### Multi-Mode System
The application operates in three distinct modes with different processing approaches:

- **Lite Mode**: 
  - Supports multiple file uploads simultaneously
  - Uses basic embedding with minimal processing overhead
  - Ideal for quick document queries across multiple files
  - Files processed via `tutor_agent_lite.py`

- **Basic Mode**: 
  - Standard RAG (Retrieval-Augmented Generation) 
  - Uses FAISS vector similarity search with 3000-char chunks
  - Comprehensive source attribution and citation tracking
  - Single document processing with deep analysis via `tutor_agent_basic.py`

- **Advanced Mode**: 
  - GraphRAG with Neo4j knowledge graphs
  - Complex relationship mapping between document concepts
  - Enhanced reasoning through graph-based context retrieval
  - Processed via `tutor_agent_advanced.py`

### Technology Stack
- **Frontend**: Streamlit with custom CSS styling
- **AI Services**: OpenAI, Azure OpenAI, SambaNova APIs
- **Vector Databases**: ChromaDB, Qdrant, FAISS
- **Graph Database**: Neo4j for GraphRAG mode
- **Document Processing**: PyMuPDF, unstructured, marker-pdf
- **Authentication**: AWS Cognito
- **Storage**: Azure Blob Storage

## Environment Configuration

Required environment variables in `.env`:
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` - Azure OpenAI
- `SAMBANOVA_API_KEY`, `SAMBANOVA_API_ENDPOINT` - SambaNova API
- `GRAPHRAG_API_KEY`, `GRAPHRAG_LLM_MODEL` - GraphRAG configuration
- `USER_POOL_ID`, `CLIENT_ID` - AWS Cognito authentication
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage
- `ENVIRONMENT` - Deployment environment (local/staging/production)
- `FILE_PATH_PREFIX` - File storage directory

## Development Standards

### Code Style (from `.cursor/rules/development-standards.mdc`)
- **String Conventions**: Use double quotes (`"`) for all strings
- **Type Hints**: Use strict typing throughout - no `any` types or non-null assertions
- **Error Handling**: Implement comprehensive error checking and validation
- **Logging**: Use the centralized logging configuration from `pipeline/science/pipeline/logging_config.py`
- **File Paths**: Use `os.path.join()` for cross-platform compatibility

### Multi-Agent Processing Architecture
The application uses a sophisticated agent routing system:

1. **Agent Selection**: Based on mode selection in `tutor.py:57-65`
2. **Agent Routing**: Through `tutor_agent.py` dispatcher
3. **Mode-Specific Processing**: 
   - `tutor_agent_lite.py` - Handles multiple files with basic embeddings
   - `tutor_agent_basic.py` - Full RAG pipeline with FAISS vector search
   - `tutor_agent_advanced.py` - GraphRAG with Neo4j integration

### Response Generation Pipeline
Key processing steps for each response (especially Basic mode):

1. **File Processing**: `embeddings_agent.py` generates FAISS embeddings
2. **Query Refinement**: `get_query_helper()` processes user input with chat context
3. **Context Retrieval**: `get_rag_context()` retrieves relevant document chunks
4. **Response Generation**: `get_response()` with streaming support and deep thinking
5. **Source Attribution**: `sources_retrieval.py` maps responses to source documents
6. **Translation**: Multi-language support via `content_translator.py`

### File Processing Constraints
- **File Size**: 50MB limit enforced in `tutor.py:79-83` and `tutor.py:161-163`
- **Page Limit**: 800-page limit enforced in `tutor.py:116-118` and `tutor.py:180-182`
- **Storage**: Files stored in `tmp/tutor_pipeline/input_files/` directory
- **Multiple Files**: Supported in Lite mode only (`tutor.py:64-152`)

### Session State Management
All application state managed through Streamlit session state via `frontend/state.py`:
- `uploaded_file` - Current PDF file(s) - single file or list for Lite mode
- `chat_session` - ChatSession object with history and mode tracking
- `mode` - Application mode (Lite/Basic/Advanced)
- `current_language` - User's selected language for responses
- `isAuth` - Authentication status for AWS Cognito integration

### Authentication System
- **AWS Cognito**: Full integration via `frontend/auth.py`
- **Development Mode**: Set `SKIP_AUTH = True` in `frontend/state.py:SKIP_AUTH`
- **Production**: Complete Cognito authentication flow with user pools

## Deployment

### Local Development
```bash
export ENVIRONMENT=local
python -m streamlit run tutor.py
```

### Production Deployment
- **AWS Amplify**: Configuration in `amplify/` directory for cloud deployment
- **Environment Variables**: Different configs for local/staging/production via `ENVIRONMENT` variable
- **Staging Environment**: Full feature testing with `ENVIRONMENT=staging`
- **Production Environment**: Live deployment with `ENVIRONMENT=production`

## Critical Implementation Details

### Vector Database Configuration
- **Embedding Model**: Sentence transformers for consistent embeddings across modes
- **Chunk Strategy**: 3000 characters with 300 character overlap (configurable in `config.json`)
- **Search Parameters**: `k=9` retrieval for similarity search (configurable)
- **Database Options**: FAISS (default), ChromaDB, Qdrant support

### Response Streaming Architecture
The application implements sophisticated streaming for real-time responses:
- **Thinking Phase**: `<thinking>...</thinking>` tags for reasoning display
- **Response Phase**: `<response>...</response>` tags for main content
- **Source Attribution**: `<source>`, `<source_page>`, `<refined_source_page>` tags
- **Follow-up Questions**: `<followup_question>` tags for suggested queries

### Mode-Specific Processing Differences
- **Lite Mode**: Minimal embedding, supports multiple files, fast processing
- **Basic Mode**: Full FAISS vector search, comprehensive source tracking, single file focus
- **Advanced Mode**: GraphRAG with Neo4j, complex relationship mapping, enhanced reasoning

### Error Handling Patterns
- **File Size Validation**: Early rejection of oversized files with user feedback
- **API Fallback**: Graceful degradation when external services fail
- **Authentication Bypass**: Development mode available for local testing
- **Embedding Recovery**: Retry logic for failed embedding generation

## Testing Strategy

### Integration Testing
Tests are organized in `pipeline/science/features_lab/` for component verification:
- **API Integration**: Test external service connections and fallbacks
- **Document Processing**: Verify PDF parsing and embedding generation
- **RAG Pipeline**: End-to-end retrieval and response generation testing
- **Streaming**: Real-time response generation and UI updates