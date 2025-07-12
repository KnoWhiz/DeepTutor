# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Activate conda environment
conda activate deeptutor

# Run the Streamlit application
python -m streamlit run tutor.py
```

### Environment Setup
```bash
# Create new environment
conda create --name deeptutor python=3.12
conda activate deeptutor

# Install dependencies
pip install -r requirements.txt
pip install -U "autogen-agentchat" "autogen-ext[openai]"
pip install -U "autogenstudio"
```

### Testing
The codebase includes test files in `pipeline/science/features_lab/` directory for various components:
- API testing: `api_test.py`, `api_test.js`
- LLM integration tests: `deepseek_test.py`, `o3mini_test.py`, `sambanova_llama_test.py`
- File processing tests: `file_api_test/` directory
- RAG and GraphRAG tests: `agentic_rag_test.py`, `graphrag_api_test.py`

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
The application operates in three modes:
- **Lite Mode**: Multiple file support with basic embedding
- **Basic Mode**: Standard RAG with vector similarity search
- **Advanced Mode**: GraphRAG with Neo4j knowledge graphs

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

### Code Style
- Use double quotes for strings
- Implement comprehensive error handling
- Use type hints throughout
- Follow patterns in `.cursor/rules/development-standards.mdc`

### File Processing
- 50MB file size limit enforced
- 200-page document limit
- Support for multiple file upload in Lite mode
- Files stored in `tmp/tutor_pipeline/input_files/`

### Session State Management
All application state managed through Streamlit session state:
- `uploaded_file` - Current PDF file(s)
- `chat_session` - Chat history and context
- `mode` - Application mode selection
- `isAuth` - Authentication status

### Authentication
- AWS Cognito integration via `frontend/auth.py`
- Development mode: Set `SKIP_AUTH = True` in `frontend/state.py`
- Production: Full Cognito authentication flow

## Deployment

### Local Development
```bash
export ENVIRONMENT=local
python -m streamlit run tutor.py
```

### Production Deployment
- AWS Amplify configuration in `amplify/` directory
- Staging and production environments supported
- Environment-specific configurations via `ENVIRONMENT` variable