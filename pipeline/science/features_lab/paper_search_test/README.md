# arXiv Paper Search Chatbot 📚

A Streamlit-based chatbot that uses AI to help you find relevant academic papers on arXiv with natural language queries.

## Features 🌟

- **🧠 AI-Powered Query Understanding**: Uses LLM to interpret your natural language queries and generate optimal arXiv search parameters
- **🔍 Smart Search Optimization**: Automatically determines the best search fields, keywords, and filters
- **📊 Top 10 Results**: Returns the most relevant papers based on your query
- **🔗 Direct Access**: Provides direct links to papers and PDF downloads
- **💬 Chat Interface**: Natural conversation flow with search history
- **📋 Detailed View**: Expandable paper cards with abstracts and metadata

## Setup Instructions 🚀

### 1. Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# Azure OpenAI Configuration (Primary)
AZURE_OPENAI_API_KEY_BACKUP=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT_BACKUP=https://your-resource.openai.azure.com/

# SambaNova Configuration (Fallback)
SAMBANOVA_API_KEY=your_sambanova_api_key
```

### 2. Install Dependencies

All required dependencies are already listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit`: Web UI framework
- `arxiv==2.1.3`: arXiv API client
- `langchain`: LLM orchestration
- `langchain-openai`: Azure OpenAI integration
- `langchain-sambanova`: SambaNova integration

### 3. Run the Application

```bash
streamlit run pipeline/science/features_lab/paper_search_test/paper_search_test.py
```

## Usage Examples 💡

### Natural Language Queries

The chatbot understands various types of queries:

**Topic-based searches:**
- "Find papers about machine learning in healthcare"
- "Latest research on transformer architectures"
- "Quantum computing and cryptography papers"

**Author-specific searches:**
- "Papers by Geoffrey Hinton on deep learning"
- "Latest work by Yann LeCun"
- "Research by Andrew Ng on machine learning"

**Time-based searches:**
- "Computer vision papers from 2024"
- "Recent advances in natural language processing"
- "Latest developments in reinforcement learning"

**Category-specific searches:**
- "Papers in computer science AI category"
- "Physics papers on general relativity"
- "Mathematics papers on graph theory"

### How It Works 🔬

1. **Query Understanding**: The LLM analyzes your natural language query
2. **Search Optimization**: Generates optimal arXiv search parameters using:
   - Field prefixes (`ti:`, `au:`, `abs:`, `cat:`, etc.)
   - Boolean operators (`AND`, `OR`, `NOT`)
   - Category filters (e.g., `cs.AI`, `physics.gr-qc`)
3. **arXiv Search**: Queries the arXiv API with optimized parameters
4. **Result Presentation**: Displays top 10 most relevant papers with:
   - Title and authors
   - Publication date and category
   - Abstract (expandable)
   - Direct links to paper and PDF

## arXiv Search Syntax 📖

The chatbot automatically generates queries using arXiv's search syntax:

| Prefix | Field | Example |
|--------|-------|---------|
| `ti:` | Title | `ti:"machine learning"` |
| `au:` | Author | `au:hinton` |
| `abs:` | Abstract | `abs:transformer` |
| `cat:` | Category | `cat:cs.AI` |
| `all:` | All fields | `all:"deep learning"` |

### Popular Categories

- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning  
- `cs.CV` - Computer Vision
- `cs.CL` - Computation and Language
- `physics.gr-qc` - General Relativity
- `math.CO` - Combinatorics
- `q-bio.QM` - Quantitative Methods

## Architecture 🏗️

```
User Query → LLM Analysis → Search Parameters → arXiv API → Results Display
```

### Components

- **ArxivSearchAgent**: Main class handling LLM integration and arXiv search
- **LLM Integration**: Supports Azure OpenAI (primary) and SambaNova (fallback)
- **Search Optimization**: Converts natural language to arXiv query syntax
- **Result Processing**: Formats and displays search results
- **Streamlit UI**: Chat interface with sidebar controls

## Customization 🛠️

### Model Selection

The chatbot automatically selects the best available LLM:
1. **Azure OpenAI** (gpt-4o-mini) - Primary choice
2. **SambaNova** (Meta-Llama-3.3-70B-Instruct) - Fallback

### UI Settings

- **Detailed View**: Toggle paper card display
- **Chat History**: Persistent conversation history
- **Example Queries**: Quick-start buttons in sidebar

## Troubleshooting 🔧

### Common Issues

1. **No LLM Access**: Ensure API keys are correctly set in `.env`
2. **No Results**: Try broader search terms or different keywords
3. **Slow Response**: Large result sets may take time to process

### Error Handling

- **LLM Failure**: Falls back to basic search without AI optimization
- **arXiv API Issues**: Displays helpful error messages
- **Network Problems**: Graceful degradation with fallback options

## Contributing 🤝

Feel free to submit issues and pull requests to improve the chatbot's functionality and user experience.

## License 📄

This project follows the same license as the parent DeepTutor repository. 