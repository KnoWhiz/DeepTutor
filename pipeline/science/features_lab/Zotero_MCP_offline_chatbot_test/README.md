# 🤖 Offline Zotero Assistant

A completely offline Zotero research assistant that uses semantic search to help you find and discuss papers in your local library.

## ✨ Features

- 🔍 **Semantic Search**: Understands your query intent, even without exact keywords
- 📚 **Local Library**: Works completely offline, protecting your privacy
- 💬 **Smart Conversation**: Natural language interaction, like talking to a real assistant
- 📊 **Relevance Scoring**: Shows relevance scores for search results
- 🎯 **Multi-result Control**: Adjustable number of returned results
- 💡 **Smart Suggestions**: Provides common question suggestions

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Zotero Desktop Application
- Configured Zotero MCP Database

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/offline-zotero-assistant.git
   cd offline-zotero-assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run Application**
   ```bash
   streamlit run simple_test.py
   ```

4. **Access Application**
   Open your browser and navigate to `http://localhost:8501`

## 📋 Usage Guide

### Basic Usage

1. **Start Application**: Run `streamlit run simple_test.py`
2. **Connect Database**: Ensure Zotero MCP database is properly configured
3. **Start Conversation**: Enter your questions in the chat box
4. **View Results**: System will display relevant papers and summaries

### Query Examples

- "Tell me about doppler tomography"
- "What papers do you have on neural networks?"
- "Show me radar signal processing papers"
- "Find papers about machine learning"
- "What's in your library about computer vision?"

### Feature Description

- **Result Count Control**: Select number of returned results (3-20) in sidebar
- **Relevance Scoring**: Each result shows relevance score (0-1)
- **Smart Suggestions**: Sidebar provides common question suggestions

## 🛠️ Technical Architecture

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Semantic Embeddings**: Sentence Transformers
- **Search Algorithm**: Cosine Similarity

## 📁 Project Structure

```
offline-chatbot/
├── simple_test.py          # Main application file
├── requirements_streamlit.txt  # Dependencies file
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
└── ...                    # Other files
```

## 🔧 Configuration

### Zotero MCP Database Configuration

Ensure your Zotero MCP database is located at `~/.config/zotero-mcp/chroma_db` with a collection named `zotero_library`.

### Environment Variables

The application currently uses default configuration. To customize paths, modify the database path in `zotero_offline_chatbot.py`.

## 🤝 Contributing

We welcome Issue submissions and Pull Requests!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For building web applications
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Semantic embedding models
- [Zotero](https://zotero.org/) - Reference management tool

