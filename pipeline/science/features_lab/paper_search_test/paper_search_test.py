import os
import sys
import json
import logging
import streamlit as st
import arxiv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to sys.path to import from pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_sambanova import ChatSambaNovaCloud

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="arXiv Paper Search Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ArxivSearchAgent:
    """Agent for searching arXiv papers using LLM-powered query understanding."""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.search_prompt = self._create_search_prompt()
        
    def _initialize_llm(self):
        """Initialize the LLM for query understanding."""
        # Try to use Azure OpenAI first, fallback to SambaNova
        try:
            if os.getenv("AZURE_OPENAI_API_KEY_BACKUP") and os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"):
                return AzureChatOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
                    api_version="2024-07-01-preview",
                    deployment_name="gpt-4o-mini",
                    temperature=0.1,
                    streaming=False
                )
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI: {e}")
        
        # Fallback to SambaNova
        try:
            if os.getenv("SAMBANOVA_API_KEY"):
                return ChatSambaNovaCloud(
                    model="Meta-Llama-3.3-70B-Instruct",
                    api_key=os.getenv("SAMBANOVA_API_KEY"),
                    base_url="https://api.sambanova.ai/v1",
                    max_tokens=1024,
                    temperature=0.1,
                    top_p=0.01
                )
        except Exception as e:
            logger.error(f"Failed to initialize SambaNova: {e}")
            st.error("❌ Failed to initialize LLM. Please check your API keys.")
            return None
    
    def _create_search_prompt(self):
        """Create the prompt template for understanding user queries."""
        system_prompt = """You are an expert academic research assistant that helps users find relevant papers on arXiv.

Your task is to analyze the user's query and generate appropriate arXiv search parameters.

Based on the user's input, you should:
1. Identify the main research topics, concepts, or keywords
2. Determine if they're looking for specific authors, time periods, or subject areas
3. Generate effective search terms for the arXiv API

arXiv search field prefixes:
- ti: Title
- au: Author  
- abs: Abstract
- co: Comment
- jr: Journal Reference
- cat: Subject Category (e.g., cs.AI, physics.gr-qc, math.CO)
- all: All fields (default)

You can use Boolean operators: AND, OR, NOT
You can use quotes for exact phrases: "machine learning"

Respond ONLY with a JSON object containing:
{
    "search_query": "optimized search query using arXiv syntax",
    "max_results": 10,
    "sort_by": "relevance or submittedDate",
    "explanation": "brief explanation of your search strategy"
}

Examples:
- User: "Find papers about machine learning in healthcare"
  Response: {"search_query": "all:\"machine learning\" AND all:healthcare", "max_results": 10, "sort_by": "relevance", "explanation": "Searching for papers mentioning both machine learning and healthcare in any field"}

- User: "Latest papers by Geoffrey Hinton on deep learning"
  Response: {"search_query": "au:hinton AND all:\"deep learning\"", "max_results": 10, "sort_by": "submittedDate", "explanation": "Searching for recent papers by Hinton related to deep learning"}

- User: "Computer vision papers from 2024"
  Response: {"search_query": "cat:cs.CV", "max_results": 10, "sort_by": "submittedDate", "explanation": "Searching computer vision category papers, sorted by submission date to get recent ones"}
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_query}")
        ])
    
    def understand_query(self, user_query: str) -> Dict[str, Any]:
        """Use LLM to understand user query and generate search parameters."""
        if not self.llm:
            return {
                "search_query": user_query,
                "max_results": 10,
                "sort_by": "relevance",
                "explanation": "Using basic search without LLM analysis"
            }
        
        try:
            # Create the chain
            chain = self.search_prompt | self.llm | StrOutputParser()
            
            # Get LLM response
            response = chain.invoke({"user_query": user_query})
            
            # Parse JSON response
            search_params = json.loads(response.strip())
            
            # Validate required fields
            if "search_query" not in search_params:
                search_params["search_query"] = user_query
            if "max_results" not in search_params:
                search_params["max_results"] = 10
            if "sort_by" not in search_params:
                search_params["sort_by"] = "relevance"
            if "explanation" not in search_params:
                search_params["explanation"] = "LLM-generated search strategy"
                
            return search_params
            
        except Exception as e:
            logger.error(f"Error in query understanding: {e}")
            # Fallback to basic search
            return {
                "search_query": user_query,
                "max_results": 10,
                "sort_by": "relevance",
                "explanation": f"Fallback search due to error: {str(e)}"
            }
    
    def search_arxiv(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search arXiv using the generated parameters."""
        try:
            # Configure arXiv client
            client = arxiv.Client()
            
            # Determine sort criteria
            sort_criteria = arxiv.SortCriterion.Relevance
            if search_params.get("sort_by") == "submittedDate":
                sort_criteria = arxiv.SortCriterion.SubmittedDate
            
            # Create search
            search = arxiv.Search(
                query=search_params["search_query"],
                max_results=search_params.get("max_results", 10),
                sort_by=sort_criteria,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # Execute search and collect results
            papers = []
            for paper in client.results(search):
                papers.append({
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "summary": paper.summary,
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
                    "categories": paper.categories,
                    "primary_category": paper.primary_category
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            st.error(f"❌ Error searching arXiv: {str(e)}")
            return []


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_agent" not in st.session_state:
        st.session_state.search_agent = ArxivSearchAgent()


def display_paper_card(paper: Dict[str, Any], index: int):
    """Display a single paper in a card format."""
    with st.container():
        st.markdown(f"### 📄 {index}. {paper['title']}")
        
        # Authors
        authors_str = ", ".join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += f" et al. ({len(paper['authors'])} authors)"
        st.markdown(f"**👥 Authors:** {authors_str}")
        
        # Categories and dates
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**📂 Category:** {paper['primary_category']}")
        with col2:
            st.markdown(f"**📅 Published:** {paper['published']}")
        with col3:
            if paper['updated']:
                st.markdown(f"**🔄 Updated:** {paper['updated']}")
        
        # Summary
        with st.expander("📖 Abstract", expanded=False):
            st.write(paper['summary'])
        
        # Links
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"[📄 View Paper]({paper['entry_id']})")
        with col2:
            st.markdown(f"[📥 Download PDF]({paper['pdf_url']})")
        
        st.markdown("---")


def format_search_results(papers: List[Dict[str, Any]], search_params: Dict[str, Any]) -> str:
    """Format search results as a readable response."""
    if not papers:
        return "❌ No papers found matching your query. Try different keywords or broader search terms."
    
    response = f"🔍 **Search Strategy:** {search_params.get('explanation', 'Generated search query')}\n\n"
    response += f"📚 **Found {len(papers)} papers** (showing top {len(papers)}):\n\n"
    
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper['authors'][:2])
        if len(paper['authors']) > 2:
            authors_str += " et al."
        response += f"**{i}. {paper['title']}**\n"
        response += f"   👥 {authors_str} | 📅 {paper['published']} | 📂 {paper['primary_category']}\n"
        response += f"   🔗 [View Paper]({paper['entry_id']}) | [Download PDF]({paper['pdf_url']})\n\n"
    
    return response


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar with information
    with st.sidebar:
        st.title("📚 arXiv Search Bot")
        st.markdown("---")
        st.markdown("### 🤖 About")
        st.markdown("""
        This chatbot helps you find relevant academic papers on arXiv using natural language queries.
        
        **Features:**
        - 🧠 AI-powered query understanding
        - 🔍 Smart arXiv search optimization  
        - 📊 Top 10 most relevant results
        - 🔗 Direct links to papers and PDFs
        """)
        
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "Latest papers on transformer architectures",
            "Machine learning in climate science",
            "Papers by Yann LeCun on deep learning",
            "Computer vision papers from 2024", 
            "Quantum computing and cryptography",
            "Natural language processing for healthcare"
        ]
        
        for query in example_queries:
            if st.button(f"💭 {query}", key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.example_query = query
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_detailed_view = st.checkbox("Show detailed paper cards", value=True)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("📚 arXiv Paper Search Chatbot")
    st.markdown("Ask me to find academic papers on any topic! I'll search arXiv and provide you with the most relevant results.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "papers" in message:
                # Display formatted response
                st.markdown(message["content"])
                
                # Show detailed paper cards if enabled
                if show_detailed_view and message["papers"]:
                    st.markdown("### 📋 Detailed Results")
                    for i, paper in enumerate(message["papers"], 1):
                        display_paper_card(paper, i)
            else:
                st.markdown(message["content"])
    
    # Handle example query from sidebar
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        
        # Add to messages and process
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process the query
        process_user_query(query, show_detailed_view)
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me to find papers on any topic..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query
        process_user_query(prompt, show_detailed_view)


def process_user_query(prompt: str, show_detailed_view: bool = True):
    """Process user query and search for papers."""
    with st.chat_message("assistant"):
        with st.spinner("🤔 Understanding your query..."):
            # Step 1: Understand the query using LLM
            search_params = st.session_state.search_agent.understand_query(prompt)
            
            st.markdown(f"🧠 **Search Strategy:** {search_params['explanation']}")
            st.markdown(f"🔍 **Query:** `{search_params['search_query']}`")
        
        with st.spinner("📡 Searching arXiv..."):
            # Step 2: Search arXiv
            papers = st.session_state.search_agent.search_arxiv(search_params)
        
        if papers:
            # Format and display results
            response_text = format_search_results(papers, search_params)
            st.markdown(response_text)
            
            # Show detailed view if enabled
            if show_detailed_view:
                st.markdown("### 📋 Detailed Results")
                for i, paper in enumerate(papers, 1):
                    display_paper_card(paper, i)
            
            # Add to message history with papers data
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "papers": papers,
                "search_params": search_params
            })
        else:
            error_msg = "❌ No papers found. Try refining your search with different keywords."
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
