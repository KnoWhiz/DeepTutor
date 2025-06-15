"""
ArXiv Paper Search Chatbot

This module implements a Streamlit-based chatbot that searches for academic papers
using the arXiv API and returns the top 10 most relevant results with URLs.
"""

import streamlit as st
import arxiv
from typing import List, Dict, Optional
import time
from datetime import datetime


class ArxivSearchBot:
    """
    A chatbot class that handles arXiv paper searches and manages conversation state.
    
    This class provides methods to search for papers, format results, and maintain
    chat history in the Streamlit application.
    """
    
    def __init__(self) -> None:
        """Initialize the ArxivSearchBot with default configuration."""
        self.max_results: int = 10
        self.client = arxiv.Client()
    
    def search_papers(self, query: str) -> List[Dict[str, str]]:
        """
        Search for papers using the arXiv API.
        
        Args:
            query: The search query string
            
        Returns:
            List of dictionaries containing paper information
            
        Raises:
            Exception: If the arXiv API request fails
        """
        try:
            # Create search object with the query
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Execute search and collect results
            papers: List[Dict[str, str]] = []
            for result in self.client.results(search):
                paper_info = {
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "summary": result.summary,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "categories": ", ".join(result.categories)
                }
                papers.append(paper_info)
            
            return papers
            
        except Exception as e:
            st.error(f"Error searching arXiv: {str(e)}")
            return []
    
    def format_paper_response(self, papers: List[Dict[str, str]], query: str) -> str:
        """
        Format the search results into a readable response.
        
        Args:
            papers: List of paper dictionaries
            query: The original search query
            
        Returns:
            Formatted string containing paper information
        """
        if not papers:
            return f"I couldn't find any papers related to '{query}'. Please try a different search term."
        
        response = f"I found {len(papers)} papers related to '{query}':\n\n"
        
        for i, paper in enumerate(papers, 1):
            response += f"**{i}. {paper['title']}**\n"
            response += f"👥 Authors: {paper['authors']}\n"
            response += f"📅 Published: {paper['published']}\n"
            response += f"🏷️ Categories: {paper['categories']}\n"
            response += f"🔗 URL: {paper['url']}\n"
            response += f"📄 PDF: {paper['pdf_url']}\n"
            response += f"📝 Summary: {paper['summary'][:200]}...\n\n"
            response += "---\n\n"
        
        return response


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        st.session_state.bot = ArxivSearchBot()


def display_chat_history() -> None:
    """Display the chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_message(role: str, content: str) -> None:
    """
    Add a message to the chat history.
    
    Args:
        role: The role of the message sender ("user" or "assistant")
        content: The message content
    """
    st.session_state.messages.append({"role": role, "content": content})


def main() -> None:
    """
    Main function that runs the Streamlit application.
    
    This function sets up the UI, handles user input, and manages the chat flow.
    """
    # Configure page
    st.set_page_config(
        page_title="ArXiv Paper Search Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("📚 ArXiv Paper Search Chatbot")
    st.markdown("---")
    st.markdown("Ask me to search for academic papers on any topic!")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Enter your search query in the chat input
        2. I'll search arXiv for the most relevant papers
        3. You'll get the top 10 papers with:
           - Title and authors
           - Publication date
           - Categories
           - Abstract summary
           - Direct links to paper and PDF
        """)
        
        st.header("💡 Example Queries")
        st.markdown("""
        - "machine learning neural networks"
        - "quantum computing algorithms"
        - "climate change modeling"
        - "CRISPR gene editing"
        - "dark matter detection"
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask me to search for papers on any topic..."):
        # Add user message to chat history
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching arXiv for relevant papers..."):
                # Search for papers
                papers = st.session_state.bot.search_papers(prompt)
                
                # Format response
                response = st.session_state.bot.format_paper_response(papers, prompt)
                
                # Display response
                st.markdown(response)
                
                # Add assistant message to chat history
                add_message("assistant", response)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This chatbot searches the arXiv preprint repository. "
        "Results are sorted by relevance and limited to the top 10 papers."
    )


if __name__ == "__main__":
    main()
