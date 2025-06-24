import streamlit as st
import arxiv
from typing import List, Dict
import time

def search_arxiv_papers(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search arXiv for papers based on the query and return top results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of dictionaries containing paper information
    """
    try:
        # Create arXiv client and search
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper_info = {
                "title": result.title,
                "authors": ", ".join([author.name for author in result.authors]),
                "summary": result.summary[:300] + "..." if len(result.summary) > 300 else result.summary,
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

def display_paper_results(papers: List[Dict[str, str]]) -> None:
    """
    Display the search results in a formatted way.
    
    Args:
        papers: List of paper dictionaries to display
    """
    if not papers:
        st.warning("No papers found for your query.")
        return
    
    st.success(f"Found {len(papers)} papers:")
    
    for i, paper in enumerate(papers, 1):
        with st.container():
            st.markdown(f"### {i}. {paper['title']}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Authors:** {paper['authors']}")
                st.markdown(f"**Published:** {paper['published']}")
                st.markdown(f"**Categories:** {paper['categories']}")
                st.markdown(f"**Summary:** {paper['summary']}")
            
            with col2:
                st.markdown("**Links:**")
                st.markdown(f"[📄 Abstract]({paper['url']})")
                st.markdown(f"[📥 PDF]({paper['pdf_url']})")
            
            st.divider()

def main():
    """
    Main Streamlit application for arXiv paper search chatbot.
    """
    # Page configuration
    st.set_page_config(
        page_title="arXiv Paper Search Chatbot",
        page_icon="🔬",
        layout="wide"
    )
    
    # Title and description
    st.title("🔬 arXiv Paper Search Chatbot")
    st.markdown("Ask me to search for academic papers on arXiv! I'll find the top 10 most relevant papers for your query.")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your arXiv paper search assistant. What topic would you like me to search for?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your search query (e.g., 'machine learning transformers', 'quantum computing', 'neural networks')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching arXiv for relevant papers..."):
                # Search for papers
                papers = search_arxiv_papers(prompt)
                
                if papers:
                    response_text = f"I found {len(papers)} papers related to '{prompt}'. Here are the results:"
                    st.markdown(response_text)
                    
                    # Display the results
                    display_paper_results(papers)
                    
                    # Add response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"{response_text}\n\n*[Papers displayed above]*"
                    })
                else:
                    error_message = f"I couldn't find any papers for '{prompt}'. Please try a different search query."
                    st.markdown(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📖 How to Use")
        st.markdown("""
        1. **Enter your query** in the chat input below
        2. **Wait for results** - I'll search arXiv for relevant papers
        3. **Browse results** - Each paper includes:
           - Title and authors
           - Publication date
           - Categories
           - Summary
           - Links to abstract and PDF
        
        **Example queries:**
        - "machine learning"
        - "quantum computing algorithms"
        - "neural networks for NLP"
        - "computer vision transformers"
        - "reinforcement learning robotics"
        """)
        
        st.header("🔧 Features")
        st.markdown("""
        - **Real-time search** using arXiv API
        - **Top 10 results** sorted by relevance
        - **Direct links** to papers and PDFs
        - **Chat interface** with history
        - **Responsive design** for easy browsing
        """)
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chat history cleared! What would you like to search for?"
            })
            st.rerun()

if __name__ == "__main__":
    main()
