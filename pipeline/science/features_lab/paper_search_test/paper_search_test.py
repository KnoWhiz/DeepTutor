import streamlit as st
import arxiv
from typing import List, Dict
import time
import re
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def refine_query_with_llm(user_query: str) -> str:
    """
    Use GPT-4 to refine a user's natural language query into an optimized arXiv search query.
    
    Args:
        user_query: The user's natural language search query
        
    Returns:
        A refined query optimized for arXiv API search
    """
    try:
        # Initialize the GPT-4 model
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
            openai_api_version='2024-06-01',
            azure_deployment='gpt-4o',
            temperature=0.3,
            streaming=False,
        )
        
        # Create a prompt template for query refinement
        prompt_template = """You are an expert at optimizing search queries for the arXiv API. 
The arXiv API supports the following search prefixes:
- ti: Title search
- au: Author search  
- abs: Abstract search
- cat: Subject Category search
- all: Search all fields
- AND/OR/NOT: Boolean operators
- Quotes for exact phrases (use ONLY for multi-word phrases within a single field)

IMPORTANT GUIDELINES:
1. Create queries that are BROAD ENOUGH to return results - avoid being too restrictive
2. Prefer OR over AND when searching for related terms
3. Use 'all:' or 'abs:' for general topic searches instead of restricting to titles
4. NEVER wrap entire boolean expressions in quotes - quotes are ONLY for exact phrases
5. For author names, be flexible (e.g., use last name only or partial names)
6. Account for potential misspellings or variations in terminology
7. If multiple concepts are mentioned, consider which are ESSENTIAL vs optional
8. DO NOT use quotes around the entire query or around OR/AND expressions

Examples of GOOD refinements:
- User: "machine learning transformers" → all:machine learning OR all:transformer
- User: "papers by John Smith on quantum" → au:Smith AND all:quantum
- User: "neural networks for NLP" → all:"neural networks" AND (all:NLP OR all:"natural language")
- User: "time multiplexing trapped ion quantum network by B You" → (all:"time multiplexing" OR all:"trapped ion" OR all:"quantum network") AND au:You
- User: "deep learning computer vision" → all:"deep learning" OR all:"computer vision"

Examples of BAD refinements (DO NOT DO THIS):
- "all:machine learning OR all:transformer" (quotes around entire query)
- ti:"quantum" AND ti:"computing" AND ti:"algorithms" (too restrictive)

Transform the following user query into an optimized arXiv search query that balances precision with recall.
Return ONLY the query string without any quotes around the entire expression.

User query: {query}

Optimized arXiv query:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        parser = StrOutputParser()
        
        # Create the chain
        chain = prompt | llm | parser
        
        # Get the refined query
        refined_query = chain.invoke({"query": user_query})
        refined_query = refined_query.strip()
        
        # Clean up the refined query - remove any surrounding quotes
        if refined_query.startswith('"') and refined_query.endswith('"'):
            refined_query = refined_query[1:-1]
        
        # Log the refinement for debugging
        if refined_query != user_query:
            st.info(f"🔍 Refined search query: `{refined_query}`")
        
        return refined_query
        
    except Exception as e:
        st.warning(f"Query refinement failed, using original query. Error: {str(e)}")
        return user_query

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
                # Refine the query using LLM
                refined_query = refine_query_with_llm(prompt)
                
                # Search for papers using the refined query
                papers = search_arxiv_papers(refined_query)
                
                # If no results, try a simpler fallback search
                if not papers and refined_query != prompt:
                    st.warning("No results found with refined query. Trying a broader search...")
                    # Try the original query
                    papers = search_arxiv_papers(prompt)
                    
                    # If still no results, try an even simpler all-fields search
                    if not papers:
                        # Extract key terms and search in all fields
                        # Remove common words and create a simple OR query
                        # Extract meaningful words (alphanumeric, longer than 2 chars)
                        words = re.findall(r'\b\w{3,}\b', prompt.lower())
                        # Remove common words
                        stop_words = {'the', 'for', 'and', 'with', 'from', 'about', 'paper', 'papers', 'article', 'articles'}
                        keywords = [w for w in words if w not in stop_words]
                        
                        if keywords:
                            # Create a simple OR query with the main keywords
                            simple_query = " OR ".join([f"all:{keyword}" for keyword in keywords[:5]])  # Limit to 5 keywords
                            st.info(f"Trying broader search: `{simple_query}`")
                            papers = search_arxiv_papers(simple_query)
                
                if papers:
                    response_text = f"I found {len(papers)} papers related to your search. Here are the results:"
                    st.markdown(response_text)
                    
                    # Display the results
                    display_paper_results(papers)
                    
                    # Add response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"{response_text}\n\n*[Papers displayed above]*"
                    })
                else:
                    error_message = f"I couldn't find any papers for your query. Please try a different search query or use more general terms."
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
