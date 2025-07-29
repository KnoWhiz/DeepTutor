import streamlit as st
import asyncio
from gpt_researcher import GPTResearcher
import time
from typing import Optional
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key by making a simple test call.
    
    Args:
        api_key: OpenAI API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        # Make a simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        st.error(f"❌ OpenAI API key validation failed: {str(e)}")
        return False

def validate_tavily_api_key(api_key: str) -> bool:
    """
    Validate Tavily API key by checking format.
    
    Args:
        api_key: Tavily API key to validate
        
    Returns:
        True if format looks valid, False otherwise
    """
    # Tavily API keys typically start with 'tvly-'
    if not api_key.startswith('tvly-'):
        st.error("❌ Tavily API key should start with 'tvly-'")
        return False
    return True

def test_gpt_researcher_installation():
    """
    Test if GPT Researcher package is properly installed and accessible.
    """
    try:
        import gpt_researcher
        st.success(f"✅ GPT Researcher package imported successfully. Version: {gpt_researcher.__version__ if hasattr(gpt_researcher, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        st.error(f"❌ Failed to import GPT Researcher: {e}")
        st.error("Please install the package with: pip install gpt-researcher")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error importing GPT Researcher: {e}")
        return False

async def conduct_research(query: str, openai_api_key: str = None, tavily_api_key: str = None) -> tuple[Optional[str], Optional[str]]:
    """
    Conduct research using GPT Researcher package.
    
    Args:
        query: Research query string
        openai_api_key: OpenAI API key
        tavily_api_key: Tavily API key
        
    Returns:
        Tuple of (research_result, report) or (None, None) if error
    """
    try:
        # Set environment variables if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.info(f"✅ OpenAI API Key set (length: {len(openai_api_key)})")
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            st.info(f"✅ Tavily API Key set (length: {len(tavily_api_key)})")
        
        # Debug: Check environment variables
        st.info(f"🔍 Environment check - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        st.info(f"🔍 Environment check - TAVILY_API_KEY: {'Set' if os.getenv('TAVILY_API_KEY') else 'Not set'}")
        
        # Create GPT Researcher instance with explicit configuration
        st.info("🔄 Creating GPT Researcher instance...")
        researcher = GPTResearcher(query=query)
        
        st.info("🔄 Starting research process...")
        # Conduct research on the given query
        research_result = await researcher.conduct_research()
        st.info(f"✅ Research completed. Result type: {type(research_result)}")
        
        st.info("🔄 Writing report...")
        # Write the report
        report = await researcher.write_report()
        st.info(f"✅ Report completed. Report type: {type(report)}")
        
        return research_result, report
    
    except Exception as e:
        st.error(f"❌ Error conducting research: {str(e)}")
        st.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        st.error(f"❌ Full traceback: {traceback.format_exc()}")
        return None, None

def display_research_results(research_result: str, report: str, query: str) -> None:
    """
    Display the research results in a formatted way.
    
    Args:
        research_result: The research result from conduct_research()
        report: The final report from write_report()
        query: The original query
    """
    if not report:
        st.warning("No research report generated for your query.")
        return
    
    st.success(f"Research completed for: '{query}'")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Final Report", "🔍 Research Process"])
    
    with tab1:
        st.markdown("### 📋 Research Report")
        # Display the final report
        st.markdown(report)
    
    with tab2:
        st.markdown("### 🔍 Research Process Details")
        if research_result:
            st.markdown("**Research Process Information:**")
            st.text(str(research_result))
        else:
            st.info("Research process details not available.")

def run_async_research(query: str, openai_api_key: str = None, tavily_api_key: str = None) -> tuple[Optional[str], Optional[str]]:
    """
    Wrapper to run async research function in Streamlit.
    
    Args:
        query: Research query string
        openai_api_key: OpenAI API key
        tavily_api_key: Tavily API key
        
    Returns:
        Tuple of (research_result, report)
    """
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(conduct_research(query, openai_api_key, tavily_api_key))
        loop.close()
        return result
    except Exception as e:
        st.error(f"Error running research: {str(e)}")
        return None, None

def main():
    """
    Main Streamlit application for GPT Researcher deep research.
    """
    # Page configuration
    st.set_page_config(
        page_title="GPT Researcher - Deep Research",
        page_icon="🔎",
        layout="wide"
    )
    
    # Title and description
    st.title("🔎 GPT Researcher - Deep Research Platform")
    st.markdown("Ask me to conduct deep research on any topic! I'll use GPT Researcher to generate detailed, factual, and unbiased research reports with citations.")
    
    # Test GPT Researcher installation
    st.subheader("🔧 System Check")
    if not test_gpt_researcher_installation():
        st.error("❌ GPT Researcher package is not properly installed. Please install it first.")
        st.stop()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your GPT Researcher assistant. What topic would you like me to research deeply? I can generate detailed reports exceeding 2,000 words with citations from multiple sources."
        })
    
    # Sidebar with API key configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Initialize API keys as None
        openai_api_key = None
        tavily_api_key = None
        
        # Manual API key input
        st.subheader("Manual API Key Input")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        tavily_api_key = st.text_input(
            "Tavily API Key",
            type="password",
            help="Enter your Tavily API key"
        )
        
        # Use only manual input keys
        final_openai_key = openai_api_key if openai_api_key and openai_api_key.strip() else None
        final_tavily_key = tavily_api_key if tavily_api_key and tavily_api_key.strip() else None
        
        # Check if we have the required keys
        if not final_openai_key:
            st.error("❌ OpenAI API Key is required!")
        else:
            # Validate OpenAI API key
            if st.button("🔍 Validate OpenAI API Key"):
                with st.spinner("Validating OpenAI API key..."):
                    if validate_openai_api_key(final_openai_key):
                        st.success("✅ OpenAI API key is valid!")
                    else:
                        st.error("❌ OpenAI API key is invalid. Please check your key.")
        
        if not final_tavily_key:
            st.error("❌ Tavily API Key is required!")
        else:
            # Validate Tavily API key
            if st.button("🔍 Validate Tavily API Key"):
                with st.spinner("Validating Tavily API key..."):
                    if validate_tavily_api_key(final_tavily_key):
                        st.success("✅ Tavily API key format looks valid!")
                    else:
                        st.error("❌ Tavily API key format is invalid.")
        
        # Test API keys button
        if final_openai_key and final_tavily_key:
            if st.button("🧪 Test API Keys"):
                with st.spinner("Testing API keys with a simple query..."):
                    try:
                        # First validate the keys
                        if not validate_openai_api_key(final_openai_key):
                            st.error("❌ OpenAI API key validation failed. Cannot proceed with test.")
                            return
                        
                        if not validate_tavily_api_key(final_tavily_key):
                            st.error("❌ Tavily API key validation failed. Cannot proceed with test.")
                            return
                        
                        test_result, test_report = run_async_research(
                            "What is artificial intelligence?", 
                            final_openai_key, 
                            final_tavily_key
                        )
                        if test_report:
                            st.success("✅ API keys are working! Test research completed successfully.")
                        else:
                            st.error("❌ API keys test failed. Check the error messages above.")
                    except Exception as e:
                        st.error(f"❌ API keys test failed: {str(e)}")
        
        st.divider()
        
        st.header("📖 How to Use GPT Researcher")
        st.markdown("""
        1. **Enter your research query** in the chat input below
        2. **Wait for research** - GPT Researcher will:
           - Generate research questions
           - Gather information from 20+ sources
           - Analyze and synthesize findings
           - Create a comprehensive report
        3. **Review results** - Each research includes:
           - Detailed research report (2000+ words)
           - Citations and sources
           - Objective conclusions
           - Research process details
        
        **Example queries:**
        - "Why is Nvidia stock going up?"
        - "Impact of artificial intelligence on healthcare"
        - "Climate change solutions and technologies"
        - "Future of renewable energy"
        - "Cryptocurrency market trends 2024"
        """)
        
        st.header("🚀 GPT Researcher Features")
        st.markdown("""
        - **📝 Detailed Reports**: Generate reports exceeding 2,000 words
        - **🌐 Multiple Sources**: Aggregate over 20 sources for objective conclusions
        - **🔍 Smart Research**: JavaScript-enabled web scraping
        - **📄 Export Ready**: Reports ready for PDF, Word, and other formats
        - **🎯 Objective**: Reduce bias through multiple source aggregation
        - **⚡ Efficient**: Parallelized agent work for faster results
        """)
        
        st.header("⚙️ Configuration")
        st.markdown("""
        **Required API Keys:**
        - OpenAI API Key
        - Tavily API Key (for web search)
        
        **Optional Features:**
        - Deep Research (tree-like exploration)
        - MCP Integration (specialized data sources)
        - Local Document Research
        """)
        
        # Help section for API keys
        with st.expander("🔑 How to Get API Keys"):
            st.markdown("""
            **OpenAI API Key:**
            1. Go to https://platform.openai.com/account/api-keys
            2. Sign in or create an account
            3. Click "Create new secret key"
            4. Copy the key (starts with 'sk-')
            5. Add credits to your account
            
            **Tavily API Key:**
            1. Go to https://tavily.com/
            2. Sign up for a free account
            3. Get your API key from dashboard
            4. Copy the key (starts with 'tvly-')
            
            **Important Notes:**
            - OpenAI keys start with 'sk-'
            - Tavily keys start with 'tvly-'
            - Keep your keys secure and don't share them
            - Free tiers have usage limits
            """)
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chat history cleared! What topic would you like me to research?"
            })
            st.rerun()
        
        # Research tips
        st.header("💡 Research Tips")
        st.markdown("""
        - **Be specific**: More specific queries yield better results
        - **Use keywords**: Include relevant technical terms
        - **Ask questions**: Frame as research questions
        - **Be patient**: Deep research takes 3-5 minutes
        - **Review sources**: Check citations in the report
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "research_data" in message:
                # Display research results if they exist
                research_result, report = message["research_data"]
                if report:
                    display_research_results(research_result, report, message.get("query", ""))
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your research query (e.g., 'why is Nvidia stock going up?', 'impact of AI on healthcare', 'climate change solutions')"):
        # Check if API keys are available
        if not final_openai_key or not final_tavily_key:
            st.error("Please configure your API keys in the sidebar before conducting research.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Conducting deep research... This may take several minutes as I gather information from multiple sources..."):
                # Conduct research with API keys
                research_result, report = run_async_research(prompt, final_openai_key, final_tavily_key)
                
                if report:
                    response_text = f"I've completed deep research on '{prompt}'. Here's the comprehensive report:"
                    st.markdown(response_text)
                    
                    # Display the results
                    display_research_results(research_result, report, prompt)
                    
                    # Add response to chat history with research data
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"{response_text}\n\n*[Research report displayed above]*",
                        "research_data": (research_result, report),
                        "query": prompt
                    })
                else:
                    error_message = f"I couldn't complete the research for '{prompt}'. Please check your API keys and try again with a different query."
                    st.markdown(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

if __name__ == "__main__":
    main()
