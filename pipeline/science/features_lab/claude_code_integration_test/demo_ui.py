#!/usr/bin/env python3
"""
Demo UI for Claude Code Integration Chatbot

This demo shows the Streamlit interface without requiring an API key.
Use this to explore the UI before setting up your actual API credentials.

Usage:
    python demo_ui.py
    OR
    streamlit run demo_ui.py
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add the main directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def demo_setup_page_config():
    """Configure Streamlit page settings for demo."""
    st.set_page_config(
        page_title="Claude Code - Codebase Understanding Chatbot (DEMO)",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def demo_sidebar():
    """Demo sidebar with mock configuration."""
    st.sidebar.title("🔧 Configuration (DEMO MODE)")
    st.sidebar.info("This is a demo interface. Add your API key to config.py for full functionality.")
    
    # Mock API Key input
    api_key = st.sidebar.text_input(
        "Anthropic API Key:",
        type="password",
        value="demo_key_placeholder",
        help="This is a demo - enter your real API key in config.py",
        disabled=True
    )
    
    # Mock Directory selection
    test_dir = str(current_dir / "test_files")
    directory_path = st.sidebar.text_input(
        "Codebase Directory:",
        value=test_dir,
        help="Path to directory containing your files",
        disabled=True
    )
    
    # Mock load button
    if st.sidebar.button("🔄 Load/Reload Codebase (DEMO)", type="secondary"):
        st.sidebar.warning("Demo mode: To load real files, set up config.py and use the main application")
    
    # Mock codebase info
    st.sidebar.info("🔍 Demo Codebase Status")
    with st.sidebar.expander("📁 Sample Codebase Overview"):
        st.text("Files that would be loaded:")
        st.text("• paper1.txt (ReAct paper)")
        st.text("• paper2.txt (HiTOP taxonomy)")
        st.text("• (your documents here)")
    
    # Sample queries
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Sample Queries")
    
    sample_queries = [
        "What are the main topics covered in these documents?",
        "Can you summarize the key methodologies described?",
        "What are the relationships between different concepts?",
        "Explain the ReAct framework mentioned in the papers",
        "What are the main findings and conclusions?",
        "How do these documents relate to each other?",
        "What technical approaches are discussed?",
        "Can you identify the main research questions?"
    ]
    
    for i, query in enumerate(sample_queries):
        if st.sidebar.button(f"📝 {query}", key=f"demo_sample_{i}"):
            st.session_state.demo_query = query

def demo_main_interface():
    """Demo main chat interface."""
    st.title("🤖 Claude Code - Codebase Understanding Chatbot (DEMO)")
    st.markdown("""
    **This is a demonstration of the interface.** The actual chatbot uses Claude's advanced 
    understanding capabilities to analyze your codebase (text/markdown files) and provide 
    intelligent responses about the content, concepts, methodologies, and relationships within 
    your documents.
    
    ### 🚀 To Use the Real Application:
    1. **Get API Key**: Sign up at [console.anthropic.com](https://console.anthropic.com/)
    2. **Configure**: Edit `config.py` and add your Anthropic API key
    3. **Run**: Use `python run_chatbot.py` or `streamlit run claude_code_integration_test.py`
    
    ### 📋 Features:
    - **Multi-file Analysis**: Processes entire directories of documents
    - **Contextual Understanding**: Maintains conversation context across exchanges
    - **Deep Analysis**: Provides comprehensive responses with citations
    - **File Type Support**: .txt, .md, .py, .js, .json, .yaml files
    """)
    
    # Initialize demo chat history
    if "demo_messages" not in st.session_state:
        st.session_state.demo_messages = [
            {
                "role": "assistant", 
                "content": """👋 Welcome to the Claude Code Integration Chatbot demo!

I'm designed to analyze your document collections and provide intelligent responses about:
- **Content Analysis**: Understanding key themes and concepts
- **Methodology Extraction**: Identifying research approaches and techniques  
- **Relationship Mapping**: Finding connections between different documents
- **Technical Insights**: Explaining complex concepts and frameworks

**Demo Note**: This interface shows how the real application works. To analyze your actual documents, set up your API key in `config.py` and run the full application.

What would you like to know about the codebase understanding capabilities?"""
            }
        ]
    
    # Display demo chat history
    for message in st.session_state.demo_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle sample query injection
    if "demo_query" in st.session_state:
        demo_query = st.session_state.demo_query
        del st.session_state.demo_query
        
        # Add user message
        st.session_state.demo_messages.append({"role": "user", "content": demo_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(demo_query)
        
        # Generate demo response
        demo_response = generate_demo_response(demo_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(demo_response)
        
        # Add to history
        st.session_state.demo_messages.append({"role": "assistant", "content": demo_response})
        st.rerun()
    
    # Demo chat input
    if prompt := st.chat_input("Try asking about the sample documents... (Demo Mode)"):
        # Add user message to demo history
        st.session_state.demo_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate demo response
        demo_response = generate_demo_response(prompt)
        
        # Display assistant response with typing effect
        with st.chat_message("assistant"):
            st.markdown(demo_response)
        
        # Add assistant response to demo history
        st.session_state.demo_messages.append({"role": "assistant", "content": demo_response})

def generate_demo_response(query: str) -> str:
    """Generate a realistic demo response based on the query."""
    query_lower = query.lower()
    
    if "main topics" in query_lower or "topics covered" in query_lower:
        return """Based on the sample documents in this codebase, the main topics include:

**🧠 Cognitive AI & Reasoning (from ReAct paper)**:
- Language model reasoning and action planning
- Synergistic combination of thought and action
- Interactive decision making in AI systems
- Chain-of-thought prompting techniques

**🧬 Psychological Classification (from HiTOP paper)**:
- Hierarchical taxonomy of psychopathology
- Dimensional approaches to mental health classification
- Alternative frameworks to traditional diagnostic systems
- Statistical modeling of psychological disorders

**🔗 Cross-Document Themes**:
- Systematic approaches to complex problem solving
- Framework development and validation
- Evidence-based methodological improvements
- Integration of theoretical and practical considerations

*Note: This is a demo response. The real application would provide deeper analysis with specific citations from your actual documents.*"""
    
    elif "methodologies" in query_lower or "methods" in query_lower:
        return """The documents demonstrate several key methodologies:

**📊 Research Design Approaches**:
- **Empirical Validation**: Both papers use extensive experimental validation
- **Comparative Analysis**: Systematic comparison with existing baselines
- **Multi-domain Testing**: Evaluation across different problem types

**🔬 Technical Methods**:
- **Prompting Techniques**: Advanced prompt engineering for language models
- **Statistical Modeling**: Factor analysis and hierarchical modeling approaches
- **Evaluation Frameworks**: Comprehensive metrics and benchmark comparisons

**🏗️ Framework Development**:
- **Systematic Integration**: Combining multiple approaches into unified frameworks
- **Iterative Refinement**: Progressive improvement through testing and validation
- **Cross-domain Application**: Testing methods across different application areas

*Demo Note: The actual application would extract these methodologies directly from your documents with specific citations and detailed analysis.*"""
    
    elif "react" in query_lower or "framework" in query_lower:
        return """The ReAct framework represents a significant advancement in AI reasoning:

**🔄 Core Concept**:
ReAct (Reasoning + Acting) interleaves reasoning traces with task-specific actions, allowing language models to:
- **Generate reasoning traces** to induce, track, and update action plans
- **Perform actions** to interface with external information sources
- **Handle exceptions** and adapt plans dynamically

**🎯 Key Innovations**:
- **Synergistic Design**: Reasoning helps with planning, while actions provide additional context
- **Interpretability**: Human-readable reasoning traces improve transparency
- **Robustness**: Better handling of hallucination and error propagation

**📈 Performance Results**:
- HotpotQA and Fever: Overcomes hallucination issues in chain-of-thought reasoning
- ALFWorld and WebShop: 34% and 10% improvement over imitation/RL methods
- Achieved with minimal examples (1-2 in-context demonstrations)

*This demo shows how the real application would extract and explain technical concepts from your documents.*"""
    
    elif "relationships" in query_lower or "connections" in query_lower:
        return """Interesting connections emerge between these seemingly different documents:

**🔗 Methodological Parallels**:
- Both papers propose **hierarchical frameworks** (ReAct's reasoning-action hierarchy vs. HiTOP's psychopathology hierarchy)
- **Dimensional approaches**: ReAct uses continuous reasoning traces; HiTOP uses dimensional rather than categorical classification
- **Integration over separation**: Both argue for combining previously separate elements

**🧩 Problem-Solving Philosophy**:
- **Systematic decomposition**: Breaking complex problems into manageable components
- **Evidence-based validation**: Extensive empirical testing across multiple domains
- **Practical applicability**: Focus on real-world implementation and utility

**📚 Theoretical Contributions**:
- Both challenge existing paradigms (traditional prompting vs. traditional diagnostic categories)
- Propose more nuanced, flexible approaches to their respective domains
- Emphasize the importance of context and interaction in understanding

*The real application would identify many more subtle connections by analyzing the full text of your documents.*"""
    
    else:
        return f"""I understand you're asking about: "{query}"

**🔍 Demo Response**: This is a demonstration of how the Claude Code chatbot would analyze your query in relation to the loaded documents. 

In the full version, I would:
- **Search through all loaded documents** for relevant information
- **Provide specific citations** and quotes from your files
- **Draw connections** between different parts of your codebase
- **Offer detailed analysis** based on the actual content

**📋 To get real analysis**:
1. Set up your Anthropic API key in `config.py`
2. Load your actual documents in the `test_files` directory
3. Run the full application with `python run_chatbot.py`

The real chatbot would understand the nuances of your specific documents and provide comprehensive, cited responses to this exact question.

*Try the sample queries in the sidebar to see more detailed demo responses!*"""

def main():
    """Main demo function."""
    demo_setup_page_config()
    
    # Demo warning
    st.warning("🎭 **DEMO MODE** - This shows the interface only. Set up `config.py` with your API key to analyze real documents.")
    
    demo_sidebar()
    demo_main_interface()
    
    # Instructions at bottom
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready to use the real application?
    
    1. **Get API Key**: Visit [console.anthropic.com](https://console.anthropic.com/) to get your Anthropic API key
    2. **Configure**: Edit `config.py` and replace `"your_anthropic_api_key_here"` with your actual key
    3. **Run**: Execute `python run_chatbot.py` for the full experience
    4. **Analyze**: Load your documents and start asking questions!
    
    **Questions?** Check the `README.md` file for detailed setup instructions.
    """)

if __name__ == "__main__":
    main() 