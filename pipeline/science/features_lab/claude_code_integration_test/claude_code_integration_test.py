"""
Claude Code Integration Test - Codebase Understanding Chatbot

This module implements a Streamlit-based chatbot that uses Claude's codebase understanding
capabilities to analyze and respond to queries about a collection of text and markdown files.
The files are treated as a unified codebase for comprehensive analysis.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import anthropic
from anthropic import Anthropic
import json
import time
from datetime import datetime

# Add the pipeline directory to Python path for imports
current_dir = Path(__file__).parent
pipeline_dir = current_dir.parent.parent.parent / "pipeline"
sys.path.append(str(pipeline_dir))

try:
    from config import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = None


class CodebaseAnalyzer:
    """
    Handles file loading and codebase analysis using Claude's understanding capabilities.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the CodebaseAnalyzer with Anthropic API key.
        
        Args:
            api_key: Anthropic API key for Claude access
        """
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = Anthropic(api_key=api_key)
        self.codebase_files: Dict[str, str] = {}
        self.codebase_summary: str = ""
        self.max_tokens_per_request = 100000  # Claude's context limit consideration
        
    def load_files_from_directory(self, directory_path: str) -> bool:
        """
        Load all text and markdown files from the specified directory.
        
        Args:
            directory_path: Path to the directory containing files
            
        Returns:
            bool: True if files were loaded successfully, False otherwise
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                st.error(f"Directory {directory_path} does not exist")
                return False
            
            supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml'}
            loaded_files = 0
            
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            relative_path = str(file_path.relative_to(directory))
                            self.codebase_files[relative_path] = content
                            loaded_files += 1
                    except Exception as e:
                        st.warning(f"Failed to load {file_path}: {str(e)}")
            
            st.success(f"Loaded {loaded_files} files from {directory_path}")
            return loaded_files > 0
            
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            return False
    
    def generate_codebase_summary(self) -> str:
        """
        Generate a comprehensive summary of the loaded codebase.
        
        Returns:
            str: Summary of the codebase structure and content
        """
        if not self.codebase_files:
            return "No files loaded in codebase."
        
        # Create file structure overview
        file_structure = "\n".join([f"- {filename} ({len(content)} chars)" 
                                   for filename, content in self.codebase_files.items()])
        
        # Create content preview for smaller files
        content_previews = []
        for filename, content in self.codebase_files.items():
            if len(content) < 1000:
                content_previews.append(f"\n### {filename}:\n{content[:500]}...")
            else:
                content_previews.append(f"\n### {filename}:\n{content[:200]}...")
        
        summary = f"""
CODEBASE OVERVIEW:
==================

FILES STRUCTURE:
{file_structure}

CONTENT PREVIEWS:
{"".join(content_previews[:5])}  # Limit to first 5 files for summary
        """
        
        self.codebase_summary = summary
        return summary
    
    def get_contextual_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Generate a response using Claude's codebase understanding capabilities.
        
        Args:
            user_query: User's question or request
            conversation_history: Previous conversation messages
            
        Returns:
            str: Claude's response based on codebase understanding
        """
        try:
            # Prepare codebase context
            codebase_context = self._prepare_codebase_context()
            
            # Build conversation history
            messages = []
            
            # Add system message with codebase context
            system_message = f"""You are an expert code analyst and research assistant with deep understanding of complex technical documents and codebases. 

You have been provided with a codebase consisting of the following files:
{codebase_context}

Your role is to:
1. Analyze and understand the content of these files as if they were part of a unified codebase
2. Provide insightful, comprehensive responses about the content, methodologies, concepts, and relationships
3. Draw connections between different parts of the documentation/papers
4. Explain complex technical concepts clearly
5. Answer questions with precision and depth based on the provided materials

Guidelines:
- Always ground your responses in the actual content provided
- Cite specific sections or papers when making claims
- Provide comprehensive analysis that considers multiple perspectives
- If asked about code or implementation, treat the text as documentation and provide insights accordingly
- Maintain academic rigor while being accessible

IMPORTANT: Only use information from the provided codebase. Do not hallucinate or add information not present in the files."""

            # Add conversation history
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current user query
            messages.append({
                "role": "user",
                "content": user_query
            })
            
            # Generate response using Claude
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.3,
                system=system_message,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _prepare_codebase_context(self) -> str:
        """
        Prepare codebase context for Claude, handling token limits.
        
        Returns:
            str: Formatted codebase content for Claude
        """
        if not self.codebase_files:
            return "No codebase files available."
        
        context_parts = []
        total_chars = 0
        max_chars = 50000  # Conservative limit to avoid token issues
        
        for filename, content in self.codebase_files.items():
            file_section = f"\n\n==== FILE: {filename} ====\n{content}\n==== END FILE ====\n"
            
            if total_chars + len(file_section) > max_chars:
                # Truncate the content if it would exceed limits
                remaining_chars = max_chars - total_chars
                if remaining_chars > 1000:  # Only add if there's meaningful space
                    truncated_content = content[:remaining_chars-200]
                    file_section = f"\n\n==== FILE: {filename} (TRUNCATED) ====\n{truncated_content}\n[... content truncated ...]\n==== END FILE ====\n"
                    context_parts.append(file_section)
                break
            
            context_parts.append(file_section)
            total_chars += len(file_section)
        
        return "".join(context_parts)


class StreamlitChatInterface:
    """
    Streamlit-based chat interface for the codebase analyzer.
    """
    
    def __init__(self):
        """Initialize the Streamlit chat interface."""
        self.setup_page_config()
        self.analyzer: Optional[CodebaseAnalyzer] = None
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Claude Code - Codebase Understanding Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_sidebar(self) -> Tuple[str, str]:
        """
        Setup sidebar with configuration options.
        
        Returns:
            Tuple[str, str]: API key and directory path
        """
        st.sidebar.title("🔧 Configuration")
        
        # API Key input
        api_key = st.sidebar.text_input(
            "Anthropic API Key:",
            type="password",
            value=ANTHROPIC_API_KEY or "",
            help="Enter your Anthropic API key for Claude access"
        )
        
        # Directory selection
        default_dir = str(Path(__file__).parent / "test_files")
        directory_path = st.sidebar.text_input(
            "Codebase Directory:",
            value=default_dir,
            help="Path to directory containing your files"
        )
        
        # Load files button
        if st.sidebar.button("🔄 Load/Reload Codebase", type="primary"):
            if api_key and directory_path:
                try:
                    self.analyzer = CodebaseAnalyzer(api_key)
                    if self.analyzer.load_files_from_directory(directory_path):
                        st.session_state["analyzer"] = self.analyzer
                        summary = self.analyzer.generate_codebase_summary()
                        st.sidebar.success("Codebase loaded successfully!")
                        
                        # Show codebase info
                        with st.sidebar.expander("📁 Codebase Overview"):
                            st.text(f"Files loaded: {len(self.analyzer.codebase_files)}")
                            for filename in self.analyzer.codebase_files.keys():
                                st.text(f"• {filename}")
                    else:
                        st.sidebar.error("Failed to load codebase")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
            else:
                st.sidebar.warning("Please provide API key and directory path")
        
        # Display current status
        if "analyzer" in st.session_state:
            analyzer = st.session_state["analyzer"]
            st.sidebar.info(f"✅ Codebase ready ({len(analyzer.codebase_files)} files)")
        else:
            st.sidebar.warning("⚠️ Codebase not loaded")
        
        return api_key, directory_path
    
    def display_main_interface(self):
        """Display the main chat interface."""
        st.title("🤖 Claude Code - Codebase Understanding Chatbot")
        st.markdown("""
        This chatbot uses Claude's advanced understanding capabilities to analyze your codebase 
        (text/markdown files) and provide intelligent responses about the content, concepts, 
        methodologies, and relationships within your documents.
        """)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your codebase..."):
            # Check if analyzer is ready
            if "analyzer" not in st.session_state:
                st.error("Please load your codebase first using the sidebar configuration.")
                return
            
            analyzer = st.session_state["analyzer"]
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing codebase and generating response..."):
                    response = analyzer.get_contextual_response(
                        prompt, 
                        st.session_state.messages
                    )
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def display_sample_queries(self):
        """Display sample queries for user guidance."""
        if "analyzer" in st.session_state:
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
            
            for query in sample_queries:
                if st.sidebar.button(f"📝 {query}", key=f"sample_{hash(query)}"):
                    # Add the query to chat input
                    st.session_state.sample_query = query
                    
    def run(self):
        """Run the complete Streamlit application."""
        # Setup sidebar
        api_key, directory_path = self.setup_sidebar()
        
        # Display sample queries
        self.display_sample_queries()
        
        # Main interface
        self.display_main_interface()
        
        # Handle sample query injection
        if "sample_query" in st.session_state:
            st.rerun()


def main():
    """Main function to run the application."""
    try:
        app = StreamlitChatInterface()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
