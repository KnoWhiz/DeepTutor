"""
Streamlit Chatbot UI for Gemini CLI Streaming

A web-based chatbot interface that demonstrates the gemini_stream.py functionality
with real-time streaming responses and chat history.

Usage:
    streamlit run streamlit_chatbot.py

Features:
    - Real-time streaming responses
    - Chat history with conversation memory
    - Model selection and parameter controls
    - Error handling with user-friendly messages
    - Export chat history
    - Responsive design
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import traceback

# Import our streaming module
try:
    from gemini_stream import stream_gemini, GeminiCLIError
except ImportError:
    st.error("Could not import gemini_stream module. Please ensure gemini_stream.py is in the same directory.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Gemini CLI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    .streaming-indicator {
        animation: pulse 1.5s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { opacity: 0.6; }
        to { opacity: 1; }
    }
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
    
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_message(role: str, content: str, timestamp: Optional[str] = None) -> None:
    """Display a chat message with proper styling."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    icon = "👤" if role == "user" else "🤖"
    css_class = "user-message" if role == "user" else "assistant-message"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div style="font-size: 1.2rem;">{icon}</div>
        <div style="flex: 1;">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">
                {role.title()} <span style="font-size: 0.8rem; color: #666;">({timestamp})</span>
            </div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_error_message(error: str, details: Optional[str] = None) -> None:
    """Display an error message with proper styling."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="chat-message error-message">
        <div style="font-size: 1.2rem;">❌</div>
        <div style="flex: 1;">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">
                Error <span style="font-size: 0.8rem; color: #666;">({timestamp})</span>
            </div>
            <div><strong>{error}</strong></div>
            {f'<div style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">{details}</div>' if details else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def stream_response(prompt: str, model: str, extra_args: List[str], timeout: float) -> None:
    """Stream response from Gemini CLI and update the UI in real-time."""
    response_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Show streaming indicator
        status_placeholder.markdown('<div class="streaming-indicator">🤖 Generating response...</div>', unsafe_allow_html=True)
        
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        
        # Stream the response
        for chunk in stream_gemini(
            prompt=prompt,
            model=model,
            extra_args=extra_args if extra_args else None,
            timeout=timeout
        ):
            # Skip empty chunks from cancellation checks
            if not chunk.strip():
                continue
                
            full_response += chunk
            chunk_count += 1
            
            # Update the display every few chunks or every second
            if chunk_count % 5 == 0 or time.time() - start_time > 1:
                response_placeholder.markdown(full_response)
                start_time = time.time()
        
        # Final update
        response_placeholder.markdown(full_response)
        status_placeholder.empty()
        
        # Add to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "model": model,
            "chunk_count": chunk_count
        })
        
        # Add to chat history for export
        st.session_state.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": prompt,
            "assistant_response": full_response,
            "model": model,
            "parameters": {
                "extra_args": extra_args,
                "timeout": timeout,
                "chunk_count": chunk_count
            }
        })
        
    except GeminiCLIError as e:
        status_placeholder.empty()
        response_placeholder.empty()
        display_error_message(
            f"Gemini CLI Error (Exit Code: {e.exit_code})",
            f"Details: {e.stderr_tail}"
        )
        
    except TimeoutError:
        status_placeholder.empty()
        response_placeholder.empty()
        display_error_message(
            "Request Timeout",
            f"The request timed out after {timeout} seconds. Try reducing the complexity of your prompt or increasing the timeout."
        )
        
    except FileNotFoundError:
        status_placeholder.empty()
        response_placeholder.empty()
        display_error_message(
            "Gemini CLI Not Found",
            "Please ensure the Gemini CLI is installed and accessible in your PATH. You can install it from the official Google AI documentation."
        )
        
    except Exception as e:
        status_placeholder.empty()
        response_placeholder.empty()
        display_error_message(
            "Unexpected Error",
            f"An unexpected error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        )


def export_chat_history() -> str:
    """Export chat history as JSON."""
    if not st.session_state.chat_history:
        return ""
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_conversations": len(st.session_state.chat_history),
        "conversations": st.session_state.chat_history
    }
    
    return json.dumps(export_data, indent=2)


def main() -> None:
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 Gemini CLI Chatbot")
    st.markdown("*Real-time streaming chatbot powered by Gemini CLI*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Model Settings")
        model = st.selectbox(
            "Select Gemini Model",
            options=[
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro"
            ],
            index=0,
            help="Choose the Gemini model for responses"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced parameters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Advanced Parameters")
        
        timeout = st.slider(
            "Timeout (seconds)",
            min_value=10.0,
            max_value=300.0,
            value=60.0,
            step=10.0,
            help="Maximum time to wait for a response"
        )
        
        # Extra CLI arguments
        extra_args_text = st.text_area(
            "Extra CLI Arguments",
            placeholder="--temperature 0.7\n--top-p 0.9\n--max-tokens 1000",
            help="Additional arguments to pass to the Gemini CLI (one per line)"
        )
        
        extra_args = []
        if extra_args_text.strip():
            extra_args = [arg.strip() for arg in extra_args_text.strip().split('\n') if arg.strip()]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Chat Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📊 Stats", use_container_width=True):
                if st.session_state.chat_history:
                    total_conversations = len(st.session_state.chat_history)
                    total_messages = len(st.session_state.messages)
                    st.info(f"💬 {total_conversations} conversations\n📝 {total_messages} messages")
                else:
                    st.info("No conversations yet!")
        
        # Export functionality
        if st.session_state.chat_history:
            export_data = export_chat_history()
            st.download_button(
                label="📥 Export Chat History",
                data=export_data,
                file_name=f"gemini_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("ℹ️ System Info")
        st.markdown(f"""
        - **Model**: {model}
        - **Timeout**: {timeout}s
        - **Extra Args**: {len(extra_args)} parameters
        - **Messages**: {len(st.session_state.messages)}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display existing messages
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
            timestamp=message.get("timestamp")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask Gemini anything..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display user message
        display_message("user", prompt)
        
        # Stream response
        with st.container():
            stream_response(prompt, model, extra_args, timeout)
    
    # Instructions
    if not st.session_state.messages:
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Configure your settings** in the sidebar (model, timeout, etc.)
        2. **Type your message** in the chat input below
        3. **Watch the response stream** in real-time
        4. **Export your chat history** when done
        
        ### 💡 Tips
        - Use the **timeout slider** to control how long to wait for responses
        - Add **extra CLI arguments** for fine-tuning (temperature, top-p, etc.)
        - **Clear the chat** to start fresh conversations
        - **Export your history** to save interesting conversations
        
        ### ⚠️ Requirements
        - Ensure the **Gemini CLI** is installed and accessible in your PATH
        - Check that your **API credentials** are properly configured
        """)


if __name__ == "__main__":
    main()
