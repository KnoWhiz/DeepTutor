"""
Streamlit UI for OpenAI Responses API Chatbot
with real-time status updates and tool call tracking.
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any
import json

from main import create_chatbot, StatusUpdate, ToolCallStatus

# Page configuration
st.set_page_config(
    page_title="AI Research Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .status-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .status-starting {
        border-left-color: #2196F3;
        background-color: #e3f2fd;
    }
    
    .status-in-progress {
        border-left-color: #FF9800;
        background-color: #fff3e0;
    }
    
    .status-completed {
        border-left-color: #4CAF50;
        background-color: #e8f5e8;
    }
    
    .status-error {
        border-left-color: #F44336;
        background-color: #ffebee;
    }
    
    .tool-output {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = create_chatbot()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.stop()
    
    if "status_updates" not in st.session_state:
        st.session_state.status_updates = []
    
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

def display_status_update(status: StatusUpdate):
    """Display a status update with appropriate styling."""
    status_class = f"status-{status.status.value.replace('_', '-')}"
    timestamp = datetime.fromtimestamp(status.timestamp).strftime("%H:%M:%S")
    
    status_icon = {
        ToolCallStatus.STARTING: "🔄",
        ToolCallStatus.IN_PROGRESS: "⏳", 
        ToolCallStatus.COMPLETED: "✅",
        ToolCallStatus.ERROR: "❌"
    }
    
    icon = status_icon.get(status.status, "ℹ️")
    
    html_content = f"""
    <div class="status-container {status_class}">
        <div style="display: flex; justify-content: between; align-items: center;">
            <div style="flex: 1;">
                <strong>{icon} {status.step.replace('_', ' ').title()}</strong>
                {f" - {status.tool_name}" if status.tool_name else ""}
            </div>
            <div style="font-size: 0.8em; color: #666;">
                {timestamp}
            </div>
        </div>
        <div style="margin-top: 5px; font-size: 0.9em;">
            {status.message}
        </div>
        {f'<div class="tool-output"><strong>Tool Output:</strong><br>{status.tool_output}</div>' if status.tool_output else ""}
    </div>
    """
    
    return html_content

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling."""
    message_class = f"{role}-message"
    icon = "👤" if role == "user" else "🤖"
    
    html_content = f"""
    <div class="chat-message {message_class}">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            <div style="font-size: 1.2em;">{icon}</div>
            <div style="flex: 1;">
                <strong>{role.title()}:</strong><br>
                {content}
            </div>
        </div>
    </div>
    """
    
    return html_content

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 AI Research Chatbot")
    st.markdown("**Powered by OpenAI Responses API with Web Search & Document Analysis**")
    
    # Sidebar for settings and information
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.status_updates = []
            st.session_state.current_response = ""
            st.rerun()
        
        st.divider()
        
        # File upload
        st.header("📄 Document Analysis")
        uploaded_file = st.file_uploader(
            "Upload a document for analysis",
            type=["txt", "pdf"],
            help="Upload a text file or PDF for the AI to analyze and research"
        )
        
        st.divider()
        
        # Information
        st.header("ℹ️ Features")
        st.markdown("""
        - **🔍 Web Search**: Real-time web search capabilities
        - **📊 Document Analysis**: Upload and analyze documents
        - **🔄 Streaming Responses**: Real-time response generation
        - **📈 Status Tracking**: Step-by-step process visibility
        - **🛠️ Tool Calling**: Automatic tool selection and execution
        """)
        
        st.divider()
        
        # Status updates section
        st.header("📊 Current Status")
        if st.session_state.status_updates:
            latest_status = st.session_state.status_updates[-1]
            st.markdown(f"**Step:** {latest_status.step.replace('_', ' ').title()}")
            st.markdown(f"**Status:** {latest_status.status.value.replace('_', ' ').title()}")
            if latest_status.tool_name:
                st.markdown(f"**Tool:** {latest_status.tool_name}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display conversation history
        if st.session_state.messages:
            for message in st.session_state.messages:
                st.markdown(
                    display_chat_message(message["role"], message["content"]),
                    unsafe_allow_html=True
                )
        
        # Current response area
        if st.session_state.current_response:
            with st.container():
                st.markdown("**🤖 Assistant (generating...):**")
                response_placeholder = st.empty()
                response_placeholder.markdown(st.session_state.current_response)
    
    with col2:
        st.header("🔄 Status Updates")
        
        # Status updates container
        status_container = st.container()
        
        with status_container:
            if st.session_state.status_updates:
                for status in st.session_state.status_updates[-10:]:  # Show last 10 updates
                    st.markdown(
                        display_status_update(status),
                        unsafe_allow_html=True
                    )
            else:
                st.info("No status updates yet. Start a conversation to see real-time updates!")
    
    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        st.subheader("✍️ Your Message")
        user_input = st.text_area(
            "Type your message here...",
            height=100,
            placeholder="Ask me anything! I can search the web, analyze documents, and provide research insights."
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button(
                "🚀 Send Message",
                use_container_width=True,
                type="primary"
            )
    
    # Process user input
    if submit_button and user_input.strip():
        # Add user message to conversation
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)
        st.session_state.conversation_history.append(user_message)
        
        # Clear previous status updates and current response
        st.session_state.status_updates = []
        st.session_state.current_response = ""
        
        # Create placeholders for real-time updates
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            # Generate streaming response
            response_content = ""
            
            for update in st.session_state.chatbot.generate_streaming_response(
                user_input, 
                uploaded_file,
                st.session_state.conversation_history
            ):
                if isinstance(update, StatusUpdate):
                    # Update status
                    st.session_state.status_updates.append(update)
                    
                    # Update status display in sidebar
                    with st.sidebar:
                        if st.session_state.status_updates:
                            latest = st.session_state.status_updates[-1]
                            st.markdown(f"**Current:** {latest.step.replace('_', ' ').title()}")
                            st.markdown(f"**Status:** {latest.status.value.replace('_', ' ').title()}")
                
                else:
                    # Update response content
                    response_content += update
                    st.session_state.current_response = response_content
                    
                    # Update response display
                    response_placeholder.markdown(f"**🤖 Assistant:** {response_content}")
                
                # Small delay to make updates visible
                time.sleep(0.01)
            
            # Add assistant response to conversation
            if response_content:
                assistant_message = {"role": "assistant", "content": response_content}
                st.session_state.messages.append(assistant_message)
                st.session_state.conversation_history.append(assistant_message)
            
            # Clear current response placeholder
            st.session_state.current_response = ""
            
            # Rerun to update the display
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
            # Add error status update
            error_status = StatusUpdate(
                step="error",
                status=ToolCallStatus.ERROR,
                message=f"Error: {str(e)}",
                timestamp=time.time()
            )
            st.session_state.status_updates.append(error_status)

if __name__ == "__main__":
    main()
