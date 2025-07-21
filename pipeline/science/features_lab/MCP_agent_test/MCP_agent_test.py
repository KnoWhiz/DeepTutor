#!/usr/bin/env python3
"""
MCP Agent Test - Advanced Chatbot with OpenAI Assistant API and MCP Integration

This module implements a Streamlit-based chatbot that uses OpenAI's Assistant API
with Model Context Protocol (MCP) integration for Notion and Dropbox.
Features ReAct (Reasoning and Acting) agentic workflow.

Author: DeepTutor Team
Created: 2025
"""

import streamlit as st
import openai
import os
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mcp_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    """Configuration for MCP endpoints"""
    name: str
    endpoint: str
    is_connected: bool = False
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

class MCPAgentChatbot:
    """
    Advanced MCP Agent Chatbot using OpenAI Assistant API
    
    This class implements a ReAct workflow chatbot that can:
    1. Connect to multiple MCP servers (Notion, Dropbox)
    2. Determine which MCP to use based on user queries
    3. Make plans and execute actions using MCP tools
    4. Provide intelligent responses with data retrieval
    """
    
    def __init__(self):
        """Initialize the MCP Agent Chatbot"""
        self.client = None
        self.assistant = None
        self.thread = None
        self.mcp_configs: Dict[str, MCPConfig] = {}
        self._setup_openai_client()
        self._initialize_session_state()
    
    def _setup_openai_client(self):
        """Setup OpenAI client with API key"""
        try:
            # Try to get API key from environment or use provided key
            api_key = os.getenv("OPENAI_API_KEY", "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")
            if not api_key:
                raise ValueError("OpenAI API key not found")
                
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "assistant_id" not in st.session_state:
            st.session_state.assistant_id = None
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = None
        if "mcp_endpoints" not in st.session_state:
            st.session_state.mcp_endpoints = {
                "notion": "",
                "dropbox": ""
            }
        if "mcp_connected" not in st.session_state:
            st.session_state.mcp_connected = {
                "notion": False,
                "dropbox": False
            }
    
    def validate_mcp_endpoint(self, endpoint: str) -> Tuple[bool, str]:
        """
        Validate MCP endpoint by making a test request
        
        Args:
            endpoint: The MCP endpoint URL to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if not endpoint or not endpoint.startswith("http"):
                return False, "Invalid URL format"
            
            # Simple validation - in real implementation, you'd test the MCP protocol
            response = requests.head(endpoint, timeout=5)
            if response.status_code < 400:
                return True, "Endpoint is accessible"
            else:
                return False, f"Endpoint returned status {response.status_code}"
                
        except requests.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def setup_mcp_connection(self, service: str, endpoint: str) -> bool:
        """
        Setup MCP connection for a service
        
        Args:
            service: Service name (notion/dropbox)
            endpoint: MCP endpoint URL
            
        Returns:
            Boolean indicating success
        """
        try:
            # Use default endpoints if provided endpoint is invalid
            default_endpoints = {
                "notion": "http://mcp.composio.dev/partner/composio/notion/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc",
                "dropbox": "http://mcp.composio.dev/partner/composio/dropbox/mcp?customerId=4d31a7c9-fe76-471d-a074-2c33da41f7bc"
            }
            
            if not endpoint:
                endpoint = default_endpoints.get(service, "")
            
            is_valid, message = self.validate_mcp_endpoint(endpoint)
            if not is_valid:
                logger.warning(f"Invalid endpoint for {service}: {message}. Using default.")
                endpoint = default_endpoints.get(service, "")
                is_valid, _ = self.validate_mcp_endpoint(endpoint)
            
            if is_valid:
                self.mcp_configs[service] = MCPConfig(
                    name=service.title(),
                    endpoint=endpoint,
                    is_connected=True,
                    capabilities=self._get_mcp_capabilities(service)
                )
                st.session_state.mcp_endpoints[service] = endpoint
                st.session_state.mcp_connected[service] = True
                logger.info(f"✅ MCP connection established for {service}")
                return True
            else:
                logger.error(f"❌ Failed to connect to {service} MCP")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error setting up MCP connection for {service}: {str(e)}")
            return False
    
    def _get_mcp_capabilities(self, service: str) -> List[str]:
        """
        Get capabilities for MCP service
        
        Args:
            service: Service name
            
        Returns:
            List of capabilities
        """
        capabilities_map = {
            "notion": [
                "create_page", "read_page", "update_page", "search_pages",
                "create_database", "query_database", "add_content"
            ],
            "dropbox": [
                "list_files", "upload_file", "download_file", "create_folder",
                "share_file", "search_files", "get_metadata"
            ]
        }
        return capabilities_map.get(service, [])
    
    def create_assistant(self) -> bool:
        """
        Create OpenAI Assistant with MCP tools integration
        
        Returns:
            Boolean indicating success
        """
        try:
            # Build assistant instructions with ReAct workflow
            instructions = self._build_assistant_instructions()
            
            # Create tools for MCP integration
            tools = self._create_mcp_tools()
            
            # Create the assistant
            assistant = self.client.beta.assistants.create(
                name="MCP Agent",
                description="Advanced AI assistant with MCP integration for Notion and Dropbox",
                instructions=instructions,
                model="gpt-4-turbo-preview",
                tools=tools
            )
            
            self.assistant = assistant
            st.session_state.assistant_id = assistant.id
            
            # Create a new thread for the conversation
            thread = self.client.beta.threads.create()
            self.thread = thread
            st.session_state.thread_id = thread.id
            
            logger.info(f"✅ Assistant created with ID: {assistant.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create assistant: {str(e)}")
            st.error(f"Failed to create assistant: {str(e)}")
            return False
    
    def _build_assistant_instructions(self) -> str:
        """
        Build comprehensive instructions for the assistant with ReAct workflow
        
        Returns:
            Instruction string for the assistant
        """
        connected_services = [name for name, config in self.mcp_configs.items() if config.is_connected]
        
        instructions = f"""
You are an advanced AI assistant with access to MCP (Model Context Protocol) services.
Connected services: {', '.join(connected_services)}

REACT WORKFLOW - Follow this pattern for every user query:

1. REASON: Analyze the user's request and determine:
   - What information or action is needed
   - Which MCP service(s) are most appropriate
   - What specific capabilities to use
   - Whether you have enough information to proceed

2. ACT: Take action by:
   - Calling appropriate MCP tools
   - Gathering necessary data
   - Performing requested operations

3. OBSERVE: Evaluate the results:
   - Check if the action was successful
   - Assess if more information is needed
   - Determine if the user's query is fully addressed

4. REPEAT: If needed, repeat the cycle until the task is complete

MCP SERVICE CAPABILITIES:

Notion MCP ({self.mcp_configs.get('notion', MCPConfig('', '')).endpoint}):
- Create and manage pages
- Query databases
- Search content
- Add page content
- Manage comments and discussions

Dropbox MCP ({self.mcp_configs.get('dropbox', MCPConfig('', '')).endpoint}):
- File operations (upload, download, list)
- Folder management
- File sharing and permissions
- Search files
- Metadata operations

DECISION MAKING:
- Use Notion for: notes, documentation, databases, project management, content creation
- Use Dropbox for: file storage, sharing, backup, document management
- If unclear which service to use, ask the user for clarification
- Always explain your reasoning before taking action
- Provide clear status updates during multi-step operations

ERROR HANDLING:
- If an MCP call fails, explain what went wrong and suggest alternatives
- Always validate user inputs before making MCP calls
- Provide helpful error messages and recovery suggestions

RESPONSE STYLE:
- Be conversational but professional
- Explain your thought process
- Provide clear action confirmations
- Ask for clarification when needed
"""
        return instructions.strip()
    
    def _create_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Create tool definitions for MCP integration
        
        Returns:
            List of tool definitions
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_notion_mcp",
                    "description": "Call Notion MCP service for content and database operations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to perform",
                                "enum": ["create_page", "read_page", "update_page", "search_pages", 
                                        "create_database", "query_database", "add_content"]
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the MCP call",
                                "additionalProperties": True
                            }
                        },
                        "required": ["action", "parameters"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_dropbox_mcp",
                    "description": "Call Dropbox MCP service for file operations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to perform",
                                "enum": ["list_files", "upload_file", "download_file", "create_folder",
                                        "share_file", "search_files", "get_metadata"]
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the MCP call",
                                "additionalProperties": True
                            }
                        },
                        "required": ["action", "parameters"]
                    }
                }
            }
        ]
        return tools
    
    def call_mcp_service(self, service: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a call to MCP service
        
        Args:
            service: Service name (notion/dropbox)
            action: Action to perform
            parameters: Parameters for the action
            
        Returns:
            Response from MCP service
        """
        try:
            if service not in self.mcp_configs or not self.mcp_configs[service].is_connected:
                return {
                    "success": False,
                    "error": f"{service.title()} MCP not connected"
                }
            
            # Simulate MCP call (in real implementation, you'd use the actual MCP protocol)
            logger.info(f"🔄 Calling {service} MCP: {action} with params {parameters}")
            
            # Mock responses for demonstration
            mock_responses = self._get_mock_mcp_responses(service, action, parameters)
            
            time.sleep(0.5)  # Simulate network delay
            return mock_responses
            
        except Exception as e:
            logger.error(f"❌ MCP call failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_mock_mcp_responses(self, service: str, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock responses for MCP calls (for demonstration)
        
        Args:
            service: Service name
            action: Action performed
            parameters: Action parameters
            
        Returns:
            Mock response data
        """
        if service == "notion":
            if action == "search_pages":
                return {
                    "success": True,
                    "data": {
                        "pages": [
                            {"id": "page1", "title": "Project Notes", "last_edited": "2025-01-01"},
                            {"id": "page2", "title": "Meeting Minutes", "last_edited": "2024-12-30"}
                        ]
                    }
                }
            elif action == "create_page":
                return {
                    "success": True,
                    "data": {
                        "page_id": "new_page_123",
                        "title": parameters.get("title", "Untitled"),
                        "url": "https://notion.so/new_page_123"
                    }
                }
            elif action == "query_database":
                return {
                    "success": True,
                    "data": {
                        "results": [
                            {"id": "db1", "properties": {"Name": "Task 1", "Status": "In Progress"}},
                            {"id": "db2", "properties": {"Name": "Task 2", "Status": "Complete"}}
                        ]
                    }
                }
        
        elif service == "dropbox":
            if action == "list_files":
                return {
                    "success": True,
                    "data": {
                        "files": [
                            {"name": "document.pdf", "size": 1024, "path": "/documents/document.pdf"},
                            {"name": "image.jpg", "size": 2048, "path": "/images/image.jpg"}
                        ]
                    }
                }
            elif action == "upload_file":
                return {
                    "success": True,
                    "data": {
                        "file_id": "file_123",
                        "name": parameters.get("filename", "uploaded_file"),
                        "path": f"/uploads/{parameters.get('filename', 'uploaded_file')}"
                    }
                }
            elif action == "search_files":
                query = parameters.get("query", "")
                return {
                    "success": True,
                    "data": {
                        "matches": [
                            {"name": f"result_{query}_1.doc", "path": "/search/result1.doc"},
                            {"name": f"result_{query}_2.pdf", "path": "/search/result2.pdf"}
                        ]
                    }
                }
        
        # Default response
        return {
            "success": True,
            "data": {
                "action": action,
                "parameters": parameters,
                "message": f"Mock response for {service} {action}"
            }
        }
    
    def process_message(self, user_message: str) -> Optional[str]:
        """
        Process user message using OpenAI Assistant with ReAct workflow
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response or None if error
        """
        try:
            if not self.assistant or not self.thread:
                st.error("Assistant not initialized. Please restart the application.")
                return None
            
            # Add user message to thread
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_message
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )
            
            # Wait for completion with tool call handling
            response = self._handle_run_completion(run.id)
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _handle_run_completion(self, run_id: str) -> str:
        """
        Handle run completion with tool calls
        
        Args:
            run_id: Run ID to monitor
            
        Returns:
            Final response from assistant
        """
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check run status
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run_id
            )
            
            if run.status == "completed":
                # Get the latest message
                messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id,
                    limit=1
                )
                return messages.data[0].content[0].text.value
            
            elif run.status == "requires_action":
                # Handle tool calls
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    output = self._handle_tool_call(tool_call)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(output)
                    })
                
                # Submit tool outputs
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )
            
            elif run.status == "failed":
                return f"Sorry, the request failed: {run.last_error.message if run.last_error else 'Unknown error'}"
            
            elif run.status == "expired":
                return "Sorry, the request timed out. Please try again."
            
            # Wait before checking again
            time.sleep(1)
        
        return "Sorry, the request took too long to complete."
    
    def _handle_tool_call(self, tool_call) -> Dict[str, Any]:
        """
        Handle individual tool call
        
        Args:
            tool_call: Tool call object
            
        Returns:
            Tool call result
        """
        try:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "call_notion_mcp":
                return self.call_mcp_service("notion", arguments["action"], arguments["parameters"])
            elif function_name == "call_dropbox_mcp":
                return self.call_mcp_service("dropbox", arguments["action"], arguments["parameters"])
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            logger.error(f"❌ Tool call error: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="MCP Agent Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 MCP Agent Chatbot")
    st.markdown("*Advanced AI Assistant with Notion & Dropbox Integration*")
    
    # Initialize the chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = MCPAgentChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar for MCP configuration
    with st.sidebar:
        st.header("🔧 MCP Configuration")
        
        # Notion Configuration
        st.subheader("📝 Notion MCP")
        notion_endpoint = st.text_input(
            "Notion MCP Endpoint",
            value=st.session_state.mcp_endpoints.get("notion", ""),
            placeholder="http://mcp.composio.dev/partner/composio/notion/mcp?customerId=..."
        )
        
        if st.button("Connect Notion MCP", type="secondary"):
            with st.spinner("Connecting to Notion MCP..."):
                success = chatbot.setup_mcp_connection("notion", notion_endpoint)
                if success:
                    st.success("✅ Notion MCP connected!")
                else:
                    st.error("❌ Failed to connect to Notion MCP")
        
        st.write(f"Status: {'🟢 Connected' if st.session_state.mcp_connected.get('notion') else '🔴 Disconnected'}")
        
        st.divider()
        
        # Dropbox Configuration
        st.subheader("📁 Dropbox MCP")
        dropbox_endpoint = st.text_input(
            "Dropbox MCP Endpoint",
            value=st.session_state.mcp_endpoints.get("dropbox", ""),
            placeholder="http://mcp.composio.dev/partner/composio/dropbox/mcp?customerId=..."
        )
        
        if st.button("Connect Dropbox MCP", type="secondary"):
            with st.spinner("Connecting to Dropbox MCP..."):
                success = chatbot.setup_mcp_connection("dropbox", dropbox_endpoint)
                if success:
                    st.success("✅ Dropbox MCP connected!")
                else:
                    st.error("❌ Failed to connect to Dropbox MCP")
        
        st.write(f"Status: {'🟢 Connected' if st.session_state.mcp_connected.get('dropbox') else '🔴 Disconnected'}")
        
        st.divider()
        
        # Initialize Assistant
        if st.button("🚀 Initialize Assistant", type="primary"):
            if any(st.session_state.mcp_connected.values()):
                with st.spinner("Creating AI Assistant..."):
                    success = chatbot.create_assistant()
                    if success:
                        st.success("✅ Assistant ready!")
                    else:
                        st.error("❌ Failed to create assistant")
            else:
                st.warning("⚠️ Please connect at least one MCP service first!")
        
        # Status indicators
        st.subheader("📊 Status")
        st.write(f"Assistant: {'🟢 Ready' if st.session_state.assistant_id else '🔴 Not Ready'}")
        st.write(f"Thread: {'🟢 Active' if st.session_state.thread_id else '🔴 Inactive'}")
    
    # Main chat interface
    if st.session_state.assistant_id:
        st.success("🎉 Assistant is ready! You can now start chatting.")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about your Notion pages or Dropbox files..."):
                # Display user message
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking and acting..."):
                        response = chatbot.process_message(prompt)
                        
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "Sorry, I couldn't process your request. Please try again."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    else:
        st.info("👈 Please configure and initialize the MCP connections and assistant in the sidebar to start chatting.")
        
        # Example queries
        st.subheader("💡 Example Queries")
        st.markdown("""
        **Notion Examples:**
        - "Search for pages about machine learning"
        - "Create a new page called 'Project Plan'"
        - "Show me all tasks in my database"
        - "Add a note about today's meeting to my project page"
        
        **Dropbox Examples:**
        - "List all files in my documents folder"
        - "Upload the file 'report.pdf' to my work folder"
        - "Search for files containing 'budget'"
        - "Share my presentation with read-only access"
        
        **Cross-Platform Examples:**
        - "Save my Notion meeting notes to Dropbox as a PDF"
        - "Create a Notion page with content from my Dropbox file"
        """)

if __name__ == "__main__":
    main()
