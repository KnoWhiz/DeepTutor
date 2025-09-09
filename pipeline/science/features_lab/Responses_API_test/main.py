"""
Main streaming generator function for OpenAI Responses API chatbot
with web search and file research capabilities.
"""

import os
import json
import time
import tempfile
from typing import Iterator, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import requests
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import io

# Load environment variables
load_dotenv()

class ToolCallStatus(Enum):
    """Enum for tool call status tracking."""
    STARTING = "starting"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class StatusUpdate:
    """Data class for status updates."""
    step: str
    status: ToolCallStatus
    message: str
    timestamp: float
    tool_name: Optional[str] = None
    tool_output: Optional[str] = None

class ResponsesAPIChatbot:
    """OpenAI Responses API Chatbot with web search and file research capabilities."""
    
    def __init__(self):
        """Initialize the chatbot with OpenAI client and tools."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Define available tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information on a given topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_document",
                    "description": "Analyze and extract information from an uploaded document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_content": {
                                "type": "string",
                                "description": "The content of the document to analyze"
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis to perform (summary, key_points, research, etc.)",
                                "enum": ["summary", "key_points", "research", "detailed_analysis"]
                            }
                        },
                        "required": ["document_content", "analysis_type"]
                    }
                }
            }
        ]
    
    def _web_search(self, query: str) -> str:
        """Perform web search using a simple search API."""
        try:
            # Using DuckDuckGo Instant Answer API as a simple web search
            url = f"https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                result = []
                if data.get("Abstract"):
                    result.append(f"Summary: {data['Abstract']}")
                if data.get("AbstractURL"):
                    result.append(f"Source: {data['AbstractURL']}")
                
                # Add related topics if available
                if data.get("RelatedTopics"):
                    topics = data["RelatedTopics"][:3]  # Limit to first 3
                    for topic in topics:
                        if isinstance(topic, dict) and topic.get("Text"):
                            result.append(f"Related: {topic['Text']}")
                
                return "\n".join(result) if result else f"Search completed for '{query}' but no detailed results found."
            else:
                return f"Web search failed with status code: {response.status_code}"
                
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    def _analyze_document(self, document_content: str, analysis_type: str) -> str:
        """Analyze document content using OpenAI."""
        try:
            analysis_prompts = {
                "summary": "Provide a concise summary of this document:",
                "key_points": "Extract the key points and main ideas from this document:",
                "research": "Identify research findings, methodologies, and conclusions in this document:",
                "detailed_analysis": "Provide a detailed analysis including themes, arguments, and insights from this document:"
            }
            
            prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant."},
                    {"role": "user", "content": f"{prompt}\n\n{document_content[:4000]}"}  # Limit content length
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Document analysis error: {str(e)}"
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        
        if function_name == "web_search":
            return self._web_search(function_args["query"])
        elif function_name == "analyze_document":
            return self._analyze_document(
                function_args["document_content"], 
                function_args["analysis_type"]
            )
        else:
            return f"Unknown function: {function_name}"
    
    def _extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded file."""
        try:
            if uploaded_file.type == "application/pdf":
                # Handle PDF files
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            else:
                # Handle text files
                return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            return f"Error extracting text from file: {str(e)}"
    
    def generate_streaming_response(
        self, 
        user_message: str, 
        uploaded_file: Optional[Any] = None,
        conversation_history: Optional[list] = None
    ) -> Iterator[Union[StatusUpdate, str]]:
        """
        Generate streaming response with step-by-step status updates.
        
        Args:
            user_message: The user's message
            uploaded_file: Optional uploaded file for analysis
            conversation_history: Optional conversation history
            
        Yields:
            StatusUpdate objects for status updates and strings for response content
        """
        try:
            # Step 1: Initialize
            yield StatusUpdate(
                step="initialization",
                status=ToolCallStatus.STARTING,
                message="Initializing chatbot and preparing request...",
                timestamp=time.time()
            )
            
            # Prepare messages
            messages = conversation_history or []
            
            # Add system message if this is the start of conversation
            if not messages:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful AI assistant with access to web search and document analysis capabilities. Use these tools when needed to provide comprehensive and accurate responses."
                })
            
            # Process uploaded file if provided
            document_content = None
            if uploaded_file:
                yield StatusUpdate(
                    step="file_processing",
                    status=ToolCallStatus.IN_PROGRESS,
                    message=f"Processing uploaded file: {uploaded_file.name}",
                    timestamp=time.time()
                )
                
                document_content = self._extract_text_from_file(uploaded_file)
                
                if document_content:
                    user_message += f"\n\nPlease also analyze this uploaded document:\n{document_content[:2000]}..."
                    
                    yield StatusUpdate(
                        step="file_processing",
                        status=ToolCallStatus.COMPLETED,
                        message="File processed successfully",
                        timestamp=time.time()
                    )
                else:
                    yield StatusUpdate(
                        step="file_processing",
                        status=ToolCallStatus.ERROR,
                        message="Failed to extract content from file",
                        timestamp=time.time()
                    )
            
            # Add user message
            messages.append({"role": "user", "content": user_message})
            
            yield StatusUpdate(
                step="initialization",
                status=ToolCallStatus.COMPLETED,
                message="Request prepared, sending to OpenAI...",
                timestamp=time.time()
            )
            
            # Step 2: Make API call
            yield StatusUpdate(
                step="api_call",
                status=ToolCallStatus.IN_PROGRESS,
                message="Sending request to OpenAI API...",
                timestamp=time.time()
            )
            
            # Create chat completion with tools
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
                max_tokens=2000
            )
            
            yield StatusUpdate(
                step="api_call",
                status=ToolCallStatus.COMPLETED,
                message="Connected to OpenAI API, receiving response...",
                timestamp=time.time()
            )
            
            # Step 3: Process streaming response
            yield StatusUpdate(
                step="response_processing",
                status=ToolCallStatus.IN_PROGRESS,
                message="Processing streaming response...",
                timestamp=time.time()
            )
            
            full_response = ""
            tool_calls = []
            current_tool_call = None
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
                
                # Handle tool calls
                if chunk.choices[0].delta.tool_calls:
                    for tool_call_delta in chunk.choices[0].delta.tool_calls:
                        if tool_call_delta.index is not None:
                            # New tool call
                            if tool_call_delta.index >= len(tool_calls):
                                tool_calls.append({
                                    "id": tool_call_delta.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_delta.function.name if tool_call_delta.function.name else "",
                                        "arguments": tool_call_delta.function.arguments if tool_call_delta.function.arguments else ""
                                    }
                                })
                                current_tool_call = tool_calls[-1]
                                
                                yield StatusUpdate(
                                    step="tool_call",
                                    status=ToolCallStatus.STARTING,
                                    message=f"Starting tool call: {tool_call_delta.function.name}",
                                    timestamp=time.time(),
                                    tool_name=tool_call_delta.function.name
                                )
                            else:
                                current_tool_call = tool_calls[tool_call_delta.index]
                        
                        # Update current tool call
                        if current_tool_call and tool_call_delta.function:
                            if tool_call_delta.function.arguments:
                                current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Execute tool calls if any
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": tool_calls
                })
                
                for tool_call in tool_calls:
                    yield StatusUpdate(
                        step="tool_execution",
                        status=ToolCallStatus.IN_PROGRESS,
                        message=f"Executing {tool_call['function']['name']}...",
                        timestamp=time.time(),
                        tool_name=tool_call['function']['name']
                    )
                    
                    # Execute the tool call
                    tool_result = self._execute_tool_call(tool_call)
                    
                    yield StatusUpdate(
                        step="tool_execution",
                        status=ToolCallStatus.COMPLETED,
                        message=f"Tool {tool_call['function']['name']} completed",
                        timestamp=time.time(),
                        tool_name=tool_call['function']['name'],
                        tool_output=tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    )
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
                
                # Get final response with tool results
                yield StatusUpdate(
                    step="final_response",
                    status=ToolCallStatus.IN_PROGRESS,
                    message="Generating final response with tool results...",
                    timestamp=time.time()
                )
                
                final_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    stream=True,
                    max_tokens=1500
                )
                
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            yield StatusUpdate(
                step="completion",
                status=ToolCallStatus.COMPLETED,
                message="Response generation completed successfully",
                timestamp=time.time()
            )
            
        except Exception as e:
            yield StatusUpdate(
                step="error",
                status=ToolCallStatus.ERROR,
                message=f"Error occurred: {str(e)}",
                timestamp=time.time()
            )


def create_chatbot() -> ResponsesAPIChatbot:
    """Factory function to create a chatbot instance."""
    return ResponsesAPIChatbot()


if __name__ == "__main__":
    # Test the chatbot
    chatbot = create_chatbot()
    
    print("Testing chatbot...")
    for update in chatbot.generate_streaming_response("What is the latest news about AI?"):
        if isinstance(update, StatusUpdate):
            print(f"[{update.step}] {update.status.value}: {update.message}")
        else:
            print(update, end="", flush=True)
    print("\nTest completed!")
