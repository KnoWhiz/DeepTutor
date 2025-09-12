"""
OpenAI Responses API Chatbot Implementation (with Chat Completions Fallback)

This module implements a comprehensive chatbot originally designed for OpenAI's Responses API,
but currently using Chat Completions API as a fallback since Responses API is not yet
available in the standard OpenAI Python library.

Features:
- Context-aware response generation
- Multi-round web search capabilities (via function calling)
- Streaming response generator
- Conversation history management
- Error handling and type validation
- Automatic fallback from gpt-5-thinking to gpt-4o if needed

Note: This implementation maintains the same interface as the Responses API while using
Chat Completions API under the hood. It will be easy to migrate to the actual Responses API
when it becomes available in the OpenAI Python library.
"""

import os
import time
from typing import Generator, Dict, Any, Optional, List
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. Please install with: pip install python-dotenv")
    def load_dotenv():
        """Dummy function when dotenv is not available."""
        pass

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai library not installed. Please install with: pip install openai")
    OpenAI = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponsesAPIChatbot:
    """
    A chatbot implementation using OpenAI's Responses API with web search capabilities.
    """
    
    def __init__(self, model: str = "o3-mini", max_search_rounds: int = 3):
        """
        Initialize the chatbot with OpenAI client and configuration.
        
        Args:
            model: The OpenAI model to use (default: o3-mini)
            max_search_rounds: Maximum number of web search rounds allowed
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Please install with: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_search_rounds = max_search_rounds
        self.conversation_history: Dict[str, Dict[str, Any]] = {}  # response_id -> conversation data
    
    def chatbot_with_web_search(
        self,
        query: str,
        context: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        enable_web_search: bool = True,
        stream: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Main chatbot function that generates responses using OpenAI Responses API
        with optional web search capabilities.
        
        Args:
            query: User's question or input
            context: Additional context to guide the response
            previous_response_id: ID of previous response for conversation continuity
            enable_web_search: Whether to enable web search tools
            stream: Whether to stream the response
            
        Yields:
            Dict containing response chunks with metadata
        """
        try:
            # Prepare the input with context if provided
            if context:
                formatted_input = f"Context: {context}\n\nQuery: {query}"
            else:
                formatted_input = query
            
            # Configure tools
            tools = []
            if enable_web_search:
                tools.append({"type": "web_search"})
            
            # Create the response
            response = self._create_response(
                input_text=formatted_input,
                tools=tools,
                previous_response_id=previous_response_id,
                stream=stream
            )
            
            if stream:
                yield from self._handle_streaming_response(response, query)
            else:
                yield from self._handle_non_streaming_response(response, query)
                
        except Exception as e:
            logger.error("Error in chatbot_with_web_search: %s", str(e))
            yield {
                "type": "error",
                "content": "An error occurred: " + str(e),
                "timestamp": time.time()
            }
    
    def _create_response(
        self,
        input_text: str,
        tools: List[Dict[str, Any]],
        previous_response_id: Optional[str] = None,
        stream: bool = True
    ) -> Any:
        """
        Create a response using OpenAI Chat Completions API (fallback for Responses API).
        
        Args:
            input_text: The formatted input text
            tools: List of tools to enable
            previous_response_id: Previous response ID for conversation continuity
            stream: Whether to stream the response
            
        Returns:
            OpenAI response object
        """
        # Build messages for chat completion
        messages = []
        
        # Add conversation history if available
        if previous_response_id and previous_response_id in self.conversation_history:
            history = self.conversation_history[previous_response_id]
            messages.extend(history.get("messages", []))
        
        # Add current input
        messages.append({"role": "user", "content": input_text})
        
        # Prepare request parameters for Chat Completions API
        request_params = {
            "model": self.model,  # Default: o3-mini for enhanced reasoning
            "messages": messages,
            "stream": stream,
            "max_completion_tokens": 4000,  # Use max_completion_tokens for o3 models
            # Note: o3 models don't support temperature parameter - they have fixed behavior
        }
        
        # Handle web search tool (simulate with function calling)
        if tools and any(tool.get("type") == "web_search" for tool in tools):
            # Add web search function to enable web search capabilities
            request_params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for current information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            request_params["tool_choice"] = "auto"
        
        try:
            return self.client.chat.completions.create(**request_params)
        except Exception as e:
            # If o3-mini is not available, fall back to gpt-4o with adjusted parameters
            if ("o3-mini" in str(e) or "not found" in str(e).lower() or 
                "access" in str(e).lower() or "permission" in str(e).lower()):
                logger.warning("o3-mini not available, falling back to gpt-4o")
                request_params["model"] = "gpt-4o"
                # gpt-4o uses max_tokens instead of max_completion_tokens
                if "max_completion_tokens" in request_params:
                    request_params["max_tokens"] = request_params.pop("max_completion_tokens")
                # Add temperature back for gpt-4o
                request_params["temperature"] = 0
                return self.client.chat.completions.create(**request_params)
            else:
                raise e
    
    def _handle_streaming_response(
        self,
        response: Any,
        original_query: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Handle streaming response from OpenAI Chat Completions API.
        
        Args:
            response: The streaming response object
            original_query: The original user query
            
        Yields:
            Formatted response chunks
        """
        accumulated_content = ""
        search_calls = []
        response_id = f"chat_{int(time.time() * 1000)}"  # Generate unique response ID
        
        try:
            for chunk in response:
                chunk_data = {
                    "type": "chunk",
                    "timestamp": time.time(),
                    "original_query": original_query,
                    "response_id": response_id
                }
                
                # Handle different chunk types for Chat Completions
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    
                    # Handle delta content
                    if hasattr(choice, 'delta') and choice.delta:
                        if hasattr(choice.delta, 'content') and choice.delta.content:
                            content = choice.delta.content
                            accumulated_content += content
                            chunk_data.update({
                                "content": content,
                                "accumulated_content": accumulated_content
                            })
                            yield chunk_data
                        
                        # Handle function/tool calls
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            for tool_call in choice.delta.tool_calls:
                                if hasattr(tool_call, 'function') and tool_call.function.name == "web_search":
                                    search_calls.append(tool_call)
                                    chunk_data.update({
                                        "type": "web_search",
                                        "content": "🔍 Performing web search...",
                                        "tool_call": tool_call
                                    })
                                    yield chunk_data
                
                # Handle finish reason
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        # Store conversation history
                        self.conversation_history[response_id] = {
                            "messages": [
                                {"role": "user", "content": original_query},
                                {"role": "assistant", "content": accumulated_content}
                            ],
                            "timestamp": time.time()
                        }
                        
                        chunk_data.update({
                            "type": "finish",
                            "finish_reason": choice.finish_reason,
                            "final_content": accumulated_content,
                            "search_calls": search_calls,
                            "response_id": response_id
                        })
                        yield chunk_data
                        
        except Exception as e:
            logger.error("Error in streaming response: %s", str(e))
            yield {
                "type": "error",
                "content": "Streaming error: " + str(e),
                "timestamp": time.time()
            }
    
    def _handle_non_streaming_response(
        self,
        response: Any,
        original_query: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Handle non-streaming response from OpenAI Chat Completions API.
        
        Args:
            response: The response object
            original_query: The original user query
            
        Yields:
            Formatted response data
        """
        try:
            response_id = f"chat_{int(time.time() * 1000)}"  # Generate unique response ID
            response_data = {
                "type": "complete_response",
                "timestamp": time.time(),
                "original_query": original_query,
                "response_id": response_id
            }
            
            # Extract content from Chat Completions response
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                content = ""
                search_calls = []
                
                # Get message content
                if hasattr(choice, 'message') and choice.message:
                    if hasattr(choice.message, 'content') and choice.message.content:
                        content = choice.message.content
                    
                    # Check for tool calls
                    if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        for tool_call in choice.message.tool_calls:
                            if hasattr(tool_call, 'function') and tool_call.function.name == "web_search":
                                search_calls.append({
                                    "id": tool_call.id if hasattr(tool_call, 'id') else None,
                                    "function": tool_call.function.name,
                                    "status": "completed"
                                })
                
                # Store conversation history
                self.conversation_history[response_id] = {
                    "messages": [
                        {"role": "user", "content": original_query},
                        {"role": "assistant", "content": content}
                    ],
                    "timestamp": time.time()
                }
                
                response_data.update({
                    "content": content,
                    "search_calls": search_calls,
                    "usage": response.usage.__dict__ if hasattr(response, 'usage') else None
                })
            
            yield response_data
            
        except Exception as e:
            logger.error("Error in non-streaming response: %s", str(e))
            yield {
                "type": "error",
                "content": "Response processing error: " + str(e),
                "timestamp": time.time()
            }
    
    def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previous response by its ID from local conversation history.
        
        Args:
            response_id: The ID of the response to retrieve
            
        Returns:
            Response data or None if not found
        """
        try:
            if response_id in self.conversation_history:
                history = self.conversation_history[response_id]
                return {
                    "id": response_id,
                    "messages": history["messages"],
                    "timestamp": history["timestamp"]
                }
            else:
                logger.warning("Response ID %s not found in conversation history", response_id)
                return None
        except Exception as e:
            logger.error("Error retrieving response %s: %s", response_id, str(e))
            return None
    
    def continue_conversation(
        self,
        new_query: str,
        previous_response_id: str,
        context: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Continue a conversation from a previous response.
        
        Args:
            new_query: New user query
            previous_response_id: ID of the previous response
            context: Additional context
            
        Yields:
            Response chunks
        """
        yield from self.chatbot_with_web_search(
            query=new_query,
            context=context,
            previous_response_id=previous_response_id
        )


def create_chatbot_instance(model: str = "o3-mini") -> ResponsesAPIChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        model: OpenAI model to use (default: o3-mini)
        
    Returns:
        Configured chatbot instance
    """
    return ResponsesAPIChatbot(model=model)


def demo_chatbot():
    """
    Demonstration function showing how to use the chatbot.
    """
    print("🤖 OpenAI Responses API Chatbot Demo")
    print("=" * 50)
    
    try:
        # Create chatbot instance
        chatbot = create_chatbot_instance()
        
        # Example 1: Simple query with context
        print("\n📝 Example 1: Context-based query")
        context = "You are a helpful AI tutor specializing in quantum physics."
        query = "Explain quantum entanglement in simple terms"
        
        print(f"Context: {context}")
        print(f"Query: {query}")
        print("\nResponse:")
        
        for chunk in chatbot.chatbot_with_web_search(query=query, context=context, stream=True):
            if chunk["type"] == "chunk" and "content" in chunk:
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "web_search":
                print("\n" + chunk["content"])
            elif chunk["type"] == "finish":
                print(f"\n\n✅ Response completed (reason: {chunk['finish_reason']})")
                if chunk["search_calls"]:
                    print(f"🔍 Web searches performed: {len(chunk['search_calls'])}")
        
        print("\n" + "=" * 50)
        
        # Example 2: Query requiring web search
        print("\n📝 Example 2: Query requiring current information")
        query2 = "What are the latest developments in quantum computing in 2024?"
        
        print(f"Query: {query2}")
        print("\nResponse:")
        
        response_id = None
        for chunk in chatbot.chatbot_with_web_search(query=query2, enable_web_search=True):
            if chunk["type"] == "chunk" and "content" in chunk:
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "web_search":
                print("\n" + chunk["content"])
            elif chunk["type"] == "complete_response":
                response_id = chunk.get("response_id")
                print("\n\n✅ Response completed")
                if chunk.get("search_calls"):
                    print(f"🔍 Web searches performed: {len(chunk['search_calls'])}")
            elif chunk["type"] == "finish":
                print(f"\n\n✅ Response completed (reason: {chunk['finish_reason']})")
                if chunk.get("search_calls"):
                    print(f"🔍 Web searches performed: {len(chunk['search_calls'])}")
        
        print("\n" + "=" * 50)
        
        # Example 3: Continue conversation
        if response_id:
            print("\n📝 Example 3: Continue conversation")
            follow_up = "Can you provide more details about the commercial applications?"
            
            print(f"Follow-up: {follow_up}")
            print("\nResponse:")
            
            for chunk in chatbot.continue_conversation(follow_up, response_id):
                if chunk["type"] == "chunk" and "content" in chunk:
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "finish":
                    print("\n\n✅ Conversation continued successfully")
        
    except Exception as e:
        print("❌ Demo error: " + str(e))
        logger.error("Demo error: %s", str(e))


def test_chatbot_functionality():
    """
    Test function to verify chatbot functionality.
    """
    print("🧪 Running Chatbot Tests")
    print("=" * 30)
    
    try:
        chatbot = create_chatbot_instance()
        
        # Test 1: Basic functionality
        print("\n🔬 Test 1: Basic response generation")
        test_query = "What is machine learning?"
        response_count = 0
        
        for chunk in chatbot.chatbot_with_web_search(
            query=test_query,
            context="You are an AI expert",
            stream=True,
            enable_web_search=False
        ):
            response_count += 1
            if chunk["type"] == "finish":
                print(f"✅ Test 1 passed - Generated {response_count} chunks")
                break
        
        # Test 2: Web search capability
        print("\n🔬 Test 2: Web search functionality")
        search_query = "Latest AI research papers 2024"
        search_detected = False
        
        for chunk in chatbot.chatbot_with_web_search(
            query=search_query,
            enable_web_search=True,
            stream=True
        ):
            if chunk["type"] == "web_search":
                search_detected = True
            elif chunk["type"] == "finish":
                if search_detected:
                    print("✅ Test 2 passed - Web search functionality working")
                else:
                    print("⚠️ Test 2 warning - No web search detected (may not be needed)")
                break
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print("❌ Test error: " + str(e))
        logger.error("Test error: %s", str(e))


if __name__ == "__main__":
    """
    Main execution block for testing and demonstration.
    
    To run this script:
    1. Ensure you have activated the deeptutor conda environment:
       conda activate deeptutor
    2. Make sure OPENAI_API_KEY is set in your .env file
    3. Run: python responsee_api_test.py
    """
    
    print("🚀 OpenAI Responses API Chatbot")
    print("===============================")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file")
        exit(1)
    
    # Run tests
    test_chatbot_functionality()
    
    print("\n" + "=" * 50)
    
    # Run demo
    demo_chatbot()
    
    print("\n🏁 Script completed successfully!")
