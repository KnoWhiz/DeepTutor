"""
Anthropic File API Test Module

This module provides functionality to interact with Anthropic's API for file-based 
question answering with chat history support.

Author: AI Assistant
Date: 2024
"""

import os
import base64
import mimetypes
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import pytest
import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AnthropicFileHandler:
    """
    Handler class for Anthropic API interactions with file processing capabilities.
    
    This class manages file uploads, chat history, and provides question-answering
    functionality using Claude models.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the AnthropicFileHandler.
        
        Args:
            api_key: Optional API key. If not provided, will use environment variable.
            
        Raises:
            ValueError: If no API key is found.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or ANTHROPIC_API_KEY environment variable")
        
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Using the latest model
        self.max_tokens = 4096
    
    def _get_file_mime_type(self, file_path: str) -> str:
        """
        Determine the MIME type of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"
    
    def _encode_file_to_base64(self, file_path: str) -> str:
        """
        Encode a file to base64 string.
        
        Args:
            file_path: Path to the file to encode
            
        Returns:
            Base64 encoded string of the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
        except IOError as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")
    
    def _prepare_file_content(self, file_path: str) -> Dict[str, Any]:
        """
        Prepare file content for Anthropic API consumption.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file content prepared for API
        """
        mime_type = self._get_file_mime_type(file_path)
        base64_data = self._encode_file_to_base64(file_path)
        
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_data
            }
        }
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> List[MessageParam]:
        """
        Format chat history for Anthropic API.
        
        Args:
            chat_history: List of chat messages with 'role' and 'content' keys
            
        Returns:
            Formatted messages for API consumption
            
        Raises:
            ValueError: If chat history format is invalid
        """
        formatted_messages: List[MessageParam] = []
        
        for message in chat_history:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Each chat history item must be a dict with 'role' and 'content' keys")
            
            if message["role"] not in ["user", "assistant"]:
                raise ValueError("Message role must be either 'user' or 'assistant'")
            
            formatted_messages.append({
                "role": message["role"],  # type: ignore
                "content": message["content"]
            })
        
        return formatted_messages
    
    def process_file_with_question(
        self, 
        file_path: str, 
        question: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a file with a question and optional chat history using Anthropic API.
        
        This is the core function that:
        1. Uploads the file to Anthropic
        2. Processes the chat history
        3. Asks the question about the file
        4. Returns the response
        
        Args:
            file_path: Path to the file to process
            question: Question to ask about the file
            chat_history: Optional list of previous chat messages
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If inputs are invalid
            anthropic.APIError: If there's an API error
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Prepare file content
            file_content = self._prepare_file_content(file_path)
            
            # Format chat history
            messages = []
            if chat_history:
                messages = self._format_chat_history(chat_history)
            
            # Prepare the current user message with file and question
            current_message_content = [
                file_content,
                {
                    "type": "text",
                    "text": question
                }
            ]
            
            # Add current message to the conversation
            messages.append({
                "role": "user",
                "content": current_message_content
            })
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )
            
            # Extract response text
            response_text = ""
            if response.content:
                for content_block in response.content:
                    if hasattr(content_block, "text"):
                        response_text += content_block.text
            
            return {
                "response": response_text,
                "model_used": self.model,
                "tokens_used": {
                    "input": response.usage.input_tokens if response.usage else 0,
                    "output": response.usage.output_tokens if response.usage else 0
                },
                "file_processed": os.path.basename(file_path),
                "success": True
            }
            
        except anthropic.APIError as e:
            return {
                "response": "",
                "error": f"Anthropic API Error: {str(e)}",
                "success": False
            }
        except Exception as e:
            return {
                "response": "",
                "error": f"Unexpected error: {str(e)}",
                "success": False
            }


class TestAnthropicFileHandler:
    """
    Comprehensive test suite for AnthropicFileHandler.
    """
    
    @pytest.fixture
    def handler(self) -> AnthropicFileHandler:
        """Create a test handler instance."""
        return AnthropicFileHandler()
    
    @pytest.fixture
    def test_file_path(self) -> str:
        """Path to test PDF file."""
        return "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/67f1e8e529b9934088beb36f.pdf"
    
    def test_handler_initialization_success(self):
        """Test successful handler initialization with API key."""
        handler = AnthropicFileHandler(api_key="test-key")
        assert handler.api_key == "test-key"
        assert handler.model == "claude-3-5-sonnet-20241022"
        assert handler.max_tokens == 4096
    
    def test_handler_initialization_no_key(self):
        """Test handler initialization failure without API key."""
        # Clear environment variable for this test
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                AnthropicFileHandler()
        finally:
            # Restore original environment
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key
    
    def test_get_file_mime_type(self, handler: AnthropicFileHandler):
        """Test MIME type detection."""
        # Test PDF file
        assert "application/pdf" in handler._get_file_mime_type("test.pdf")
        
        # Test unknown extension
        assert handler._get_file_mime_type("test.unknown") == "application/octet-stream"
    
    def test_encode_file_to_base64_nonexistent(self, handler: AnthropicFileHandler):
        """Test base64 encoding with non-existent file."""
        with pytest.raises(FileNotFoundError):
            handler._encode_file_to_base64("nonexistent_file.pdf")
    
    def test_format_chat_history_valid(self, handler: AnthropicFileHandler):
        """Test chat history formatting with valid input."""
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = handler._format_chat_history(chat_history)
        assert len(formatted) == 3
        assert all(msg["role"] in ["user", "assistant"] for msg in formatted)
        assert all("content" in msg for msg in formatted)
    
    def test_format_chat_history_invalid_role(self, handler: AnthropicFileHandler):
        """Test chat history formatting with invalid role."""
        chat_history = [
            {"role": "invalid_role", "content": "Hello"}
        ]
        
        with pytest.raises(ValueError, match="Message role must be either"):
            handler._format_chat_history(chat_history)
    
    def test_format_chat_history_missing_keys(self, handler: AnthropicFileHandler):
        """Test chat history formatting with missing keys."""
        chat_history = [
            {"role": "user"}  # Missing content
        ]
        
        with pytest.raises(ValueError, match="must be a dict with 'role' and 'content' keys"):
            handler._format_chat_history(chat_history)
    
    def test_process_file_with_question_empty_question(self, handler: AnthropicFileHandler, test_file_path: str):
        """Test processing with empty question."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            handler.process_file_with_question(test_file_path, "")
    
    def test_process_file_with_question_nonexistent_file(self, handler: AnthropicFileHandler):
        """Test processing with non-existent file."""
        with pytest.raises(FileNotFoundError):
            handler.process_file_with_question("nonexistent.pdf", "What is this about?")
    
    def test_process_file_with_question_basic(self, handler: AnthropicFileHandler, test_file_path: str):
        """Test basic file processing with question."""
        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file not found: {test_file_path}")
        
        result = handler.process_file_with_question(
            test_file_path,
            "What is the main topic of this document?"
        )
        
        # Check response structure
        assert "response" in result
        assert "success" in result
        assert "model_used" in result
        assert "tokens_used" in result
        assert "file_processed" in result
        
        # If successful, response should not be empty
        if result["success"]:
            assert result["response"].strip() != ""
            assert result["model_used"] == "claude-3-5-sonnet-20241022"
            assert "input" in result["tokens_used"]
            assert "output" in result["tokens_used"]
    
    def test_process_file_with_chat_history(self, handler: AnthropicFileHandler, test_file_path: str):
        """Test file processing with chat history."""
        if not os.path.exists(test_file_path):
            pytest.skip(f"Test file not found: {test_file_path}")
        
        chat_history = [
            {"role": "user", "content": "Hello, I have a document to analyze."},
            {"role": "assistant", "content": "I'd be happy to help you analyze your document. Please share it and let me know what you'd like to know about it."}
        ]
        
        result = handler.process_file_with_question(
            test_file_path,
            "Based on our previous conversation, can you summarize the key points of this document?",
            chat_history
        )
        
        # Check response structure
        assert "response" in result
        assert "success" in result
        
        # If successful, response should acknowledge the context
        if result["success"]:
            assert result["response"].strip() != ""


def demo_usage() -> None:
    """
    Demonstration of how to use the AnthropicFileHandler.
    """
    # Initialize handler
    handler = AnthropicFileHandler()
    
    # Example file path
    file_path = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/67f1e8e529b9934088beb36f.pdf"
    
    # Example 1: Simple question without chat history
    print("=== Example 1: Simple Question ===")
    result1 = handler.process_file_with_question(
        file_path,
        "What is the main topic or subject of this document?"
    )
    
    if result1["success"]:
        print(f"Response: {result1['response'][:200]}...")
        print(f"Tokens used: {result1['tokens_used']}")
    else:
        print(f"Error: {result1.get('error', 'Unknown error')}")
    
    # Example 2: Question with chat history
    print("\n=== Example 2: With Chat History ===")
    chat_history = [
        {"role": "user", "content": "I'm a student studying machine learning."},
        {"role": "assistant", "content": "That's great! I'd be happy to help you with machine learning concepts. What would you like to learn about?"}
    ]
    
    result2 = handler.process_file_with_question(
        file_path,
        "Given that I'm studying ML, can you explain any machine learning concepts mentioned in this document in simple terms?",
        chat_history
    )
    
    if result2["success"]:
        print(f"Response: {result2['response'][:200]}...")
        print(f"Tokens used: {result2['tokens_used']}")
    else:
        print(f"Error: {result2.get('error', 'Unknown error')}")
    
    # Example 3: Follow-up question
    print("\n=== Example 3: Follow-up Question ===")
    extended_history = chat_history + [
        {"role": "user", "content": "Given that I'm studying ML, can you explain any machine learning concepts mentioned in this document in simple terms?"},
        {"role": "assistant", "content": result2["response"] if result2["success"] else "I couldn't process the previous question."}
    ]
    
    result3 = handler.process_file_with_question(
        file_path,
        "Can you provide some specific examples or practical applications of the concepts you just explained?",
        extended_history
    )
    
    if result3["success"]:
        print(f"Response: {result3['response'][:200]}...")
        print(f"Tokens used: {result3['tokens_used']}")
    else:
        print(f"Error: {result3.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Run the demo (API key loaded from .env file)
    demo_usage()
