#!/usr/bin/env python3
"""
Claude Client Wrapper for using Anthropic Claude client format with proxy conversion to Azure OpenAI.
This allows you to use the same client.messages.create() format while the proxy converts to Azure OpenAI.
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger("tutorpipeline.science.claude_client_wrapper")


class ClaudeClientWrapper:
    """Wrapper that mimics the Anthropic Claude client but uses proxy conversion to Azure OpenAI."""
    
    def __init__(self, api_key: str = "your-key", proxy_url: str = "http://localhost:8082"):
        self.api_key = api_key
        self.proxy_url = proxy_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=90.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "your-key":
            headers["x-api-key"] = self.api_key
        return headers
    
    class Messages:
        """Mimics the Anthropic client.messages interface."""
        
        def __init__(self, parent_client):
            self.parent = parent_client
        
        async def create(
            self,
            model: str,
            messages: List[Dict[str, Any]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            system: Optional[str] = None,
            stream: bool = False,
            **kwargs
        ):
            """Create a message using Claude API format through proxy."""
            try:
                # Prepare request data in Claude API format
                request_data = {
                    "model": model,
                    "messages": messages,
                }
                
                if max_tokens is not None:
                    request_data["max_tokens"] = max_tokens
                if temperature is not None:
                    request_data["temperature"] = temperature
                if system is not None:
                    request_data["system"] = system
                if stream:
                    request_data["stream"] = True
                
                # Add any additional kwargs
                request_data.update(kwargs)
                
                # Send to proxy (which converts to Azure OpenAI format)
                response = await self.parent.client.post(
                    f"{self.parent.proxy_url}/v1/messages",
                    headers=self.parent._get_headers(),
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.status_code} - {response.text}")
                
                data = response.json()
                
                # Return a response object that mimics Anthropic's response
                return ClaudeResponseWrapper(data)
                
            except Exception as e:
                logger.error(f"Error in messages.create: {str(e)}")
                raise
    
    @property
    def messages(self):
        """Return the messages interface."""
        return self.Messages(self)


class ClaudeResponseWrapper:
    """Wrapper for Claude API response that mimics Anthropic's response format."""
    
    def __init__(self, response_data: Dict[str, Any]):
        self.data = response_data
        self.id = response_data.get("id", "")
        self.type = response_data.get("type", "")
        self.role = response_data.get("role", "assistant")
        self.model = response_data.get("model", "")
        self.content = response_data.get("content", [])
        self.stop_reason = response_data.get("stop_reason", "")
        self.usage = response_data.get("usage", {})
    
    @property
    def text(self) -> str:
        """Get the text content from the response."""
        for block in self.content:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""
    
    def __getattr__(self, name):
        """Delegate to the underlying data."""
        return getattr(self.data, name)


# Convenience function to create a client
def create_claude_client(api_key: str = None, proxy_url: str = "http://localhost:8082") -> ClaudeClientWrapper:
    """Create a Claude client wrapper."""
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "your-key")
    
    return ClaudeClientWrapper(api_key=api_key, proxy_url=proxy_url)


# Example usage functions
async def example_basic_chat():
    """Example of basic chat using Claude client format."""
    async with create_claude_client() as client:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0.3,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print(f"Response: {response.text}")


async def example_with_conversation_history():
    """Example with conversation history using Claude client format."""
    async with create_claude_client() as client:
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"role": "user", "content": "Can you give me an example?"}
        ]
        
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.3,
            messages=messages
        )
        print(f"Response: {response.text}")


async def example_streaming():
    """Example of streaming using Claude client format."""
    async with create_claude_client() as client:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            temperature=0.3,
            stream=True,
            messages=[
                {"role": "user", "content": "Tell me a story about a robot"}
            ]
        )
        
        print("Streaming response:")
        async for chunk in response:
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)


# Example: Replace your existing Claude client usage
async def replace_existing_claude_usage():
    """Example of replacing existing Claude client usage."""
    
    # Old way (direct Anthropic client)
    # from anthropic import Anthropic
    # client = Anthropic(api_key="your-key")
    # response = client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=4000,
    #     temperature=0.3,
    #     system=system_message,
    #     messages=messages
    # )
    # return response.content[0].text
    
    # New way (with proxy conversion to Azure OpenAI)
    async with create_claude_client() as client:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        return response.text


if __name__ == "__main__":
    # Run examples
    async def main():
        print("🚀 Claude Client Wrapper Examples")
        print("="*50)
        
        print("=== Basic Chat Example ===")
        await example_basic_chat()
        
        print("\n=== Conversation History Example ===")
        await example_with_conversation_history()
        
        print("\n=== Replace Existing Usage Example ===")
        result = await replace_existing_claude_usage()
        print(f"Result: {result}")
        
        print("\n✅ All examples completed successfully!")
        print("\nTo use in your code:")
        print("1. Replace 'from anthropic import Anthropic' with 'from claude_client_wrapper import create_claude_client'")
        print("2. Use the same client.messages.create() format")
        print("3. Proxy converts to Azure OpenAI format automatically")
    
    asyncio.run(main()) 