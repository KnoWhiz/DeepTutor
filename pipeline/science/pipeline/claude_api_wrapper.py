#!/usr/bin/env python3
"""
Claude API Wrapper for direct Claude API usage with proxy conversion to Azure OpenAI.
This allows you to use Claude API format in your code while the proxy converts to Azure OpenAI.
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger("tutorpipeline.science.claude_api_wrapper")


@dataclass
class ClaudeMessage:
    """Claude API message format."""
    role: str  # "user", "assistant", "system"
    content: Union[str, List[Dict[str, Any]]]


@dataclass
class ClaudeRequest:
    """Claude API request format."""
    model: str
    messages: List[ClaudeMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    stream: bool = False


@dataclass
class ClaudeResponse:
    """Claude API response format."""
    id: str
    type: str
    role: str
    model: str
    content: List[Dict[str, Any]]
    stop_reason: str
    usage: Dict[str, int]
    
    @property
    def text(self) -> str:
        """Extract text content from response."""
        for block in self.content:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""
    
    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Extract tool calls from response."""
        tool_calls = []
        for block in self.content:
            if block.get("type") == "tool_use":
                tool_calls.append(block)
        return tool_calls


class ClaudeAPIWrapper:
    """Wrapper for using Claude API format with proxy conversion to Azure OpenAI."""
    
    def __init__(self, proxy_url: str = "http://localhost:8082", api_key: str = "your-key"):
        self.proxy_url = proxy_url.rstrip('/')
        self.api_key = api_key
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
    
    def _convert_messages_to_dict(self, messages: List[ClaudeMessage]) -> List[Dict[str, Any]]:
        """Convert ClaudeMessage objects to dictionary format."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    
    async def chat_completion(
        self,
        request: ClaudeRequest
    ) -> ClaudeResponse:
        """Send a Claude API format request through the proxy."""
        try:
            # Convert request to Claude API format
            request_data = {
                "model": request.model,
                "messages": self._convert_messages_to_dict(request.messages),
            }
            
            if request.max_tokens is not None:
                request_data["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                request_data["temperature"] = request.temperature
            if request.system is not None:
                request_data["system"] = request.system
            if request.tools is not None:
                request_data["tools"] = request.tools
            if request.tool_choice is not None:
                request_data["tool_choice"] = request.tool_choice
            if request.stream:
                request_data["stream"] = True
            
            # Send to proxy (which converts to Azure OpenAI format)
            response = await self.client.post(
                f"{self.proxy_url}/v1/messages",
                headers=self._get_headers(),
                json=request_data
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            data = response.json()
            return ClaudeResponse(**data)
            
        except Exception as e:
            logger.error(f"Error in chat_completion: {str(e)}")
            raise
    
    async def streaming_chat(
        self,
        request: ClaudeRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a streaming Claude API format request through the proxy."""
        try:
            # Convert request to Claude API format
            request_data = {
                "model": request.model,
                "messages": self._convert_messages_to_dict(request.messages),
                "stream": True
            }
            
            if request.max_tokens is not None:
                request_data["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                request_data["temperature"] = request.temperature
            if request.system is not None:
                request_data["system"] = request.system
            if request.tools is not None:
                request_data["tools"] = request.tools
            if request.tool_choice is not None:
                request_data["tool_choice"] = request.tool_choice
            
            # Send to proxy (which converts to Azure OpenAI format)
            async with self.client.stream(
                "POST",
                f"{self.proxy_url}/v1/messages",
                headers=self._get_headers(),
                json=request_data
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Streaming API request failed: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.strip():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data.strip() == "[DONE]":
                                yield {"type": "done"}
                            else:
                                try:
                                    yield json.loads(data)
                                except json.JSONDecodeError:
                                    continue
                            
        except Exception as e:
            logger.error(f"Error in streaming_chat: {str(e)}")
            raise


# Convenience functions for easy usage
def create_claude_message(role: str, content: Union[str, List[Dict[str, Any]]]) -> ClaudeMessage:
    """Create a Claude message."""
    return ClaudeMessage(role=role, content=content)


def create_claude_request(
    model: str,
    messages: List[ClaudeMessage],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Dict[str, Any]] = None,
    stream: bool = False
) -> ClaudeRequest:
    """Create a Claude request."""
    return ClaudeRequest(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream
    )


# Example usage functions
async def example_basic_chat():
    """Example of basic chat using Claude API format."""
    async with ClaudeAPIWrapper() as client:
        # Create Claude API format messages
        messages = [
            create_claude_message("user", "Hello, how are you?")
        ]
        
        # Create Claude API format request
        request = create_claude_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            max_tokens=100
        )
        
        # Send request (proxy converts to Azure OpenAI format)
        response = await client.chat_completion(request)
        print(f"Response: {response.text}")


async def example_streaming_chat():
    """Example of streaming chat using Claude API format."""
    async with ClaudeAPIWrapper() as client:
        messages = [
            create_claude_message("user", "Tell me a story")
        ]
        
        request = create_claude_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            max_tokens=200,
            stream=True
        )
        
        print("Streaming response:")
        async for chunk in client.streaming_chat(request):
            if chunk.get("type") == "content_block_delta":
                content = chunk.get("delta", {}).get("text", "")
                if content:
                    print(content, end="", flush=True)


async def example_with_system_message():
    """Example with system message using Claude API format."""
    async with ClaudeAPIWrapper() as client:
        messages = [
            create_claude_message("user", "Explain quantum computing")
        ]
        
        request = create_claude_request(
            model="claude-3-5-opus-20241022",
            messages=messages,
            system="You are a helpful physics professor. Explain concepts clearly and simply.",
            max_tokens=300
        )
        
        response = await client.chat_completion(request)
        print(f"Response: {response.text}")


async def example_function_calling():
    """Example of function calling using Claude API format."""
    async with ClaudeAPIWrapper() as client:
        messages = [
            create_claude_message("user", "What's the weather like in New York?")
        ]
        
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        request = create_claude_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            max_tokens=200
        )
        
        response = await client.chat_completion(request)
        print(f"Response: {response.text}")
        print(f"Tool calls: {response.tool_calls}")


if __name__ == "__main__":
    # Run examples
    async def main():
        print("=== Basic Chat Example ===")
        await example_basic_chat()
        
        print("\n=== Streaming Chat Example ===")
        await example_streaming_chat()
        
        print("\n=== System Message Example ===")
        await example_with_system_message()
        
        print("\n=== Function Calling Example ===")
        await example_function_calling()
    
    asyncio.run(main()) 