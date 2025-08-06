#!/usr/bin/env python3
"""
Example of integrating Claude API format directly into your existing codebase.
This shows how to use Claude API format while the proxy converts to Azure OpenAI.
"""

import asyncio
import os
from claude_api_wrapper import (
    ClaudeAPIWrapper, 
    create_claude_message, 
    create_claude_request
)


class ClaudeAPIService:
    """Service class for using Claude API format with proxy conversion."""
    
    def __init__(self, proxy_url: str = "http://localhost:8082", api_key: str = "your-key"):
        self.proxy_url = proxy_url
        self.api_key = api_key
    
    async def chat_completion(
        self,
        messages: list,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system: str = None,
        stream: bool = False
    ):
        """Send a chat completion request using Claude API format."""
        async with ClaudeAPIWrapper(self.proxy_url, self.api_key) as client:
            # Convert messages to Claude format
            claude_messages = [
                create_claude_message(msg["role"], msg["content"])
                for msg in messages
            ]
            
            # Create Claude request
            request = create_claude_request(
                model=model,
                messages=claude_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                stream=stream
            )
            
            if stream:
                return client.streaming_chat(request)
            else:
                response = await client.chat_completion(request)
                return response.text


# Example: Replace your existing API handler calls
async def replace_api_handler_usage():
    """Example of replacing API handler usage with Claude API format."""
    
    # Initialize Claude API service
    claude_service = ClaudeAPIService()
    
    # Example 1: Basic chat (replaces API handler basic model)
    print("=== Basic Chat (replaces API handler basic model) ===")
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    response = await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-sonnet-20241022",  # Maps to your Azure deployment
        max_tokens=100
    )
    print(f"Response: {response}")
    
    # Example 2: Advanced chat (replaces API handler advanced model)
    print("\n=== Advanced Chat (replaces API handler advanced model) ===")
    messages = [
        {"role": "user", "content": "Explain quantum computing in detail"}
    ]
    
    response = await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-opus-20241022",  # Maps to your Azure deployment
        max_tokens=300,
        system="You are a helpful physics professor. Explain concepts clearly."
    )
    print(f"Response: {response}")
    
    # Example 3: Streaming (replaces API handler streaming)
    print("\n=== Streaming Chat ===")
    messages = [
        {"role": "user", "content": "Tell me a story about a robot"}
    ]
    
    async for chunk in await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        stream=True
    ):
        if chunk.get("type") == "content_block_delta":
            content = chunk.get("delta", {}).get("text", "")
            if content:
                print(content, end="", flush=True)
    print()


# Example: Integration with your existing pipeline
async def integrate_with_existing_pipeline():
    """Example of integrating with your existing pipeline code."""
    
    claude_service = ClaudeAPIService()
    
    # This could replace calls in your existing files like:
    # - get_response.py
    # - get_rag_response.py
    # - inference.py
    # etc.
    
    # Example: Replace a call in get_response.py
    print("=== Integration with Existing Pipeline ===")
    
    # Simulate a call that might be in get_response.py
    user_input = "What is machine learning?"
    system_prompt = "You are a helpful AI assistant."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    response = await claude_service.chat_completion(
        messages=messages,
        model="claude-3-5-sonnet-20241022",
        max_tokens=200
    )
    
    print(f"User: {user_input}")
    print(f"Assistant: {response}")


# Example: Function calling (replaces OpenAI function calling)
async def function_calling_example():
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


# Example: Replace specific API handler methods
class ClaudeAPIHandler:
    """Replacement for your existing API handler using Claude API format."""
    
    def __init__(self, proxy_url: str = "http://localhost:8082", api_key: str = "your-key"):
        self.claude_service = ClaudeAPIService(proxy_url, api_key)
    
    async def basic_model_invoke(self, prompt: str) -> str:
        """Replace API handler basic model invocation."""
        messages = [{"role": "user", "content": prompt}]
        return await self.claude_service.chat_completion(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000
        )
    
    async def advanced_model_invoke(self, prompt: str) -> str:
        """Replace API handler advanced model invocation."""
        messages = [{"role": "user", "content": prompt}]
        return await self.claude_service.chat_completion(
            messages=messages,
            model="claude-3-5-opus-20241022",
            max_tokens=2000
        )
    
    async def creative_model_invoke(self, prompt: str, temperature: float = 0.9) -> str:
        """Replace API handler creative model invocation."""
        messages = [{"role": "user", "content": prompt}]
        return await self.claude_service.chat_completion(
            messages=messages,
            model="claude-3-5-opus-20241022",
            max_tokens=1500,
            temperature=temperature
        )
    
    async def streaming_invoke(self, prompt: str, model: str = "claude-3-5-sonnet-20241022"):
        """Replace API handler streaming invocation."""
        messages = [{"role": "user", "content": prompt}]
        return await self.claude_service.chat_completion(
            messages=messages,
            model=model,
            max_tokens=1000,
            stream=True
        )


async def test_claude_api_handler():
    """Test the Claude API handler replacement."""
    
    handler = ClaudeAPIHandler()
    
    print("=== Testing Claude API Handler ===")
    
    # Test basic model
    response = await handler.basic_model_invoke("What is 2+2?")
    print(f"Basic model: {response}")
    
    # Test advanced model
    response = await handler.advanced_model_invoke("Explain the concept of derivatives in calculus")
    print(f"Advanced model: {response}")
    
    # Test creative model
    response = await handler.creative_model_invoke("Write a short creative story about a robot")
    print(f"Creative model: {response}")
    
    # Test streaming
    print("Streaming response:")
    async for chunk in await handler.streaming_invoke("Count from 1 to 5"):
        if chunk.get("type") == "content_block_delta":
            content = chunk.get("delta", {}).get("text", "")
            if content:
                print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    async def main():
        print("🚀 Claude API Integration Examples")
        print("="*50)
        
        # Test basic replacement
        await replace_api_handler_usage()
        
        # Test pipeline integration
        await integrate_with_existing_pipeline()
        
        # Test function calling
        await function_calling_example()
        
        # Test API handler replacement
        await test_claude_api_handler()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo use in your code:")
        print("1. Replace API handler calls with ClaudeAPIHandler")
        print("2. Use Claude API format for requests")
        print("3. Proxy converts to Azure OpenAI format automatically")
    
    asyncio.run(main()) 