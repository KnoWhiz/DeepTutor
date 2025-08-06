import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Iterator, Union
from dataclasses import dataclass
from enum import Enum
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.pydantic_v1 import Field, root_validator

logger = logging.getLogger("tutorpipeline.science.claude_proxy_wrapper")


class ModelType(Enum):
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-5-sonnet-20241022"
    OPUS = "claude-3-5-opus-20241022"


@dataclass
class ClaudeResponse:
    """Structured response from Claude proxy."""
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


class ClaudeProxyError(Exception):
    """Custom exception for Claude proxy errors."""
    pass


class ClaudeProxyClient:
    """Client for Claude proxy with LangChain compatibility."""
    
    def __init__(self, base_url: str = "http://localhost:8082", api_key: str = "your-key"):
        self.base_url = base_url.rstrip('/')
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
    
    def _parse_streaming_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a streaming SSE line."""
        if not line.startswith("data: "):
            return None
        
        try:
            data = line[6:]  # Remove "data: " prefix
            if data.strip() == "[DONE]":
                return {"type": "done"}
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        model: str = ModelType.SONNET.value,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ClaudeResponse:
        """Send a chat completion request."""
        try:
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            if system:
                request_data["system"] = system
            if tools:
                request_data["tools"] = tools
            if tool_choice:
                request_data["tool_choice"] = tool_choice
            
            # Add any additional kwargs
            request_data.update(kwargs)
            
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                headers=self._get_headers(),
                json=request_data
            )
            
            if response.status_code != 200:
                raise ClaudeProxyError(f"API request failed: {response.status_code} - {response.text}")
            
            data = response.json()
            return ClaudeResponse(**data)
            
        except Exception as e:
            logger.error(f"Error in chat_completion: {str(e)}")
            raise ClaudeProxyError(f"Chat completion failed: {str(e)}")
    
    async def streaming_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: str = ModelType.SONNET.value,
        max_tokens: int = 1000,
        temperature: float = 1.0,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a streaming chat completion request."""
        try:
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            if system:
                request_data["system"] = system
            if tools:
                request_data["tools"] = tools
            if tool_choice:
                request_data["tool_choice"] = tool_choice
            
            # Add any additional kwargs
            request_data.update(kwargs)
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=self._get_headers(),
                json=request_data
            ) as response:
                if response.status_code != 200:
                    raise ClaudeProxyError(f"Streaming API request failed: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.strip():
                        parsed = self._parse_streaming_line(line)
                        if parsed:
                            yield parsed
                            
        except Exception as e:
            logger.error(f"Error in streaming_chat: {str(e)}")
            raise ClaudeProxyError(f"Streaming chat failed: {str(e)}")


class ClaudeProxyChatModel(BaseChatModel):
    """LangChain-compatible wrapper for Claude proxy."""
    
    base_url: str = Field(default="http://localhost:8082")
    api_key: str = Field(default="your-key")
    model_name: str = Field(default="claude-3-5-sonnet-20241022")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=20000)
    streaming: bool = Field(default=False)
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is properly set up."""
        return values
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "claude_proxy"
    
    def _convert_messages_to_claude_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Claude format."""
        claude_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                # System messages are handled separately in Claude API
                continue
            elif isinstance(message, HumanMessage):
                claude_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                claude_messages.append({
                    "role": "assistant", 
                    "content": message.content
                })
        return claude_messages
    
    def _extract_system_message(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract system message from LangChain messages."""
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return None
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        try:
            client = ClaudeProxyClient(base_url=self.base_url, api_key=self.api_key)
            
            claude_messages = self._convert_messages_to_claude_format(messages)
            system_message = self._extract_system_message(messages)
            
            if self.streaming:
                # Handle streaming
                chunks = []
                async for chunk in client.streaming_chat(
                    messages=claude_messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_message,
                    **kwargs
                ):
                    if chunk.get("type") == "content_block_delta":
                        content = chunk.get("delta", {}).get("text", "")
                        if content:
                            chunk_obj = ChatGenerationChunk(
                                message=AIMessage(content=content),
                                generation_info=chunk
                            )
                            chunks.append(chunk_obj)
                            if run_manager:
                                await run_manager.on_llm_new_token(content)
                
                # Combine all chunks
                full_content = "".join([chunk.message.content for chunk in chunks])
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=full_content))]
                )
            else:
                # Handle non-streaming
                response = await client.chat_completion(
                    messages=claude_messages,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_message,
                    **kwargs
                )
                
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=response.text))]
                )
                
        except Exception as e:
            logger.error(f"Error in Claude proxy generation: {str(e)}")
            raise
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously."""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """Stream chat response asynchronously."""
        try:
            client = ClaudeProxyClient(base_url=self.base_url, api_key=self.api_key)
            
            claude_messages = self._convert_messages_to_claude_format(messages)
            system_message = self._extract_system_message(messages)
            
            async for chunk in client.streaming_chat(
                messages=claude_messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                **kwargs
            ):
                if chunk.get("type") == "content_block_delta":
                    content = chunk.get("delta", {}).get("text", "")
                    if content:
                        yield ChatGenerationChunk(
                            message=AIMessage(content=content),
                            generation_info=chunk
                        )
                        if run_manager:
                            await run_manager.on_llm_new_token(content)
                            
        except Exception as e:
            logger.error(f"Error in Claude proxy streaming: {str(e)}")
            raise


def create_claude_proxy_model(
    base_url: str = "http://localhost:8082",
    api_key: str = "your-key",
    model_name: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    max_tokens: int = 20000,
    streaming: bool = False
) -> ClaudeProxyChatModel:
    """Factory function to create a Claude proxy model."""
    return ClaudeProxyChatModel(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming
    ) 