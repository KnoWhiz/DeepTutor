"""Claude Code SDK integration for DeepTutor.

This module provides a chatbot implementation using Claude's API with code analysis capabilities.
It maintains compatibility with the existing get_response function interface.
"""

import os
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


@dataclass
class ChatSession:
    """Simplified ChatSession class for compatibility."""
    session_id: str
    mode: str = "ADVANCED"
    model: str = "claude-3-5-sonnet-20241022"
    formatted_context: str = ""
    

@dataclass
class Question:
    """Simplified Question class for compatibility."""
    text: str
    special_context: str = ""


class ClaudeCodeSDK:
    """Main class for Claude Code SDK integration."""
    
    def __init__(self, codebase_path: str):
        """Initialize the Claude Code SDK.
        
        Args:
            codebase_path: Path to the codebase folder to analyze
        """
        self.codebase_path = Path(codebase_path)
        self.client = client
        self.file_contents_cache = {}
        
    def load_codebase_files(self, file_path_list: Optional[List[str]] = None) -> Dict[str, str]:
        """Load files from the codebase.
        
        Args:
            file_path_list: Optional list of specific files to load.
                          If None, loads all code files in the codebase.
        
        Returns:
            Dictionary mapping file paths to their contents
        """
        file_contents = {}
        
        if file_path_list:
            # Load specific files
            for file_path in file_path_list:
                full_path = self.codebase_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
                if full_path.exists() and full_path.is_file():
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            file_contents[str(full_path)] = f.read()
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
        else:
            # Load all code files in the codebase
            code_extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rs", ".md", ".txt"}
            
            for file_path in self.codebase_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in code_extensions:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            relative_path = file_path.relative_to(self.codebase_path)
                            file_contents[str(relative_path)] = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        self.file_contents_cache = file_contents
        return file_contents
    
    def format_codebase_context(self, file_contents: Dict[str, str]) -> str:
        """Format codebase files as context for Claude.
        
        Args:
            file_contents: Dictionary mapping file paths to contents
        
        Returns:
            Formatted string with file contents
        """
        if not file_contents:
            return "No codebase files loaded."
        
        context_parts = ["<codebase_context>\n"]
        
        for file_path, content in file_contents.items():
            context_parts.append(f"\n<file path=\"{file_path}\">")
            context_parts.append(content)
            context_parts.append("</file>\n")
        
        context_parts.append("\n</codebase_context>")
        return "".join(context_parts)
    
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for Claude.
        
        Args:
            chat_history: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Formatted chat history string
        """
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        
        return "\n\n".join(formatted)
    
    async def stream_response(self, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> AsyncGenerator[str, None]:
        """Stream response from Claude API.
        
        Args:
            prompt: The prompt to send to Claude
            model: The model to use
        
        Yields:
            Chunks of the response text
        """
        try:
            # Use synchronous streaming API
            with self.client.messages.stream(
                model=model,
                max_tokens=4096,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"Error: {str(e)}"


async def get_response(
    chat_session: ChatSession,
    file_path_list: Optional[List[str]],
    question: Question,
    chat_history: List[Dict[str, str]],
    embedding_folder_list: Optional[List[str]] = None,
    deep_thinking: bool = True,
    stream: bool = False
) -> AsyncGenerator[str, None]:
    """Main function to get response from Claude Code SDK.
    
    This function maintains compatibility with the existing get_response interface
    while using Claude's API for code analysis and question answering.
    
    Args:
        chat_session: Chat session information
        file_path_list: List of files to analyze (relative to codebase)
        question: The user's question
        chat_history: Previous conversation history
        embedding_folder_list: Not used in this implementation
        deep_thinking: Whether to use deep thinking mode
        stream: Whether to stream the response
    
    Yields:
        Response chunks if streaming, otherwise returns complete response
    """
    # Initialize the SDK with the test files folder
    codebase_path = "/Users/bingranyou/Documents/GitHub_Mac_mini/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/test_files"
    sdk = ClaudeCodeSDK(codebase_path)
    
    # Load codebase files
    file_contents = sdk.load_codebase_files(file_path_list)
    codebase_context = sdk.format_codebase_context(file_contents)
    
    # Format chat history
    formatted_history = sdk.format_chat_history(chat_history)
    
    # Combine question text with special context
    user_input = question.text
    if question.special_context:
        user_input += "\n\n" + question.special_context
    
    # Build the prompt
    prompt = f"""You are an expert code tutor helping a student understand code and technical documents.

You have access to the following codebase files:
{codebase_context}

Previous conversation:
{formatted_history}

RESPONSE GUIDELINES:
1. **TL;DR First**: Start with a 1-2 sentence summary answering the question directly
2. **Code Analysis**: When analyzing code, reference specific functions, classes, and line numbers
3. **Clear Explanations**: Use clear, precise technical language
4. **Formatting**: 
   - Use **bold** for key concepts
   - Use `inline code` for code references
   - Use code blocks with language syntax highlighting for longer code snippets
   - Use LaTeX with $...$ for inline math or $$...$$ for block math
5. **Structure**: Break down complex topics into logical segments
6. **Examples**: Include relevant examples when explaining concepts
7. **Accuracy**: Only use information from the provided context when available
8. **Professional Tone**: Maintain an academic, helpful tone

{"DEEP THINKING MODE: Provide thorough, detailed analysis with step-by-step explanations." if deep_thinking else ""}

Student's Question:
{user_input}

Please provide a comprehensive response following the guidelines above."""
    
    # Store context in session for reference
    chat_session.formatted_context = codebase_context
    
    # Generate response
    if stream:
        # Return a streaming generator
        async def response_generator():
            yield "<response>\n\n"
            async for chunk in sdk.stream_response(prompt, chat_session.model):
                yield chunk
            yield "\n\n</response>"
        
        return response_generator()
    else:
        # Return complete response (for non-streaming mode)
        response_parts = []
        async for chunk in sdk.stream_response(prompt, chat_session.model):
            response_parts.append(chunk)
        return "".join(response_parts)


# Convenience function for testing
async def create_test_session() -> ChatSession:
    """Create a test chat session."""
    return ChatSession(
        session_id="test_session_001",
        mode="ADVANCED",
        model="claude-3-5-sonnet-20241022"
    )


# Convenience function to run a simple query
async def simple_query(
    question_text: str,
    file_paths: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AsyncGenerator[str, None]:
    """Run a simple query against the codebase.
    
    Args:
        question_text: The question to ask
        file_paths: Optional list of specific files to analyze
        chat_history: Optional chat history
    
    Yields:
        Response chunks
    """
    session = await create_test_session()
    question = Question(text=question_text)
    history = chat_history or []
    
    async for chunk in await get_response(
        chat_session=session,
        file_path_list=file_paths,
        question=question,
        chat_history=history,
        embedding_folder_list=[],
        deep_thinking=True,
        stream=True
    ):
        yield chunk
