"""
Claude Code SDK Chatbot Implementation (Final, SDK-consistent Streaming)

This module implements a Claude Code SDK–based chatbot for code analysis and generation,
using the Python SDK's streaming APIs from:
https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python

Key points:
- Uses ClaudeSDKClient with receive_response() for robust streaming
- Handles AssistantMessage content blocks (Text, Thinking, ToolUse, ToolResult)
- Gracefully supports older claude_code_sdk versions (e.g., no max_thinking_tokens)
- Threaded sync wrapper with backpressure (Queue) and clean sentinel shutdown

Requirements:
  pip install claude-code-sdk python-dotenv
  npm install -g @anthropic-ai/claude-code
  export ANTHROPIC_API_KEY=...

Run:
  python claude_code_sdk_test.py
"""

from __future__ import annotations

import os
import json
import uuid
import asyncio
import threading
import queue
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Dict, List, Optional, Set, AsyncIterator

from dotenv import load_dotenv

# Optional dependency (used only for nice session IDs)
try:
    from bson import ObjectId
except ImportError:
    ObjectId = None

# ---- Claude Code SDK imports (match current public docs) ----
try:
    from claude_code_sdk import (
        query,
        ClaudeCodeOptions,
        ClaudeSDKClient,
        AssistantMessage,
        ResultMessage,
        TextBlock,
        # The following may not exist in older SDK versions; we handle that gracefully.
    )
    try:
        from claude_code_sdk import ThinkingBlock, ToolUseBlock, ToolResultBlock
    except Exception:
        ThinkingBlock = None  # type: ignore
        ToolUseBlock = None   # type: ignore
        ToolResultBlock = None  # type: ignore
    try:
        from claude_code_sdk import (
            CLINotFoundError,
            ProcessError,
            CLIConnectionError,
            CLIJSONDecodeError,
        )
    except Exception:
        # Older SDKs may not expose all errors; provide fallbacks
        class _FallbackSDKError(Exception): ...
        class CLINotFoundError(_FallbackSDKError): ...
        class ProcessError(_FallbackSDKError):
            def __init__(self, message: str, exit_code: int | None = None, stderr: str | None = None):
                super().__init__(message)
                self.exit_code = exit_code
                self.stderr = stderr
        class CLIConnectionError(_FallbackSDKError): ...
        class CLIJSONDecodeError(_FallbackSDKError): ...
except ImportError as e:
    raise ImportError(
        "claude_code_sdk is not installed or not importable. "
        "Please install it with: pip install claude-code-sdk"
    ) from e

# ------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Utilities: chat history I/O
# -------------------------

def get_chat_history_path(session_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    return os.path.join(base_dir, f"{session_id}.json")

def create_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"chat_session_{timestamp}_{unique_id}"

def create_session_id_from_objectid() -> str:
    return str(ObjectId()) if ObjectId else create_session_id()

def save_chat_history(session_id: str, chat_history: List[Dict]) -> None:
    if not chat_history:
        return
    file_path = get_chat_history_path(session_id)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": session_id,
                    "last_updated": datetime.now().isoformat(),
                    "messages": chat_history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception as e:
        logger.info(f"Error saving chat history: {e}")

def load_chat_history(session_id: str) -> Optional[List[Dict]]:
    file_path = get_chat_history_path(session_id)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except Exception as e:
        logger.info(f"Error loading chat history: {e}")
        return None

def delete_chat_history(session_id: str) -> bool:
    file_path = get_chat_history_path(session_id)
    if not os.path.exists(file_path):
        return False
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        logger.info(f"Error deleting chat history: {e}")
        return False

def cleanup_old_sessions() -> None:
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    if not os.path.exists(base_dir):
        return
    current_time = datetime.now()
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(base_dir, filename)
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (current_time - file_time).total_seconds() > 86400:
                os.remove(file_path)
        except Exception as e:
            logger.info(f"Error cleaning up old session {filename}: {e}")

# -------------------------
# Question & Session models
# -------------------------

class Question:
    def __init__(
        self,
        text: str = "",
        language: str = "English",
        question_type: str = "global",
        special_context: str = "",
        answer_planning: Optional[Dict] = None,
        image_url: Optional[str] = None,
    ):
        self.text = text
        self.language = language
        if question_type not in ["local", "global", "image"]:
            self.question_type = "global"
        else:
            self.question_type = question_type
        self.special_context = special_context
        self.answer_planning = answer_planning or {}
        self.image_url = image_url

    def __str__(self):
        return (
            f"Question(text='{self.text}', language='{self.language}', "
            f"type='{self.question_type}', image_url='{self.image_url}')"
        )

    def to_dict(self):
        return {
            "text": self.text,
            "language": self.language,
            "question_type": self.question_type,
            "special_context": self.special_context,
            "answer_planning": self.answer_planning,
            "image_url": str(self.image_url),
        }

class ChatMode(str, Enum):
    LITE = "lite"
    BASIC = "basic"
    ADVANCED = "advanced"

@dataclass
class ChatSession:
    session_id: str = field(default_factory=create_session_id_from_objectid)
    mode: ChatMode = ChatMode.BASIC
    chat_history: List[Dict] = field(default_factory=list)
    uploaded_files: Set[str] = field(default_factory=set)
    current_language: Optional[str] = None
    is_initialized: bool = False
    current_message: Optional[str] = ""
    new_message_id: str = str(ObjectId()) if ObjectId else create_session_id()
    question: Optional[Question] = None
    formatted_context: Optional[Dict] = None
    page_formatted_context: Optional[Dict] = None
    accumulated_cost: float = 0.0  # Placeholder for parity with other backends

    def initialize(self) -> None:
        if self.is_initialized:
            return
        loaded_history = load_chat_history(self.session_id)
        if loaded_history:
            self.chat_history = loaded_history
        self.is_initialized = True

    def add_message(self, message: Dict) -> None:
        self.chat_history.append(message)
        save_chat_history(self.session_id, self.chat_history)

    def clear_history(self) -> None:
        self.chat_history = []
        delete_chat_history(self.session_id)
        self.accumulated_cost = 0.0

    def set_mode(self, mode: ChatMode) -> None:
        self.mode = mode

    def add_file(self, file_path: str) -> None:
        self.uploaded_files.add(file_path)

    def remove_file(self, file_path: str) -> None:
        self.uploaded_files.discard(file_path)

    def set_language(self, language: str) -> None:
        self.current_language = language

    def update_cost(self, cost: float) -> None:
        self.accumulated_cost += cost
        logger.info(
            f"Updated accumulated cost for session {self.session_id}: ${self.accumulated_cost:.6f}"
        )

    def get_accumulated_cost(self) -> float:
        return self.accumulated_cost

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "chat_history": self.chat_history,
            "uploaded_files": list(self.uploaded_files),
            "current_language": self.current_language,
            "last_updated": datetime.now().isoformat(),
            "current_message": self.current_message,
            "new_message_id": self.new_message_id,
            "question": self.question,
            "formatted_context": self.formatted_context,
            "page_formatted_context": self.page_formatted_context,
            "accumulated_cost": self.accumulated_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        if "current_message" not in data:
            data["current_message"] = ""
        accumulated_cost = data.get("accumulated_cost", 0.0)
        session = cls(
            session_id=data["session_id"],
            mode=ChatMode(data.get("mode", ChatMode.BASIC.value)),
            chat_history=data.get("chat_history", []),
            uploaded_files=set(data.get("uploaded_files", [])),
            current_language=data.get("current_language"),
            current_message=data["current_message"],
            new_message_id=data.get(
                "new_message_id", str(ObjectId()) if ObjectId else create_session_id()
            ),
            question=data.get("question"),
            formatted_context=data.get("formatted_context"),
            page_formatted_context=data.get("page_formatted_context"),
            accumulated_cost=accumulated_cost,
        )
        session.is_initialized = True
        return session

# -------------------------
# Claude Code streaming core
# -------------------------

def _make_options_with_fallback(**kwargs) -> ClaudeCodeOptions:
    """
    Instantiate ClaudeCodeOptions, gracefully handling SDK versions that
    don't support some fields (e.g., max_thinking_tokens).
    """
    try:
        return ClaudeCodeOptions(**kwargs)
    except TypeError:
        # Remove keys that older SDKs may not support
        unsupported = ("max_thinking_tokens", "can_use_tool", "hooks")
        filtered = {k: v for k, v in kwargs.items() if k not in unsupported}
        return ClaudeCodeOptions(**filtered)

def _block_name(b: object) -> str:
    return getattr(b, "__class__", type(b)).__name__

async def _iter_response_messages(client: ClaudeSDKClient) -> AsyncIterator[str]:
    """
    Iterate over messages from Claude and yield user-facing text chunks.
    Handles Text, Thinking, ToolUse, and ToolResult blocks when available.
    """
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                # Text
                if isinstance(block, TextBlock):
                    yield block.text

                # Thinking (if SDK provides this class)
                elif ThinkingBlock and isinstance(block, ThinkingBlock):  # type: ignore
                    thinking = getattr(block, "thinking", "")
                    if thinking:
                        yield f"\n<thinking>{thinking}</thinking>\n"

                # ToolUse (if SDK provides)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):  # type: ignore
                    name = getattr(block, "name", "tool")
                    yield f"\n<tool name='{name}' />\n"

                # ToolResult (if SDK provides)
                elif ToolResultBlock and isinstance(block, ToolResultBlock):  # type: ignore
                    content = getattr(block, "content", None)
                    if isinstance(content, str):
                        yield f"\n<tool_result>{content}</tool_result>\n"
                    elif isinstance(content, list):
                        snippet = json.dumps(content)
                        if len(snippet) > 2000:
                            snippet = snippet[:2000] + "…"
                        yield f"\n<tool_result>{snippet}</tool_result>\n"
                    else:
                        yield "\n<tool_result />\n"

                # Fallback for unknown blocks
                else:
                    yield f"\n<!-- unsupported block: {_block_name(block)} -->\n"

        elif isinstance(message, ResultMessage):
            # Marks completion of this response
            yield "\n</done>\n"

async def get_claude_code_response_async(
    chat_session: ChatSession,
    file_path_list: List[str],
    question: Question,
    chat_history: str,
    deep_thinking: bool = True,
    stream: bool = True,
) -> AsyncIterator[str]:
    """
    Async implementation using ClaudeSDKClient for robust streaming.
    """
    # Load env & key
    load_dotenv(".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield "<error>ANTHROPIC_API_KEY is not set. Please export it or add to .env</error>\n"
        return
    os.environ["ANTHROPIC_API_KEY"] = api_key  # ensure CLI/SDK can see it

    # Codebase path
    if not file_path_list:
        yield "<error>No codebase directory provided.</error>\n"
        return
    codebase_folder_dir = file_path_list[0]

    # Build options (per docs)
    options = _make_options_with_fallback(
        cwd=codebase_folder_dir,
        allowed_tools=["Read", "Glob", "Grep", "WebSearch"],
        permission_mode="plan",  # read-only (no edits)
        env={"ANTHROPIC_API_KEY": api_key},
        system_prompt=(
            "You are a patient, helpful, and professional tutor assisting with reading code "
            "and research papers. You can read files and search the web when needed. "
            "Be precise, cite filenames/paths you inspected, and keep answers concise unless asked.\n\n"
            f"Context from chat history:\n{chat_history}"
        ),
        # Use token budget if supported by current SDK; otherwise silently ignored by fallback
        max_thinking_tokens=8000 if deep_thinking else 0,
    )

    prompt = f"""Analyze the repository and answer the question.

Steps:
1) Explore the file base structure (list key dirs/files you looked at)
2) Read & analyze relevant code/files
3) Use web search if additional context is needed
4) Provide a concise, actionable answer

Codebase: {codebase_folder_dir}
Question: {question.text}
"""

    # Begin streaming
    yield "<response>\n"
    try:
        async with ClaudeSDKClient(options=options) as client:
            # Send query into a named session for resumability
            await client.query(prompt, session_id=chat_session.session_id)

            # Stream messages until ResultMessage is received
            async for text in _iter_response_messages(client):
                yield text

        yield "</response>\n"

    except CLINotFoundError:
        yield (
            "<error>Claude Code CLI not found. "
            "Install with: npm install -g @anthropic-ai/claude-code</error>\n</response>\n"
        )
    except CLIJSONDecodeError as e:
        yield f"<error>Failed to parse CLI JSON output: {e}</error>\n</response>\n"
    except ProcessError as e:
        detail = f" (exit_code={e.exit_code})" if getattr(e, 'exit_code', None) else ""
        yield f"<error>Claude Code process failed{detail}. Check logs.</error>\n</response>\n"
    except CLIConnectionError as e:
        yield f"<error>Failed to connect to Claude Code: {e}</error>\n</response>\n"
    except asyncio.CancelledError:
        yield "<error>Streaming cancelled.</error>\n</response>\n"
        raise
    except Exception as e:
        yield f"<error>{type(e).__name__}: {str(e)}</error>\n</response>\n"

# --------------------------------------------
# Synchronous streaming wrapper (threaded)
# --------------------------------------------

def get_claude_code_response(
    chat_session: ChatSession,
    file_path_list: List[str],
    question: Question,
    chat_history: str,
    deep_thinking: bool = True,
    stream: bool = True,
) -> Generator[str, None, None]:
    """
    True streaming wrapper over the async generator using a background thread.

    - Dedicated event loop via asyncio.run()
    - Bounded queue with backpressure (maxsize=256)
    - Sentinel-based shutdown (no busy-waiting)
    """
    q: "queue.Queue[object]" = queue.Queue(maxsize=256)
    SENTINEL = object()

    def runner():
        async def consume():
            try:
                agen = get_claude_code_response_async(
                    chat_session, file_path_list, question, chat_history, deep_thinking, stream
                )
                async for chunk in agen:
                    q.put(chunk)  # blocks if full → backpressure
            except Exception as e:
                q.put(f"\n[stream-error] {type(e).__name__}: {e}\n")
            finally:
                q.put(SENTINEL)

        asyncio.run(consume())

    t = threading.Thread(target=runner, name="claude-code-stream", daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item if isinstance(item, str) else str(item)

# --------------------
# Example test harness
# --------------------

def test_claude_code_sdk():
    """
    Demonstrates how to use the Claude Code SDK chatbot with streaming.
    """
    session = ChatSession()
    session.initialize()

    test_questions = [
        Question(
            text="Analyze the file base structure. Keep the response as concise as possible.",
            language="English",
            question_type="global",
            special_context="Focus on understanding the overall architecture",
        )
    ]

    # Replace with your actual path
    codebase_dir = (
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/"
        "pipeline/science/features_lab/claude_code_integration_test/test_files"
    )

    chat_history = """Previous conversation:
User: Hello, I'd like to analyze some code.
Assistant: I'd be happy to help you analyze your code! Please share the codebase or specific files you'd like me to review.
"""

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question.text}")
        print("=" * 70)

        try:
            response_generator = get_claude_code_response(
                chat_session=session,
                file_path_list=[codebase_dir],
                question=question,
                chat_history=chat_history,
                deep_thinking=True,
                stream=True,
            )

            print("Streaming response:")
            print("-" * 50)
            response_content = ""
            for chunk in response_generator:
                print(chunk, end="", flush=True)  # show live chunks
                response_content += chunk

            # Append to history
            session.add_message({"role": "user", "content": question.text})
            session.add_message({"role": "assistant", "content": response_content})
            chat_history += f"\nUser: {question.text}\nAssistant: {response_content}"

            print("\n" + "-" * 50)
            print(f"Test {i} completed successfully!")

        except Exception as e:
            print(f"Error during test {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_claude_code_sdk()