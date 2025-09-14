"""
Claude Code SDK Chatbot Implementation

This module implements a chatbot using the Claude Code SDK for code analysis and generation.
It provides a streaming response generator similar to the get_response function format.
"""

import os
from typing import Generator, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from dotenv import load_dotenv
import json
import uuid
import asyncio
import threading
import queue
from enum import Enum

try:
    from bson import ObjectId
    from anthropic import Anthropic
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install: pip install anthropic pymongo")
    ObjectId = None
    Anthropic = None

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage, TextBlock, ResultMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chat_history_path(session_id: str) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    return os.path.join(base_dir, f"{session_id}.json")

def create_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"chat_session_{timestamp}_{unique_id}"

def save_chat_history(session_id: str, chat_history: List[Dict]) -> None:
    if not chat_history:
        return
    file_path = get_chat_history_path(session_id)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "last_updated": datetime.now().isoformat(),
                "messages": chat_history
            }, f, indent=2, ensure_ascii=False)
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
    def __init__(self, text="", language="English", question_type="global",
                 special_context="", answer_planning=None, image_url=None):
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
        return f"Question(text='{self.text}', language='{self.language}', type='{self.question_type}', image_url='{self.image_url}')"

    def to_dict(self):
        return {
            "text": self.text,
            "language": self.language,
            "question_type": self.question_type,
            "special_context": self.special_context,
            "answer_planning": self.answer_planning,
            "image_url": str(self.image_url)
        }

class ChatMode(str, Enum):
    LITE = "lite"
    BASIC = "basic"
    ADVANCED = "advanced"

def create_session_id_from_objectid() -> str:
    return str(ObjectId()) if ObjectId else create_session_id()

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
    accumulated_cost: float = 0.0

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
        logger.info(f"Updated accumulated cost for session {self.session_id}: ${self.accumulated_cost:.6f}")

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
            "accumulated_cost": self.accumulated_cost
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
            new_message_id=data.get("new_message_id", str(ObjectId()) if ObjectId else create_session_id()),
            question=data.get("question"),
            formatted_context=data.get("formatted_context"),
            page_formatted_context=data.get("page_formatted_context"),
            accumulated_cost=accumulated_cost
        )
        session.is_initialized = True
        return session

# -------------------------
# Claude Code streaming core
# -------------------------

async def get_claude_code_response_async(
    chat_session: ChatSession,
    file_path_list: List[str],
    question: Question,
    chat_history: str,
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    """
    Async implementation of Claude Code SDK chatbot that analyzes codebases and performs web searches.
    Streams text chunks as they arrive.
    """
    import subprocess

    # Install Claude Code SDK globally (best-effort)
    try:
        logger.info("Installing Claude Code SDK globally...\n")
        subprocess.run(
            ["npm", "install", "-g", "@anthropic-ai/claude-code"],
            check=True, capture_output=True, text=True
        )
        logger.info("Claude Code SDK installed successfully.\n")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Warning: Failed to install Claude Code SDK: {e}\n")

    codebase_folder_dir = file_path_list[0]

    # Load env
    # project_root = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor"
    # env_path = os.path.join(project_root, ".env")
    # load_dotenv(env_path, override=True)
    load_dotenv(".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        logger.error("Error: ANTHROPIC_API_KEY not found in .env file\n")
        return

    logger.info(f"✅ API key loaded: {api_key[:15]}... (length: {len(api_key)})\n")

    # Stream start marker
    yield "<response>\n"

    try:
        os.environ["ANTHROPIC_API_KEY"] = api_key

        options = ClaudeCodeOptions(
            cwd=codebase_folder_dir,
            allowed_tools=["Read", "Glob", "Grep", "WebSearch"],
            env={"ANTHROPIC_API_KEY": api_key},
            permission_mode="plan",  # read-only
            system_prompt=f"""You are a patient, helpful, and friendly tutor helping a student reading papers.

Instructions:
1) Explore and understand the file base structure
2) Analyze the relevant files
3) If needed, use web search
4) Provide a comprehensive response based on analysis and searches
5) Be professional, helpful and informative

You can only read files and search the web. Do not edit files.

Context from chat history: {chat_history}

Your task: {question.text}
"""
        )

        prompt = f"""Please analyze this file base and answer the question at the end.
Steps:
1) Explore the file base structure
2) Read & analyze relevant code/files
3) Use web search if additional context is needed
4) Provide a comprehensive analysis and final answer

File base: {codebase_folder_dir}
Question: {question.text}
"""

        # Stream messages from Claude Code
        async for message in query(prompt=prompt, options=options):
            # Assistant text blocks
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield block.text
                    elif hasattr(block, "text"):
                        yield block.text

            # Result messages (final structured result)
            elif isinstance(message, ResultMessage):
                if hasattr(message, "result") and message.result:
                    yield message.result

            else:
                # Generic status updates inferred from message shape/content
                # Keep these short & informative
                mstr = str(message)
                if "subtype='init'" in mstr:
                    yield "\n🔧 Initializing DeepTutor session...\n"
                elif "Glob" in mstr or "content=" in mstr and ("/" in mstr or "\\n" in mstr):
                    yield "\n📁 Exploring directory structure...\n"
                elif "Read" in mstr:
                    yield "\n📄 Reading file contents...\n"
                elif "Grep" in mstr:
                    yield "\n🔍 Searching within files...\n"
                elif "WebSearch" in mstr:
                    yield "\n🌐 Performing web search...\n"
                else:
                    # Fallback to raw text if it looks meaningful
                    # Avoid dumping giant tool payloads
                    snippet = mstr
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "…\n"
                    yield snippet

        # Stream end marker
        yield "\n</response>\n"

    except Exception as e:
        logger.error(f"Error during DeepTutor Claude Code analysis: {str(e)}\n")
        yield "</response>\n"

# --------------------------------------------
# Synchronous streaming wrapper (FIXED VERSION)
# --------------------------------------------

def get_claude_code_response(
    chat_session: ChatSession,
    file_path_list: List[str],
    question: Question,
    chat_history: str,
    deep_thinking: bool = True,
    stream: bool = True
) -> Generator[str, None, None]:
    """
    True streaming wrapper over the async generator.
    Runs the async producer in a background thread and yields chunks as they arrive.
    """
    q: queue.Queue = queue.Queue(maxsize=100)
    SENTINEL = object()

    def runner():
        async def consume():
            try:
                async for chunk in get_claude_code_response_async(
                    chat_session, file_path_list, question, chat_history, deep_thinking, stream
                ):
                    # Non-blocking put with backpressure
                    q.put(chunk)
            except Exception as e:
                q.put(f"\n[stream-error] {e}\n")
            finally:
                q.put(SENTINEL)

        # Start a fresh event loop in this thread
        asyncio.run(consume())

    t = threading.Thread(target=runner, name="claude-code-stream", daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item

# --------------------
# Example test harness
# --------------------

def test_claude_code_sdk():
    """
    Test function to demonstrate how to use the Claude Code SDK chatbot with streaming.
    """
    session = ChatSession()
    session.initialize()

    test_questions = [
        Question(
            text="Analyze the file base structure. Keep the response as concise as possible.",
            language="English",
            question_type="global",
            special_context="Focus on understanding the overall architecture"
        )
    ]

    codebase_dir = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/claude_code_integration_test/test_files"

    chat_history = """Previous conversation:
User: Hello, I'd like to analyze some code.
Assistant: I'd be happy to help you analyze your code! Please share the codebase or specific files you'd like me to review.
"""

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question.text}")
        print('='*70)

        try:
            response_generator = get_claude_code_response(
                chat_session=session,
                file_path_list=[codebase_dir],
                question=question,
                chat_history=chat_history,
                deep_thinking=True,
                stream=True
            )

            print("Streaming response:")
            print("-" * 50)
            response_content = ""
            for chunk in response_generator:
                # IMPORTANT: flush=True to see live streaming in the terminal
                print(chunk, end="", flush=True)
                response_content += chunk

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