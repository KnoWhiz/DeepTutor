"""Module for managing chat history persistence in JSON files."""

import os
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime


def get_chat_history_path(session_id: str) -> str:
    """Get the full path for a chat history file.
    
    Args:
        session_id: Unique identifier for the chat session
        
    Returns:
        str: Full path to the chat history JSON file
    """
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    return os.path.join(base_dir, f"{session_id}.json")


def create_session_id() -> str:
    """Create a unique session ID using timestamp and UUID.
    
    Returns:
        str: Unique session identifier
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"chat_session_{timestamp}_{unique_id}"


def save_chat_history(session_id: str, chat_history: List[Dict]) -> None:
    """Save chat history to a JSON file.
    
    Args:
        session_id: Unique identifier for the chat session
        chat_history: List of chat messages with their metadata
    """
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
        print(f"Error saving chat history: {e}")


def load_chat_history(session_id: str) -> Optional[List[Dict]]:
    """Load chat history from a JSON file.
    
    Args:
        session_id: Unique identifier for the chat session
        
    Returns:
        Optional[List[Dict]]: List of chat messages if file exists, None otherwise
    """
    file_path = get_chat_history_path(session_id)
    
    if not os.path.exists(file_path):
        return None
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return None


def delete_chat_history(session_id: str) -> bool:
    """Delete a chat history file.
    
    Args:
        session_id: Unique identifier for the chat session
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    file_path = get_chat_history_path(session_id)
    
    if not os.path.exists(file_path):
        return False
        
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting chat history: {e}")
        return False


def cleanup_old_sessions() -> None:
    """Clean up chat history files older than 24 hours."""
    base_dir = os.path.join(os.path.dirname(__file__), "chat_history")
    if not os.path.exists(base_dir):
        return
        
    current_time = datetime.now()
    
    for filename in os.listdir(base_dir):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(base_dir, filename)
        try:
            # Check file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            # Delete if older than 24 hours
            if (current_time - file_time).total_seconds() > 86400:  # 24 hours in seconds
                os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up old session {filename}: {e}") 