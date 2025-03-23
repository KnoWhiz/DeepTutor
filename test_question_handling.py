import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.tutor_agent import tutor_agent_basic_streaming

async def test_question_handling():
    """Test the Question handling in tutor_agent_basic_streaming"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    print("Starting question handling test...")
    
    # Create a temporary directory for embeddings
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embedding_dir = Path(temp_dir) / "test_embedding"
        temp_embedding_dir.mkdir(exist_ok=True)
        
        # Create a dummy documents_summary.txt file
        with open(temp_embedding_dir / "documents_summary.txt", "w") as f:
            f.write("This is a test document summary.")
        
        count = 0
        async for chunk in tutor_agent_basic_streaming(
            chat_session=chat_session,
            file_path_list=[],
            user_input="What is the meaning of life?",
            time_tracking={},
            deep_thinking=False,
            stream=True
        ):
            count += 1
            print(f"CHUNK {count}: {chunk}")
        
        print(f"\nReceived {count} chunks total")

if __name__ == "__main__":
    asyncio.run(test_question_handling()) 