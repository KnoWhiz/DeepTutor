import asyncio
import sys
import os

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.tutor_agent import tutor_agent_basic_streaming
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

async def test_tutor_agent_basic_streaming():
    """Test tutor_agent_basic_streaming to make sure our fix works"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    
    print("Starting tutor_agent_basic_streaming test...")
    count = 0
    
    # Process each chunk from the generator
    async for chunk in tutor_agent_basic_streaming(
        chat_session=chat_session,
        file_path_list=[],
        user_input='What is the meaning of life?', 
        time_tracking={},
        deep_thinking=False,
        stream=True
    ):
        count += 1
        print(f"CHUNK {count}: {chunk}")
    
    print(f"\nReceived {count} chunks total")

if __name__ == "__main__":
    asyncio.run(test_tutor_agent_basic_streaming()) 