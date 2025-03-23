import asyncio
import sys
import os

# Add the project root to the Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.science.pipeline.get_response import get_query_helper, Question
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode

async def test_get_query_helper():
    """Test the get_query_helper generator function"""
    chat_session = ChatSession('test', ChatMode.BASIC)
    
    print("Starting get_query_helper test...")
    count = 0
    question_obj = None
    
    # Process each chunk from the generator
    async for chunk in get_query_helper(chat_session, 'What is the meaning of life?', [], []):
        count += 1
        if isinstance(chunk, Question):
            question_obj = chunk
            print(f"CHUNK {count}: Question - {question_obj}")
        else:
            print(f"CHUNK {count}: {type(chunk).__name__} - {chunk}")
    
    print(f"\nReceived {count} chunks total")
    print(f"Final Question object: {question_obj}")

if __name__ == "__main__":
    asyncio.run(test_get_query_helper()) 