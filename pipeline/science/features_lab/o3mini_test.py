import os
import dotenv
from typing import Optional, Dict, Any, Union, Iterator

# Load environment variables
dotenv.load_dotenv()

# Import LangChain components
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def generate_response(user_prompt: str, system_prompt: str = "", stream: bool = False) -> Union[str, Iterator]:
    """
    Generate a response using Azure OpenAI through LangChain
    
    Args:
        system_prompt: The system instruction for the AI
        user_prompt: The user's query or input
        stream: Whether to stream the response (default: False)
    
    Returns:
        If stream=False: The text content of the model's response as a string
        If stream=True: A streaming response iterator that can be iterated over
    """
    # Azure OpenAI credentials
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP")
    deployment = "o3-mini"
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY_BACKUP")
    
    # Initialize the Azure OpenAI model through LangChain
    # Use model_kwargs to properly pass the max_completion_tokens parameter
    model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-12-01-preview",
        deployment_name=deployment,
        model_kwargs={"max_completion_tokens": 100000},
        streaming=stream
    )
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Generate response
    response = model.invoke(messages)
    
    # Return appropriate response based on streaming mode
    if stream:
        return response  # Return the streaming response iterator
    else:
        # Return just the content string
        return response.content

if __name__ == "__main__":
    # Example usage
    system_prompt = "You are a deep thinking AI assistant that helps people find information."
    user_prompt = "What are the key principles of machine learning?"
    
    # Non-streaming example
    response = generate_response(system_prompt, user_prompt)
    print(response)
    
    # Streaming example
    print("\nStreaming response example:")
    streaming_response = generate_response(system_prompt, user_prompt, stream=True)
    
    # Create a dictionary to store collected response parts
    response_parts = {}
    
    # Process each chunk in the stream
    for chunk in streaming_response:
        # Each chunk is a tuple where the first element is the key
        # and the second element would be the value (if present)
        if isinstance(chunk, tuple) and len(chunk) > 0:
            key = chunk[0]
            
            # If this is a content chunk and has a value
            if key == "content" and len(chunk) > 1:
                # Print the actual content chunk
                print(chunk[1], end="", flush=True)
    
    print()  