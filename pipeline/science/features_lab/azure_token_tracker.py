from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import json
import sys

# Load environment variables
load_dotenv()

def check_env_vars():
    """Check if required environment variables are set"""
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please make sure these variables are set in your .env file or environment.")
        return False
    
    print(f"✓ Found AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')[:20]}...")
    print(f"✓ Found AZURE_OPENAI_API_KEY: {os.getenv('AZURE_OPENAI_API_KEY')[:5]}...")
    return True

def track_azure_token_usage():
    """Track token usage for Azure OpenAI API calls"""
    if not check_env_vars():
        return None
    
    try:
        # Configure Azure OpenAI client based on api_handler.py configuration
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-08-01-preview",
            azure_deployment="gpt-4o",
            temperature=0
        )
        
        print("Making API call to Azure OpenAI...")
        
        # Track token usage with callback
        with get_openai_callback() as cb:
            response = llm.invoke("Tell me a joke about programming. Explain the joke in detail.")
            
            # Create JSON for token usage data
            usage_data = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost,
                "successful_requests": cb.successful_requests
            }
            
            # Print results
            print("\nResponse:")
            print(response.content)
            print("\nToken Usage:")
            print(json.dumps(usage_data, indent=2))
            
            return usage_data
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    result = track_azure_token_usage()
    if not result:
        sys.exit(1)