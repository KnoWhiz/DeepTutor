#!/usr/bin/env python
"""
Streaming Test Script

This script demonstrates how to use the ApiHandler class to make a streaming
LLM API call with langchain and process the response in real-time using
LangChain Expression Language (LCEL).
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from pipeline
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.append(str(project_root))

from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from science.pipeline.api_handler import ApiHandler

def main():
    """
    Main function to demonstrate streaming LLM API calls using LCEL.
    """
    # Define parameters for ApiHandler
    parameters = {
        "openai_key_dir": ".env",  # Path to your .env file
        "temperature": 0.0,        # Temperature setting for the model
        "creative_temperature": 0.7,  # Temperature for creative outputs
        "llm_source": "azure",     # Source for LLM API (azure, openai, sambanova)
        "stream": True             # Enable streaming response
    }
    
    # Initialize ApiHandler with streaming enabled
    api_handler = ApiHandler(parameters, stream=True)
    
    # Get the model instance (using the 'basic' model for simplicity)
    model = api_handler.models["basic"]["instance"]
    
    # Define the question
    question = "what is 1+1 explain that in details"
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template("{question}")
    
    # Create a parser for the output
    parser = StrOutputParser()
    
    # Create the LCEL chain: prompt -> model -> parser
    chain = prompt | model | parser
    
    print(f"Question: {question}\n")
    print("Streaming response:")
    print("-" * 50)
    
    # Stream the response using LCEL
    response_chunks = []
    for chunk in chain.stream({"question": question}):
        response_chunks.append(chunk)
        # Print the chunk in real-time
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 50)
    print("\nFull response:")
    print("".join(response_chunks))

if __name__ == "__main__":
    main()
