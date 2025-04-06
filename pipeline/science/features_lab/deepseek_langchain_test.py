import os
import sys
from dotenv import load_dotenv
from typing import Optional, Union, Iterator
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Handle imports for both direct execution and external import cases
try:
    # When imported as a module from elsewhere
    from pipeline.science.pipeline.session_manager import ChatSession
    from pipeline.science.pipeline.config import load_config
except ModuleNotFoundError:
    try:
        # When run directly
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from science.pipeline.session_manager import ChatSession
        from science.pipeline.config import load_config
    except ModuleNotFoundError:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        from science.pipeline.session_manager import ChatSession
        from science.pipeline.config import load_config

load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.deepseek_langchain_test")


def deepseek_langchain_inference(
    prompt: str,
    system_message: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or $$...$$.",
    stream: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.1,
    model: str = "DeepSeek-R1",
    chat_session: ChatSession = None
) -> Union[str, Iterator]:
    """
    Get completion from the DeepSeek model via LangChain with optional streaming support.

    Args:
        prompt: The user's input prompt
        system_message: The system message to set the AI's behavior
        stream: Whether to stream the output or not
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        top_p: Controls diversity via nucleus sampling
        model: The model to use (DeepSeek-R1 or DeepSeek-R1-Distill-Llama-70B)
        chat_session: Chat session object for managing conversation state

    Returns:
        The generated text if streaming is False, or a streaming iterator if streaming is True
    """
    if chat_session is None:
        chat_session = ChatSession()

    config = load_config()
    max_tokens = config["inference_token_limit"]
    
    # Adjust max_tokens based on the model
    if model == "DeepSeek-R1-Distill-Llama-70B":
        max_tokens *= 3
    elif model == "DeepSeek-R1":
        max_tokens *= 1
    else:
        max_tokens *= 3  # Default multiplier

    # Create the LangChain SambaNova model
    llm = ChatSambaNovaCloud(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        streaming=stream
    )

    # Create the messages
    messages = [
        (
            "system",
            system_message
        ),
        (
            "human",
            prompt
        )
    ]

    try:
        if stream:
            # For streaming responses
            def process_stream():
                yield "<think>"
                first_chunk = True
                
                for chunk in llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else ""
                    
                    if first_chunk:
                        yield "</think><response>"
                        first_chunk = False
                    
                    yield content
                
                yield "</response>"
            
            return process_stream()
        else:
            # For non-streaming responses
            response = llm.invoke(messages)
            return response.content
            
    except Exception as e:
        logger.exception(f"An error occurred while using LangChain with SambaNova: {str(e)}")
        return None


def create_langchain_chain(
    system_template: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or $$...$$.",
    model: str = "DeepSeek-R1",
    temperature: float = 0.6,
    top_p: float = 0.1
):
    """
    Create a LangChain chain with a prompt template and the DeepSeek model.
    
    Args:
        system_template: The system message template
        model: The model to use
        temperature: Controls randomness
        top_p: Controls diversity
        
    Returns:
        A LangChain chain
    """
    config = load_config()
    max_tokens = config["inference_token_limit"]
    
    # Adjust max_tokens based on the model
    if model == "DeepSeek-R1-Distill-Llama-70B":
        max_tokens *= 3
    elif model == "DeepSeek-R1":
        max_tokens *= 1
    else:
        max_tokens *= 3  # Default multiplier
    
    # Create the LangChain SambaNova model
    llm = ChatSambaNovaCloud(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate(
        [
            ("system", system_template),
            ("human", "{input}"),
        ]
    )
    
    # Create and return the chain
    return prompt | llm


# Example usage
if __name__ == "__main__":
    # Set up the SAMBANOVA_API_KEY environment variable if not already set
    if "SAMBANOVA_API_KEY" not in os.environ:
        sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
        if sambanova_api_key:
            os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key
        else:
            print("SAMBANOVA_API_KEY not found in environment variables")
            exit(1)
    
    # Example 1: Basic usage without streaming
    print("\nDeepSeek-R1 via LangChain (no streaming):")
    response = deepseek_langchain_inference(
        prompt="What is 1+1? Explain it in a detailed way.",
        model="DeepSeek-R1"
    )
    print(f"Response: {response}")
    
    # Example 2: With streaming
    print("\nDeepSeek-R1 via LangChain with streaming:")
    stream_response = deepseek_langchain_inference(
        prompt="What is quantum computing? Explain it in a detailed way.",
        model="DeepSeek-R1",
        stream=True
    )
    for chunk in stream_response:
        print(chunk, end="", flush=True)
    
    # Example 3: Using a chain
    print("\n\nDeepSeek-R1 via LangChain chain:")
    chain = create_langchain_chain(
        system_template="You are a math tutor. Explain concepts clearly and step by step.",
        model="DeepSeek-R1"
    )
    
    response = chain.invoke({"input": "Explain the concept of derivatives in calculus"})
    print(f"Chain response: {response.content}")
