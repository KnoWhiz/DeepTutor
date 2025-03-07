import os
import openai
from typing import Optional, Dict, Any, Union, Iterator
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.science.pipeline.config import load_config
load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.inference")


def deep_inference_agent(
    user_prompt: str,
    system_prompt: str = "You are a deep thinking researcher reading a paper. If you don't know the answer, say you don't know.",
    stream: bool = False,
):
    try:
        response = deepseek_inference(prompt=user_prompt, 
                                      system_message=system_prompt, 
                                      stream=stream,
                                      model="DeepSeek-R1")
        if response == None:
            raise Exception("No response from DeepSeek-R1")
        return response
    except Exception as e:
        logger.exception(f"An error occurred when calling DeepSeek-R1: {str(e)}")
        try:
            response = deepseek_inference(prompt=user_prompt, 
                                          system_message=system_prompt, 
                                          stream=stream,
                                          model="DeepSeek-R1-Distill-Llama-70B")
            if response == None:
                raise Exception("No response from DeepSeek-R1-Distill-Llama-70B")
            return response
        except Exception as e:
            logger.exception(f"An error occurred when calling DeepSeek-R1-Distill-Llama-70B: {str(e)}")
            try:
                response = o3mini_inference(user_prompt=user_prompt, 
                                            stream=stream)
                if response == None:
                    raise Exception("No response from o3mini")
                return response
            except Exception as e:
                logger.exception(f"An error occurred when calling o3mini: {str(e)}")
                response = "I'm sorry, I don't know the answer to that question."
                return response


def deepseek_inference(
    prompt: str,
    system_message: str = "You are a deep thinking researcher reading a paper. If you don't know the answer, say you don't know.",
    stream: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.1,
    max_tokens: int = 2000,
    model: str = "DeepSeek-R1-Distill-Llama-70B"
) -> Optional[str]:
    """
    Get completion from the DeepSeek model with optional streaming support.

    Args:
        prompt: The user's input prompt
        system_message: The system message to set the AI's behavior
        stream: Whether to stream the output or not
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate

    Returns:
        The generated text if streaming is False, None if streaming is True
    """
    config = load_config()
    max_tokens = config["inference_token_limit"]
    if model == "DeepSeek-R1-Distill-Llama-70B":
        model = "DeepSeek-R1-Distill-Llama-70B"
        base_url = "https://api.sambanova.ai/v1"
        max_tokens *= 3
    elif model == "DeepSeek-R1":
        model = "DeepSeek-R1"
        base_url = "https://preview.snova.ai/v1"
        max_tokens *= 1
    else:
        model = "DeepSeek-R1-Distill-Llama-70B"
        base_url = "https://api.sambanova.ai/v1"
        max_tokens *= 10

    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url=base_url
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            # Process the streaming response
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # Add a newline at the end
            return None
        else:
            # Return the complete response
            return response.choices[0].message.content

    except openai.APIError as e:
        logger.exception(f"API Error: {str(e)}")
        return None
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return None


def o3mini_inference(user_prompt: str, 
                     system_prompt: str = "You are a deep thinking researcher reading a paper. \
                        If you don't know the answer, say you don't know.",
                     stream: bool = False) -> Union[str, Iterator]:
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


# Example usage
if __name__ == "__main__":
    # Example with streaming
    logger.info("Streaming response:")
    deepseek_inference("what is 1+1?", stream=True)

    logger.info("\nNon-streaming response:")
    response = deepseek_inference("what is 1+1?", stream=False)
    if response:
        logger.info(response)