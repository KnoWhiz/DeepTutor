import os
import openai
from typing import Optional, Dict, Any, Union, Iterator
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import sys
import os.path
from pipeline.science.pipeline.session_manager import ChatSession
from pipeline.science.pipeline.cost_tracker import track_cost_and_stream
from langchain_sambanova import ChatSambaNovaCloud

# Handle imports for both direct execution and external import cases
try:
    # When imported as a module from elsewhere
    from pipeline.science.pipeline.config import load_config
except ModuleNotFoundError:
    try:
        # When run directly
        from config import load_config
    except ModuleNotFoundError:
        # If in the pipeline directory and running the script
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from science.pipeline.config import load_config

load_dotenv()

import logging
logger = logging.getLogger("tutorpipeline.science.inference")


def deep_inference_agent(
    user_prompt: str,
    system_prompt: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.",
    stream: bool = False,
    chat_session: ChatSession = None
):
    if chat_session is None:
        # Initialize the chat session
        chat_session = ChatSession()
    if stream is False:
        try:
            # response = deepseek_langchain_inference(prompt=user_prompt,
            #                             system_message=system_prompt,
            #                             stream=stream,
            #                             model="DeepSeek-R1",
            #                             chat_session=chat_session)
            response = o3mini_inference(user_prompt=user_prompt,
                                        stream=stream,
                                        chat_session=chat_session)
            if response == None:
                # raise Exception("No response from DeepSeek-R1")
                raise Exception("No response from o3mini")
            return response
        except Exception as e:
            # logger.exception(f"An error occurred when calling DeepSeek-R1: {str(e)}")
            logger.exception(f"An error occurred when calling o3mini: {str(e)}")
            try:
                # response = deepseek_langchain_inference(prompt=user_prompt,
                #                             system_message=system_prompt,
                #                             stream=stream,
                #                             model="DeepSeek-R1-Distill-Llama-70B",
                #                             chat_session=chat_session)
                response = o3mini_inference(user_prompt=user_prompt,
                                            stream=stream,
                                            chat_session=chat_session)
                if response == None:
                    # raise Exception("No response from DeepSeek-R1-Distill-Llama-70B")
                    raise Exception("No response from o3mini")
                return response
            except Exception as e:
                # logger.exception(f"An error occurred when calling DeepSeek-R1-Distill-Llama-70B: {str(e)}")
                logger.exception(f"An error occurred when calling o3mini: {str(e)}")
                try:
                    response = o3mini_inference(user_prompt=user_prompt,
                                                stream=stream,
                                                chat_session=chat_session)
                    if response == None:
                        raise Exception("No response from o3mini")
                    return response
                except Exception as e:
                    logger.exception(f"An error occurred when calling o3mini: {str(e)}")
                    response = "I'm sorry, I don't know the answer to that question."
                    return response

    else:
        # The stream is True, the answer is a generator
        # We need to return a generator that handles errors internally
        def safe_stream_generator():
            # Try DeepSeek-R1
            try:
                # stream_response = deepseek_langchain_inference(prompt=user_prompt,
                #                                     system_message=system_prompt,
                #                                     stream=stream,
                #                                     model="DeepSeek-R1",
                #                                     chat_session=chat_session)
                stream_response = o3mini_inference(user_prompt=user_prompt,
                                                    stream=stream,
                                                    chat_session=chat_session)
                if stream_response is None:
                    # raise Exception("No response from DeepSeek-R1")
                    raise Exception("No response from o3mini")

                # Try to consume the generator - any errors will be caught here
                for chunk in stream_response:
                    yield chunk
                return  # Exit if successful
            except Exception as e:
                # logger.exception(f"An error occurred when calling DeepSeek-R1: {str(e)}")
                logger.exception(f"An error occurred when calling o3mini: {str(e)}")

            # Try DeepSeek-R1-Distill-Llama-70B as fallback
            try:
                # stream_response = deepseek_langchain_inference(prompt=user_prompt,
                #                                     system_message=system_prompt,
                #                                     stream=stream,
                #                                     model="DeepSeek-R1-Distill-Llama-70B",
                #                                     chat_session=chat_session)
                stream_response = o3mini_inference(user_prompt=user_prompt,
                                                    stream=stream,
                                                    chat_session=chat_session)
                if stream_response is None:
                    # raise Exception("No response from DeepSeek-R1-Distill-Llama-70B")
                    raise Exception("No response from o3mini")

                # Try to consume the generator - any errors will be caught here
                for chunk in stream_response:
                    yield chunk
                return  # Exit if successful
            except Exception as e:
                # logger.exception(f"An error occurred when calling DeepSeek-R1-Distill-Llama-70B: {str(e)}")
                logger.exception(f"An error occurred when calling o3mini: {str(e)}")

            # Try o3mini as final fallback
            try:
                stream_response = o3mini_inference(user_prompt=user_prompt,
                                                stream=stream,
                                                chat_session=chat_session)
                if stream_response is None:
                    raise Exception("No response from o3mini")

                # Try to consume the generator - any errors will be caught here
                for chunk in stream_response:
                    yield chunk
                return  # Exit if successful
            except Exception as e:
                logger.exception(f"An error occurred when calling o3mini: {str(e)}")
                yield "I'm sorry, I don't know the answer to that question."

        return safe_stream_generator()


# def deepseek_inference(
#     prompt: str,
#     system_message: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.",
#     stream: bool = False,
#     temperature: float = 0.6,
#     top_p: float = 0.1,
#     max_tokens: int = 2000,
#     model: str = "DeepSeek-R1-Distill-Llama-70B",
#     chat_session: ChatSession = None
# ) -> Optional[str]:
#     """
#     Get completion from the DeepSeek model with optional streaming support.

#     Args:
#         prompt: The user's input prompt
#         system_message: The system message to set the AI's behavior
#         stream: Whether to stream the output or not
#         temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
#         top_p: Controls diversity via nucleus sampling
#         max_tokens: Maximum number of tokens to generate

#     Returns:
#         The generated text if streaming is False, None if streaming is True
#     """
#     if chat_session is None:
#         # Initialize the chat session
#         chat_session = ChatSession()
#     config = load_config()
#     max_tokens = config["inference_token_limit"]
#     if model == "DeepSeek-R1-Distill-Llama-70B":
#         model = "DeepSeek-R1-Distill-Llama-70B"
#         base_url = "https://api.sambanova.ai/v1"
#         max_tokens *= 3
#     elif model == "DeepSeek-R1":
#         model = "DeepSeek-R1"
#         base_url = "https://preview.snova.ai/v1"
#         max_tokens *= 1
#     else:
#         model = "DeepSeek-R1-Distill-Llama-70B"
#         base_url = "https://api.sambanova.ai/v1"
#         max_tokens *= 10

#     client = openai.OpenAI(
#         api_key=os.environ.get("SAMBANOVA_API_KEY"),
#         base_url=base_url
#     )

#     if stream is False:
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_message},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 stream=stream
#             )

#             # Return the complete response
#             return response.choices[0].message.content

#         except openai.APIError as e:
#             logger.exception(f"API Error: {str(e)}")
#             return None
#         except Exception as e:
#             logger.exception(f"An error occurred: {str(e)}")
#             return None
#     else:
#         # If the response is streaming, process the streaming response. The response is a generator.
#         logger.info("Streaming response from DeepSeek:")
#         try:
#             def deepseek_stream_response(chat_session, model, system_message, prompt, temperature, top_p, max_tokens, stream):
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=[
#                         {"role": "system", "content": system_message},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=temperature,
#                     top_p=top_p,
#                     max_tokens=max_tokens,
#                     stream=stream
#                 )

#                 # Process the streaming response
#                 found_think_end = False
#                 accumulated_text = ""

#                 for chunk in response:
#                     if chunk.choices[0].delta.content is not None:
#                         chunk_content = chunk.choices[0].delta.content
#                         accumulated_text += chunk_content

#                         # Check if we just found the end of the thinking section
#                         if "</think>" in accumulated_text and not found_think_end:
#                             # Split the accumulated text at "</think>"
#                             parts = accumulated_text.split("</think>", 1)
#                             if len(parts) > 1:
#                                 # Yield everything up to and including "</think>"
#                                 yield parts[0] + "</think>"
#                                 # chat_session.current_message += parts[0] + "</think>"
#                                 # Yield the "<response>" tag
#                                 yield "<response>"
#                                 # chat_session.current_message += "<response>"
#                                 # Yield the remainder of the text after "</think>"
#                                 if parts[1]:
#                                     yield parts[1]
#                                 # Mark that we've found the end of thinking
#                                 found_think_end = True
#                                 # Reset accumulated text since we've processed it
#                                 accumulated_text = ""
#                         else:
#                             # If we've already found the thinking end or haven't found it yet in this chunk
#                             if found_think_end or "</think>" not in chunk_content:
#                                 yield chunk_content

#                 # Add the closing response tag at the end
#                 yield "</response>"
#                 # chat_session.current_message += "</response>"
#             return deepseek_stream_response(chat_session, model, system_message, prompt, temperature, top_p, max_tokens, stream)

#         except openai.APIError as e:
#             logger.exception(f"API Error: {str(e)}")
#             return None
#         except Exception as e:
#             logger.exception(f"An error occurred: {str(e)}")
#             return None


def deepseek_langchain_inference(
    prompt: str,
    system_message: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.",
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
        max_tokens *= 2  # Default multiplier

    # Create the LangChain SambaNova model
    llm = ChatSambaNovaCloud(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        streaming=stream,
        model_kwargs={"stream_options": {"include_usage": True}} if stream else {}
    )

    # Create the prompt template for the chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{user_input}")
    ])
    
    # Create a parser
    parser = StrOutputParser()
    
    # Define the chain using LCEL
    chain = prompt_template | llm | parser

    try:
        if stream:
            # For streaming responses
            def process_stream():
                # yield "<think>"
                first_chunk = True

                stream_generator, cost_info = track_cost_and_stream(user_input=prompt, chain=chain)
                
                for chunk in stream_generator:
                    if "</think>" in chunk:
                        if chunk.startswith("</think>"):
                            chunk = chunk.replace("</think>", "")
                            yield "</think>"
                            yield "<response>"
                            yield chunk
                            first_chunk = False
                        elif chunk.endswith("</think>"):
                            chunk = chunk.replace("</think>", "")
                            yield chunk
                            yield "</think>"
                            yield "<response>"
                            first_chunk = False
                        else:
                            yield chunk.split("</think>")[0]   # Before the thinking section
                            yield "</think>"
                            yield "<response>"
                            yield chunk.split("</think>")[1]   # After the thinking section
                            first_chunk = False
                    else:
                        yield chunk
                
                yield "</response>"

                logger.info(f"Cost info dict: {cost_info}")
                # Update the accumulated cost in chat_session
                if chat_session is not None and "total_cost" in cost_info:
                    chat_session.update_cost(cost_info["total_cost"])
                    logger.info(f"Updated session cost: ${chat_session.get_accumulated_cost():.6f}")
            
            return process_stream()
        else:
            # For non-streaming responses
            response = chain.invoke({"user_input": prompt})
            return response
            
    except Exception as e:
        logger.exception(f"An error occurred while using LangChain with SambaNova: {str(e)}")
        return None


def o3mini_inference(user_prompt: str,
                     system_prompt: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.",
                     stream: bool = False,
                     chat_session: ChatSession = None) -> Union[str, Iterator]:
    """
    Generate a response using Azure OpenAI through LangChain

    Args:
        system_prompt: The system instruction for the AI
        user_prompt: The user's query or input
        stream: Whether to stream the response (default: False)

    Returns:
        If stream = False: The text content of the model's response as a string
        If stream = True: A streaming response iterator that can be iterated over
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
        model_kwargs={"max_completion_tokens": 100000, "stream_options": {"include_usage": True}} if stream else {"max_completion_tokens": 100000},
        streaming=stream
    )

    # Create prompt template for the chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    # Create a parser
    parser = StrOutputParser()
    
    # Define the chain using LCEL
    chain = prompt_template | model | parser

    # Generate response - use different methods depending on streaming mode
    if stream:
        # Return the streaming response generator
        def o3mini_stream_response(chain, user_prompt, stream):
            yield "<think>"
            first_token = True
            stream_generator, cost_info = track_cost_and_stream(user_input=user_prompt, chain=chain)
            for chunk in stream_generator:
                if first_token:
                    yield "</think>"
                    yield "<response>"
                    first_token = False
                yield chunk
            yield "</response>"
            logger.info(f"Cost info dict: {cost_info}")
            # Update the accumulated cost in chat_session
            if chat_session is not None and "total_cost" in cost_info:
                chat_session.update_cost(cost_info["total_cost"])
                logger.info(f"Updated session cost: ${chat_session.get_accumulated_cost():.6f}")
        return o3mini_stream_response(chain, user_prompt, stream)
    else:
        # Return just the content string from the complete response
        response = chain.invoke({"user_input": user_prompt})
        return response


def o4mini_inference(user_prompt: str,
                     system_prompt: str = "You are a professional deep thinking researcher reading a paper. Analyze the paper context content and answer the question. If the information is not provided in the paper, say you cannot find the answer in the paper but will try to answer based on your knowledge. For formulas, use LaTeX format with $...$ or \n$$...\n$$.",
                     stream: bool = False,
                     chat_session: ChatSession = None) -> Union[str, Iterator]:
    """
    Generate a response using Azure OpenAI through LangChain

    Args:
        system_prompt: The system instruction for the AI
        user_prompt: The user's query or input
        stream: Whether to stream the response (default: False)

    Returns:
        If stream = False: The text content of the model's response as a string
        If stream = True: A streaming response iterator that can be iterated over
    """
    # Azure OpenAI credentials
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP")
    deployment = "o4-mini"
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY_BACKUP")

    # Initialize the Azure OpenAI model through LangChain
    # Use model_kwargs to properly pass the max_completion_tokens parameter
    model = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-12-01-preview",
        deployment_name=deployment,
        model_kwargs={"max_completion_tokens": 100000, "stream_options": {"include_usage": True}} if stream else {"max_completion_tokens": 100000},
        streaming=stream
    )

    # Create prompt template for the chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    # Create a parser
    parser = StrOutputParser()
    
    # Define the chain using LCEL
    chain = prompt_template | model | parser

    # Generate response - use different methods depending on streaming mode
    if stream:
        # Return the streaming response generator
        def o4mini_stream_response(chain, user_prompt, stream):
            yield "<think>"
            first_token = True
            stream_generator, cost_info = track_cost_and_stream(user_input=user_prompt, chain=chain)
            for chunk in stream_generator:
                if first_token:
                    yield "</think>"
                    yield "<response>"
                    first_token = False
                yield chunk
            yield "</response>"
            logger.info(f"Cost info dict: {cost_info}")
            # Update the accumulated cost in chat_session
            if chat_session is not None and "total_cost" in cost_info:
                chat_session.update_cost(cost_info["total_cost"])
                logger.info(f"Updated session cost: ${chat_session.get_accumulated_cost():.6f}")
        return o4mini_stream_response(chain, user_prompt, stream)
    else:
        # Return just the content string from the complete response
        response = chain.invoke({"user_input": user_prompt})
        return response


# Example usage
if __name__ == "__main__":
    # # Example with DeepSeek streaming
    # print("DeepSeek streaming response:")
    # # Use DeepSeek-R1 model
    # stream_response = deepseek_inference("what is 1+1?", stream=True, model="DeepSeek-R1")
    # for chunk in stream_response:
    #     print(chunk, end="", flush=True)

    # # Example with DeepSeek streaming
    # print("DeepSeek streaming response:")
    # # Use DeepSeek-R1-Distill-Llama-70B model
    # stream_response = deepseek_inference("what is 1+1?", stream=True, model="DeepSeek-R1-Distill-Llama-70B")
    # for chunk in stream_response:
    #     print(chunk, end="", flush=True)

    # # Example with O3-mini streaming
    # print("\nO3-mini streaming response:")
    # stream_response = o3mini_inference("what is 1+1? Explain it in detail and deep-thinking way", stream=True)
    # for chunk in stream_response:
    #     print(chunk, end="", flush=True)

    stream_response = deep_inference_agent(user_prompt="what is 1+1? Explain it in detail and deep-thinking way", stream=True)
    for chunk in stream_response:
        print(chunk, end="", flush=True)