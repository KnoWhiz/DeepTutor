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


from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv  
from typing import Iterable
# import os


def _format_thinking_delta(delta: str) -> str:
    """
    Only transform '**XXX' -> '\n\n**XXX'.
    If the chunk is exactly '**', or starts with '**' followed by a newline
    (e.g., '**\\n', '**\\r\\n') or only whitespace, treat it as a closing marker
    and do nothing.
    """
    if not delta:
        return delta

    if delta == "**":
        return delta

    if delta.startswith("**"):
        after = delta[2:]
        # If the very next char is a newline, or there's only whitespace after '**',
        # it's likely a closing '**' chunk -> leave unchanged.
        if after[:1] in ("\n", "\r") or after.strip() == "":
            return delta
        # Otherwise it's an opening '**Title' chunk -> add two leading newlines
        if not delta.startswith("\n\n**"):
            return "\n\n" + delta

    return delta


def stream_response_with_tags_detailed(**create_kwargs) -> Iterable[str]:
    """
    Yields a single XML-like stream:
      <think> ...reasoning summary + tool progress... </think><response> ...final answer... </response>
    With detailed tool calling updates inside <think>.
    """
    # load_dotenv(".env")
    # If OpenAI API
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # If AzureOpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        api_version="2025-03-01-preview",
    )
    stream = client.responses.create(stream=True, **create_kwargs)

    thinking_open = True
    response_open = False
    yield "<think>"

    try:
        for event in stream:
            t = event.type or ""

            # --- Reasoning summary stream ---
            if t == "response.reasoning_summary_text.delta":
                yield _format_thinking_delta(getattr(event, "delta", "") or "")

            elif t == "response.reasoning_summary_text.done":
                pass  # keep <think> open for tool progress

            # --- Output item lifecycle (covers tools like web_search, file_search, image_generation, etc.) ---
            elif t == "response.output_item.added":
                item = getattr(event, "item", None) or getattr(event, "output_item", None)
                item_type = getattr(item, "type", None) or getattr(event, "item_type", None)
                if item_type:
                    yield f"\n[tool:item-added type={item_type}]\n"

            elif t == "response.output_item.done":
                item = getattr(event, "item", None) or getattr(event, "output_item", None)
                item_type = getattr(item, "type", None) or getattr(event, "item_type", None)
                if item_type:
                    yield f"[tool:item-done type={item_type}]\n\n"

            # --- Built-in web_search progress stream ---
            elif t.startswith("response.web_search_call."):
                phase = t.split(".")[-1]  # e.g., 'in_progress', 'completed', possibly 'result'
                q = getattr(event, "query", None)
                if phase in ("created", "started", "searching", "in_progress"):
                    yield f"[web_search:{phase}{' q='+q if q else ''}]\n"
                elif phase == "result":
                    title = getattr(event, "title", None)
                    url = getattr(event, "url", None)
                    if title or url:
                        yield f"- {title or ''} {url or ''}\n"
                elif phase == "completed":
                    results = getattr(event, "results", None) or []
                    n = getattr(event, "num_results", None) or (len(results) if isinstance(results, list) else None)
                    yield f"[web_search:completed results={n if n is not None else 'unknown'}]\n\n"

            # --- Function calling (your own tools) ---
            elif t == "response.function_call_arguments.delta":
                yield getattr(event, "delta", "") or ""
            elif t == "response.function_call_arguments.done":
                yield "\n[function_call:args_done]\n"

            # --- Main model answer text ---
            elif t == "response.output_text.delta":
                if thinking_open:
                    yield "\n</think>\n\n"
                    thinking_open = False
                if not response_open:
                    response_open = True
                    yield "<response>\n\n"
                yield getattr(event, "delta", "") or ""

            elif t == "response.output_text.done":
                if response_open:
                    yield "\n\n</response>\n"
                    response_open = False

            # --- Finalization / errors ---
            elif t == "response.completed":
                if thinking_open:
                    yield "\n</think>\n"
                    thinking_open = False

            elif t == "response.error":
                if thinking_open:
                    yield "\n</think>\n"
                    thinking_open = False
                if response_open:
                    yield "\n</response>\n"
                    response_open = False
                err = getattr(event, "error", None)
                msg = getattr(err, "message", None) if err else None
                yield f"<!-- error: {msg or err or 'unknown'} -->"

            # else: ignore other event types

    finally:
        try:
            stream.close()
        except Exception:
            pass


def stream_response_with_tags(**create_kwargs) -> Iterable[str]:
    """
    Yields a single XML-like stream:
      <think> ...reasoning summary + tool progress... </think><response> ...final answer... </response>
    Without detailed tool calling updates inside <think>.
    """
    # load_dotenv(".env")
    # If OpenAI API
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # If AzureOpenAI API
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY_BACKUP"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_BACKUP"),
        api_version="2025-03-01-preview",
    )
    stream = client.responses.create(stream=True, **create_kwargs)

    # Show a thinking container immediately
    thinking_open = True
    response_open = False
    yield "<think>"

    try:
        for event in stream:
            t = event.type

            # --- Reasoning summary stream ---
            if t == "response.reasoning_summary_text.delta":
                yield _format_thinking_delta(event.delta)

            elif t == "response.reasoning_summary_text.done":
                # keep <think> open for tool progress; we'll close when answer starts or at the very end
                pass

            # --- Main model answer text ---
            elif t == "response.output_text.delta":
                if thinking_open:
                    yield "\n</think>\n\n"
                    thinking_open = False
                if not response_open:
                    response_open = True
                    yield "<response>\n\n"
                yield event.delta

            # ✅ Close <response> as soon as the model finishes its text
            elif t == "response.output_text.done":
                if response_open:
                    yield "\n\n</response>\n"
                    response_open = False

            # --- Finalization / errors ---
            elif t == "response.completed":
                # We may already have closed </response>; just ensure well-formed
                if thinking_open:
                    yield "\n</think>\n"
                    thinking_open = False

            elif t == "response.error":
                if thinking_open:
                    yield "\n</think>\n"
                    thinking_open = False
                if response_open:
                    yield "\n</response>\n"
                    response_open = False
                # Optionally surface the error:
                # yield f"<!-- error: {event.error} -->"

    finally:
        try:
            stream.close()
        except Exception:
            pass


# # ------------------------------
# # Example usage
# # ------------------------------
# if __name__ == "__main__":
#     kwargs = dict(
#         model="gpt-5",
#         # reasoning={"effort": "high", "summary": "detailed"},
#         reasoning={"effort": "medium", "summary": "auto"},
#         # reasoning={"effort": "low", "summary": "auto"},
#         tools=[{"type": "web_search"}],  # built-in tool
#         instructions=f"{system_prompt}\n\n You should search the web as needed (multiple searches OK) and cite sources.",
#         input=f"Context from the paper: {context_from_paper}\n\n What is this paper mainly about? Do web search if needed to find related multiplexing papers and compare with this paper.",
#     )

#     for chunk in stream_response_with_tags(**kwargs):
#         print(chunk, end="", flush=True)
#     print()