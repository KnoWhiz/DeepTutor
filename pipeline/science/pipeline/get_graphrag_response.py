import os
import pandas as pd
import tiktoken
import asyncio
from dotenv import load_dotenv

# GraphRAG imports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from pipeline.science.pipeline.session_manager import ChatSession
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    responses_refine,
    Question
)
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.rag_agent import get_rag_context

import logging
logger = logging.getLogger("tutorpipeline.science.get_graphrag_response")


async def get_GraphRAG_global_response(question: Question, chat_history, file_path_list, embedding_folder_list, deep_thinking = True, chat_session: ChatSession = None, stream: bool = False):
    user_input=question.text + "\n\n" + question.special_context
    config = load_config()
    token_limit = config["inference_token_limit"]
    map_symbol_to_index = config["map_symbol_to_index"]
    # Get the first 3 keys from map_symbol_to_index for examples in the prompt
    first_keys = list(map_symbol_to_index.keys())[:3]
    example_keys = ", or ".join(first_keys)
    
    embedding_folder = embedding_folder_list[0]

    # Chat history and user input
    chat_history_text = truncate_chat_history(chat_history)
    if chat_session is not None and chat_session.question is not None:
        user_input_text = chat_session.question.text + "\n\n" + str(chat_session.question.special_context)
        rag_user_input_text = user_input_text   # + "\n\n" + str(chat_session.question.answer_planning)
        logger.info(f"User input text is from Question object in chat session: {user_input_text}")
    else:
        user_input_text = str(user_input)
        logger.info(f"User input text is from user input: {user_input_text}")

    # Search for the document in the GraphRAG embeddings
    try:
        load_dotenv(".env")
    except Exception as e:
        logger.info("Error loading .env file:", e)
        raise
        
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
    api_base = os.getenv("GRAPHRAG_API_BASE")
    api_version = os.getenv("GRAPHRAG_API_VERSION")

    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        model=llm_model,
        api_type=OpenaiApiType.AzureOpenAI,
        max_retries=15,
    )
    token_encoder = tiktoken.encoding_for_model(llm_model)

    INPUT_DIR = os.path.join(embedding_folder, "GraphRAG/output")
    COMMUNITY_TABLE = "create_final_communities"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    COMMUNITY_LEVEL = 2
    community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    logger.info(f"Total report count: {len(report_df)}")
    logger.info(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=token_encoder,
    )
    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    # Directly await the search operation
    search_engine_result = await search_engine.asearch(
        f"""
        You are a patient and honest professor helping a student reading a paper.
        The student asked the following question:
        ```{rag_user_input_text}```
        Use the given context to answer the question.
        Previous conversation history:
        ```{chat_history_text}```
        This is a detailed plan for constructing the answer: {str(chat_session.question.answer_planning)}
        """,
    )
    
    context = search_engine_result.context_data["reports"]
    answer_string = search_engine_result.response
    context_string = str(answer_string)
    logger.info(f"TEST: formatted_context_string before get_rag_context: {chat_session.formatted_context}")
    formatted_context = await get_rag_context(chat_session=chat_session,
                                        file_path_list=file_path_list,
                                        question=question,
                                        chat_history=chat_history,
                                        embedding_folder_list=embedding_folder_list,
                                        deep_thinking=deep_thinking,
                                        stream=stream,
                                        context=context_string)
    # formatted_context_string = str(formatted_context)
    formatted_context_string = chat_session.formatted_context
    logger.info(f"TEST: formatted_context_string after get_rag_context: {chat_session.formatted_context}")
    prompt = f"""
    You are a deep thinking tutor helping a student reading a paper.
    This is the global context from the paper: {context_string}
    Reference context chunks with relevance scores from the paper: 
    ```
    {formatted_context_string}
    ```
    This is a detailed plan for constructing the answer: {str(chat_session.question.answer_planning)}
    The student's query is: {user_input_text}
    For formulas, use LaTeX format with $...$ or 
    $$
    ...
    $$
    and make sure latex syntax can be properly rendered in the response.

    Requirement:
    Only use the information from the context chunks to answer the question. Give the response in a scientific and academic tone. Do not make up or assume anything or guess without any evidence. If you answer some questions based on your own knowledge, clearly state that you are using your own knowledge.

    Format requirement:
    1. Make sure each sentence in the response there is a corresponding context chunk to support the sentence, and cite the most relevant context chunk keys in the format "[<chunk_key, like {example_keys}, etc>]" at the end of the sentence after the period mark. If there are more than one context chunk keys, use the format "[<chunk_key_1>][<chunk_key_2>] ..." to cite all the context chunk keys.
    2. Use markdown syntax for formatting the response to make it more clear and readable.
    """

    if deep_thinking:
        from pipeline.science.pipeline.inference import deep_inference_agent
        try:
            logger.info(f"Deep thinking and stream is {stream}, calling deep_inference_agent")
            logger.info(f"Prompt for advanced deep thinking: {prompt}")
            answer = deep_inference_agent(user_prompt=prompt, stream=stream, chat_session=chat_session)
        except Exception as e:
            logger.exception(f"Error in deep_inference_agent, trying standard deep thinking mode: {e}")
            prompt = f"""
            You are a deep thinking tutor helping a student reading a paper.
            Reference context chunks with relevance scores from the paper: 
            {formatted_context_string}
            This is a detailed plan for constructing the answer: {str(question.answer_planning)}
            The student's query is: {user_input_text}
            For formulas, use LaTeX format with $...$ or 
            $$
            ...
            $$
            and make sure latex syntax can be properly rendered in the response.

            Requirement:
            Only use the information from the context chunks to answer the question. Give the response in a scientific and academic tone. Do not make up or assume anything or guess without any evidence. If you answer some questions based on your own knowledge, clearly state that you are using your own knowledge.

            Format requirement:
            1. Make sure each sentence in the response there is a corresponding context chunk to support the sentence, and cite the most relevant context chunk keys in the format "[<chunk_key, like {example_keys}, etc>]" at the end of the sentence after the period mark. If there are more than one context chunk keys, use the format "[<chunk_key_1>][<chunk_key_2>] ..." to cite all the context chunk keys.
            2. Use markdown syntax for formatting the response to make it more clear and readable.
            """
            answer = deep_inference_agent(user_prompt=prompt, stream=stream, chat_session=chat_session)

        if stream is False:
            # If there is <think> in the answer, split it into thinking and summary
            if "<think>" in answer:
                answer_thinking = answer.split("<think>")[1].split("</think>")[0]
                answer_summary = answer.split("<think>")[1].split("</think>")[1]
                answer_summary_refined = responses_refine(search_engine_result.response, answer_summary, stream=stream)
                # answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary
                # answer = answer_summary + "\n\n" + answer_summary_refined
                answer = answer_summary_refined
            else:
                answer = responses_refine(answer, stream=stream)
        else:   # If stream is True, we need to return the response as a stream generator
            answer = answer
    else:
        if stream is False:
            answer = responses_refine(search_engine_result.response, stream=stream)
        else:
            # FIXME: We need to return the response as a stream generator
            answer = search_engine_result.response

    return answer