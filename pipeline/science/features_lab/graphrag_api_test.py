"""Test script for GraphRAG API usage.

This script demonstrates how to use GraphRAG for:
1. Generating embeddings from text documents
2. Building a knowledge graph
3. Retrieving responses using the graph-based RAG approach
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
import asyncio

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
import graphrag.api as api
from graphrag.config.create_graphrag_config import create_graphrag_config

from pipeline.science.pipeline.utils import tiktoken, deepseek_inference
from pipeline.science.pipeline.helper.index_files_saving import (
    graphrag_index_files_check,
    graphrag_index_files_compress,
    graphrag_index_files_decompress,
)

import logging
logger = logging.getLogger("graphrag_api_test.py")

def setup_graphrag_environment(embedding_folder: str) -> None:
    """Set up the GraphRAG environment with necessary configurations.

    Args:
        embedding_folder: Path to the folder where embeddings will be stored
    """
    # Load environment variables
    load_dotenv()

    # Create GraphRAG folder structure
    graphrag_folder = os.path.join(embedding_folder, "GraphRAG")
    os.makedirs(graphrag_folder, exist_ok=True)
    os.makedirs(os.path.join(graphrag_folder, "input"), exist_ok=True)
    os.makedirs(os.path.join(graphrag_folder, "output"), exist_ok=True)

async def generate_embeddings(document_text: str, embedding_folder: str) -> bool:
    """Generate GraphRAG embeddings for the given document.

    Args:
        document_text: Text content to generate embeddings for
        embedding_folder: Path to store the embeddings

    Returns:
        bool: True if embedding generation was successful
    """
    # Save document text to input folder
    input_file = os.path.join(embedding_folder, "GraphRAG/input/document.txt")
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(document_text)

    # Create GraphRAG config
    settings = {
        "encoding_model": "cl100k_base",
        "llm": {
            "api_key": os.getenv("GRAPHRAG_API_KEY"),
            "type": "azure_openai_chat",
            "api_base": os.getenv("GRAPHRAG_API_BASE"),
            "api_version": os.getenv("GRAPHRAG_API_VERSION"),
            "deployment_name": os.getenv("GRAPHRAG_LLM_MODEL"),
            "model": os.getenv("GRAPHRAG_LLM_MODEL"),
            "model_supports_json": True
        },
        "embeddings": {
            "async_mode": "threaded",
            "vector_store": {
                "type": "lancedb",
                "db_uri": "output/lancedb",
                "container_name": "default",
                "overwrite": True
            },
            "llm": {
                "api_key": os.getenv("GRAPHRAG_API_KEY"),
                "type": "azure_openai_embedding",
                "api_base": os.getenv("GRAPHRAG_API_BASE"),
                "api_version": os.getenv("GRAPHRAG_API_VERSION"),
                "deployment_name": "text-embedding-3-small",
                "model": "text-embedding-3-small"
            }
        },
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": "input",
            "file_encoding": "utf-8",
            "file_pattern": ".*\\.txt$"
        }
    }

    graphrag_config = create_graphrag_config(
        values=settings,
        root_dir=os.path.join(embedding_folder, "GraphRAG")
    )

    try:
        await api.build_index(config=graphrag_config)
        return True
    except Exception as e:
        logger.info(f"Error generating embeddings: {e}")
        return False

def get_response(query: str, chat_history: str, embedding_folder: str) -> str:
    """Get response using GraphRAG's global search functionality.

    Args:
        query: User's question
        chat_history: Previous conversation history
        embedding_folder: Path to the embeddings folder

    Returns:
        str: Generated response
    """
    # Initialize LLM
    llm = ChatOpenAI(
        api_key=os.getenv("GRAPHRAG_API_KEY"),
        api_base=os.getenv("GRAPHRAG_API_BASE"),
        api_version=os.getenv("GRAPHRAG_API_VERSION"),
        model=os.getenv("GRAPHRAG_LLM_MODEL"),
        api_type=OpenaiApiType.AzureOpenAI,
        max_retries=20,
    )
    token_encoder = tiktoken.encoding_for_model(os.getenv("GRAPHRAG_LLM_MODEL"))

    # Load GraphRAG index files
    input_dir = os.path.join(embedding_folder, "GraphRAG/output")
    community_df = pd.read_parquet(f"{input_dir}/create_final_communities.parquet")
    entity_df = pd.read_parquet(f"{input_dir}/create_final_nodes.parquet")
    report_df = pd.read_parquet(f"{input_dir}/create_final_community_reports.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/create_final_entities.parquet")

    # Initialize GraphRAG components
    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, 2)  # Level 2 community hierarchy
    entities = read_indexer_entities(entity_df, entity_embedding_df, 2)

    # Set up context builder
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
        "max_tokens": 12000,
        "context_name": "Reports",
    }

    # Configure search parameters
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    # Initialize search engine
    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    # Generate response
    search_result = search_engine.search(
        f"""
        You are a patient and honest professor helping a student reading a paper.
        The student asked the following question:
        ```{query}```
        Use the given context to answer the question.
        Previous conversation history:
        ```{chat_history}```
        """
    )

    # Process and format response
    context = str(search_result.context_data["reports"])
    prompt = f"""
    The previous conversation is: {chat_history}
    Reference context from the paper: {context}
    The user's query is: {query}
    """
    answer = str(deepseek_inference(prompt))

    # Extract thinking process and summary
    answer_thinking = answer.split("<think>")[1].split("</think>")[0]
    answer_summary = answer.split("<think>")[1].split("</think>")[1]

    # return f"### Here is my thinking process\n\n{answer_thinking}\n\n### Here is my summarized answer\n\n{answer_summary}"
    return answer_summary

async def main():
    """Main function to demonstrate GraphRAG usage."""
    # Example usage
    embedding_folder = "test_embeddings"
    document_text = """
    This is a sample document for testing GraphRAG embeddings.
    It contains information about machine learning and natural language processing.
    GraphRAG uses a graph-based approach for better context understanding.
    """

    # Set up environment
    setup_graphrag_environment(embedding_folder)

    # Generate embeddings
    success = await generate_embeddings(document_text, embedding_folder)
    if success:
        print("Successfully generated embeddings")

        # Test response generation
        query = "How does GraphRAG improve context understanding?"
        chat_history = "Previous conversation about NLP models."

        response = get_response(query, chat_history, embedding_folder)
        print("\nGenerated Response:")
        print(response)
    else:
        print("Failed to generate embeddings")

if __name__ == "__main__":
    asyncio.run(main())
