import os
import yaml
import time
from typing import Dict
from pathlib import Path

import logging
logger = logging.getLogger("tutorpipeline.science.embeddings_graphrag")

# GraphRAG imports
import graphrag.api as api
from graphrag.cli.initialize import initialize_project_at
from graphrag.index.typing import PipelineRunResult
from graphrag.config.create_graphrag_config import create_graphrag_config
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

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_relationships,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.config.init_content import INIT_DOTENV, INIT_YAML
from graphrag.prompts.index.claim_extraction import CLAIM_EXTRACTION_PROMPT
from graphrag.prompts.index.community_report import (
    COMMUNITY_REPORT_PROMPT,
)
from graphrag.prompts.index.entity_extraction import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
)
from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT

from pipeline.science.pipeline.api_handler import create_env_file
from pipeline.science.pipeline.utils import file_check_list


async def generate_GraphRAG_embedding(embedding_folder, time_tracking: Dict[str, float] = {}):
    # The embedding_folder is the folder end with file_id
    file_id = embedding_folder.split("/")[-1]
    check_list_start_time = time.time()
    GraphRAG_embedding_folder, path_list = file_check_list(embedding_folder)
    time_tracking['graphrag_check_list'] = time.time() - check_list_start_time
    logger.info("graphrag_check_list for %s completed in %.2fs", file_id, time_tracking['graphrag_check_list'])

    # Check if all necessary paths in path_list exist
    if all([os.path.exists(path) for path in path_list]):
        # Load existing embeddings
        logger.info("All necessary index files exist. Loading existing knowledge graph embeddings...")
        yield "\n\n**🗺️ All necessary index files exist. Loading existing knowledge graph embeddings...**"
    else:
        # Create the GraphRAG embeddings
        logger.info("Loading knowledge graph embeddings ...")
        yield "\n\n**🗺️ Loading knowledge graph embeddings ...**"
        # Initialize the project
        create_env_file_start_time = time.time()
        create_env_file(GraphRAG_embedding_folder)
        time_tracking['graphrag_create_env_file'] = time.time() - create_env_file_start_time
        logger.info("graphrag_create_env_file for %s completed in %.2fs", file_id, time_tracking['graphrag_create_env_file'])
        try:
            """Initialize the project at the given path."""
            # Initialize the project
            initialize_project_start_time = time.time()
            path = GraphRAG_embedding_folder
            root = Path(path)
            if not root.exists():
                root.mkdir(parents=True, exist_ok=True)
            dotenv = root / ".env"
            if not dotenv.exists():
                with dotenv.open("wb") as file:
                    file.write(INIT_DOTENV.encode(encoding="utf-8", errors="strict"))
            prompts_dir = root / "prompts"
            if not prompts_dir.exists():
                prompts_dir.mkdir(parents=True, exist_ok=True)
            prompts = {
                "entity_extraction": GRAPH_EXTRACTION_PROMPT,
                "summarize_descriptions": SUMMARIZE_PROMPT,
                "claim_extraction": CLAIM_EXTRACTION_PROMPT,
                "community_report": COMMUNITY_REPORT_PROMPT,
                "drift_search_system_prompt": DRIFT_LOCAL_SYSTEM_PROMPT,
                "global_search_map_system_prompt": MAP_SYSTEM_PROMPT,
                "global_search_reduce_system_prompt": REDUCE_SYSTEM_PROMPT,
                "global_search_knowledge_system_prompt": GENERAL_KNOWLEDGE_INSTRUCTION,
                "local_search_system_prompt": LOCAL_SEARCH_SYSTEM_PROMPT,
                "question_gen_system_prompt": QUESTION_SYSTEM_PROMPT,
            }
            for name, content in prompts.items():
                prompt_file = prompts_dir / f"{name}.txt"
                if not prompt_file.exists():
                    with prompt_file.open("wb") as file:
                        file.write(content.encode(encoding="utf-8", errors="strict"))
            time_tracking['graphrag_initialize_project'] = time.time() - initialize_project_start_time
            logger.info("graphrag_initialize_project for %s completed in %.2fs", file_id, time_tracking['graphrag_initialize_project'])
            # yield "\n\n**Initialized GraphRAG project successfully...**"
            logger.info("Initialized GraphRAG project successfully...")
        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            yield "\n\n**Initialization error: {e}**"
            
        create_graphrag_config_start_time = time.time()
        settings = yaml.safe_load(open("./pipeline/science/pipeline/graphrag_settings.yaml"))
        # logger.info(f"root_dir: {GraphRAG_embedding_folder}")
        # yield "\n\n**Creating GraphRAG config...**"
        logger.info("Creating GraphRAG config...")
        graphrag_config = create_graphrag_config(
            values=settings, root_dir=GraphRAG_embedding_folder
        )
        time_tracking['graphrag_create_config'] = time.time() - create_graphrag_config_start_time
        logger.info("graphrag_create_config for %s completed in %.2fs", file_id, time_tracking['graphrag_create_config'])
        # yield "\n\n**Created GraphRAG config successfully...**"
        logger.info("Created GraphRAG config successfully...")
        # graphrag_config.storage.base_dir = os.path.join(GraphRAG_embedding_folder, "output")
        # graphrag_config.reporting.base_dir = os.path.join(GraphRAG_embedding_folder, "logs")
        # # logger.info(f"graphrag_config before build: {graphrag_config}")

        try:
            yield "\n\n**🗺️ Loading GraphRAG index ...**"
            build_index_start_time = time.time()
            await api.build_index(config=graphrag_config)
            time_tracking['graphrag_build_index'] = time.time() - build_index_start_time
            logger.info("graphrag_build_index for %s completed in %.2fs", file_id, time_tracking['graphrag_build_index'])
            yield "\n\n**🗺️ GraphRAG index loaded successfully ...**"
            # logger.info(f"graphrag_config after build: {graphrag_config}")
        except Exception as e:
            logger.exception(f"Index loading error: {e}")

    # return time_tracking
    return
