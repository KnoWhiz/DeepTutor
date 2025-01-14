import os
import yaml
import fitz
import asyncio
import tiktoken
import pandas as pd
import json
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

from streamlit_float import *

from pipeline.api_handler import ApiHandler
from pipeline.api_handler import create_env_file
from pipeline.helper.index_files_saving import index_files_check, index_files_compress, index_files_decompress


def count_tokens(text, model_name='gpt-4o'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


# @st.cache_resource
def truncate_chat_history(chat_history, model_name='gpt-4o'):
    """
    Only keep the messages from \"assistant\" and \"User\" that fit within the token limit.
    """
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        if(message['role'] == 'assistant' or message['role'] == 'user'):
            message = str(message)
            message_tokens = count_tokens(message, model_name)
            if total_tokens + message_tokens > max_tokens:
                break
            truncated_history.insert(0, message)
            total_tokens += message_tokens
    return truncated_history


# @st.cache_resource
def truncate_document(_document, model_name='gpt-4o'):
    """
    Only keep the beginning part of the document that fit within the token limit.
    """
    config = load_config()
    para = config['llm']
    api = ApiHandler(para)
    if model_name == 'gpt-4o':
        model_level = 'advance'
    else:
        model_level = 'basic'
    max_tokens = int(api.models[model_level]['context_window']/1.2)
    _document = str(_document)
    document_tokens = count_tokens(_document, model_name)
    if document_tokens > max_tokens:
        _document = _document[:max_tokens]
    return _document


@st.cache_resource
def get_llm(llm_type, para):
    para = para
    api = ApiHandler(para)
    llm_basic = api.models['basic']['instance']
    llm_advance = api.models['advance']['instance']
    llm_creative = api.models['creative']['instance']
    if llm_type == 'basic':
        return llm_basic
    elif llm_type == 'advance':
        return llm_advance
    elif llm_type == 'creative':
        return llm_creative
    return llm_basic


@st.cache_resource
def get_embedding_models(embedding_model_type, para):
    para = para
    api = ApiHandler(para)
    embedding_model_default = api.embedding_models['default']['instance']
    if embedding_model_type == 'default':
        return embedding_model_default
    else:
        return embedding_model_default


# @st.cache_resource
def generate_embedding(_documents, embedding_folder):
    config = load_config()
    para = config['llm']
    embeddings = get_embedding_models('default', para)

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")
    documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path) and os.path.exists(documents_summary_path):
        # Load existing embeddings
        print("Loading existing embeddings...")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Split the documents into chunks
        print("Creating new embeddings...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['embedding']['chunk_size'],
            chunk_overlap=config['embedding']['chunk_overlap']
        )
        texts = text_splitter.split_documents(_documents)
        print(f"length of document chunks generated for get_response_source:{len(texts)}")

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

        # Generate and save document summary
        generate_document_summary(_documents, embedding_folder)

    return


# @st.cache_resource
async def generate_GraphRAG_embedding(_documents, embedding_folder):
    GraphRAG_embedding_folder = os.path.join(embedding_folder, "GraphRAG/")
    create_final_community_reports_path = GraphRAG_embedding_folder + "output/create_final_community_reports.parquet"
    create_final_covariates_path = GraphRAG_embedding_folder + "output/create_final_covariates.parquet"
    create_final_documents_path = GraphRAG_embedding_folder + "output/create_final_documents.parquet"
    create_final_entities_path = GraphRAG_embedding_folder + "output/create_final_entities.parquet"
    create_final_nodes_path = GraphRAG_embedding_folder + "output/create_final_nodes.parquet"
    create_final_relationships_path = GraphRAG_embedding_folder + "output/create_final_relationships.parquet"
    create_final_text_units_path = GraphRAG_embedding_folder + "output/create_final_text_units.parquet"
    create_final_communities_path = GraphRAG_embedding_folder + "output/create_final_communities.parquet"
    lancedb_path = GraphRAG_embedding_folder + "output/lancedb/"
    path_list = [
        create_final_community_reports_path,
        create_final_covariates_path,
        create_final_documents_path,
        create_final_entities_path,
        create_final_nodes_path,
        create_final_relationships_path,
        create_final_text_units_path,
        create_final_communities_path,
        lancedb_path
    ]

    # Check if all necessary paths in path_list exist
    if all([os.path.exists(path) for path in path_list]):
        # Load existing embeddings
        print("All necessary index files exist. Loading existing knowledge graph embeddings...")
    else:
        # Create the GraphRAG embedding
        print("Creating new knowledge graph embeddings...")

        # Initialize the project
        create_env_file(GraphRAG_embedding_folder)
        try:
            """Initialize the project at the given path."""
            # Initialize the project
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
        except Exception as e:
            print("Initialization error:", e)
        settings = yaml.safe_load(open("./pipeline/graphrag_settings.yaml"))
        graphrag_config = create_graphrag_config(
            values=settings, root_dir=GraphRAG_embedding_folder
        )

        try:
            await api.build_index(config=graphrag_config)
        except Exception as e:
            print("Index building error:", e)

    return


# @st.cache_resource
def get_response(mode, _documents, user_input, chat_history, embedding_folder):
    if mode == 'Professor':
        try:
            answer = get_GraphRAG_global_response(_documents, user_input, chat_history, embedding_folder)
            return answer
        except Exception as e:
            print("Error getting response from GraphRAG:", e)

    config = load_config()
    para = config['llm']
    llm = get_llm('advance', para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Check if all necessary files exist to load the embeddings
    generate_embedding(_documents, embedding_folder)

    # Load existing embeddings
    print("Loading existing embeddings...")
    db = FAISS.load_local(
        embedding_folder, embeddings, allow_dangerous_deserialization=True
    )

    # Expose this index in a retriever interface
    # retriever = db.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.8}
    # )
    config = load_config()
    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

    # Create the RetrievalQA chain
    system_prompt = (
        """
        You are a patient and honest professor helping a student reading a paper.
        Use the given context to answer the question.
        If you don't know the answer, say you don't know.
        Context: ```{context}```
        If the concept can be better explained by formulas, use LaTeX syntax in markdown
        For inline formulas, use single dollar sign: $a/b = c/d$
        FOr block formulas, use double dollar sign:
        $$
        \frac{{a}}{{b}} = \frac{{c}}{{d}}
        $$
        """
    )
    human_prompt = (
        """
        Our previous conversation is: {chat_history}
        This time my query is: {input}
        Answer the question based on the context provided.
        Since I am a student with no related knowledge background, 
        provide a concise answer and directly answer the question in easy to understand language.
        Use markdown syntax for bold formatting to highlight important points or words.
        Use emojis when suitable to make the answer more engaging and interesting.
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],  # input query
            "chat_history": lambda x: x["chat_history"],  # chat history
            "context": lambda x: format_docs(x["context"]),  # context
        }
        | prompt  # format query and context into prompt
        | llm  # generate response
        | error_parser  # parse response
    )
    # Pass input query to retriever
    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    parsed_result = chain.invoke({"input": user_input, "chat_history": truncate_chat_history(chat_history)})
    answer = parsed_result['answer']
    return answer


# @st.cache_resource
def get_GraphRAG_global_response(_documents, user_input, chat_history, embedding_folder):
    # Chat history and user input
    chat_history_text = truncate_chat_history(chat_history)
    user_input_text = str(user_input)

    # Search for the documents in the GraphRAG embedding
    try:
        load_dotenv(".env")
    except Exception as e:
        print("Error loading .env file:", e)
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = os.getenv("GRAPHRAG_LLM_MODEL")
    api_base = os.getenv("GRAPHRAG_API_BASE")
    api_version = os.getenv("GRAPHRAG_API_VERSION")

    # print("api_key", api_key)

    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        model=llm_model,
        api_type=OpenaiApiType.AzureOpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )
    token_encoder = tiktoken.encoding_for_model(llm_model)

    INPUT_DIR = os.path.join(embedding_folder, "GraphRAG/output")
    COMMUNITY_TABLE = "create_final_communities"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2
    community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )
    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    answer = search_engine.search(
        f"""
        You are a patient and honest professor helping a student reading a paper.
        The student asked the following question:
        ```{user_input_text}```
        Use the given context to answer the question.
        Previous conversation history:
        ```{chat_history_text}```
        """,
    )

    return answer.response


# @st.cache_resource
def get_response_source(_documents, user_input, answer, chat_history, embedding_folder):
    config = load_config()
    para = config['llm']
    llm = get_llm('advance', para)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Define the default filenames used by FAISS when saving
    faiss_path = os.path.join(embedding_folder, "index.faiss")
    pkl_path = os.path.join(embedding_folder, "index.pkl")

    # Check if all necessary files exist to load the embeddings
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        # Load existing embeddings
        print("Loading existing embeddings...")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # Split the documents into chunks
        print("Creating new embeddings...")
        # text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['embedding']['chunk_size'],
            chunk_overlap=config['embedding']['chunk_overlap']
        )
        texts = text_splitter.split_documents(_documents)
        print(f"length of document chunks generated for get_response_source:{len(texts)}")

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

    # Expose this index in a retriever interface
    # retriever = db.as_retriever(
    #     search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.8}
    # )
    config = load_config()
    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

    # Create the RetrievalQA chain
    system_prompt = (
        """
        You are a honest professor helping a student reading a paper.
        For the given question about the context,
        find and provide sources that are related to a given question or answer.
        Use direct sentences or paragraphs from the context that are related to the question or answer.
        ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS.
        DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING.
        Make as comprehensive a list of all the things that might be related

        Context: ```{context}```

        Organize final response in the following JSON format:

        ```json
        {{
            "sources": [
                <source_1>,
                <source_2>,
                ...
                <source_n>,
            ]
        }}
        ```
        """
    )
    human_prompt = (
        """
        For text: ```{input}```, I want to find the sources sentences that are related.
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],  # input query
            "context": lambda x: format_docs(x["context"]),  # context
        }
        | prompt  # format query and context into prompt
        | llm  # generate response
        | error_parser  # parse response
    )
    # Pass input query to retriever
    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    parsed_result_question = chain.invoke({"input": user_input})
    sources_question = parsed_result_question['answer']['sources']

    # Pass answer to retriever
    retrieve_docs = (lambda x: x["input"]) | retriever
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )
    parsed_result_answer = chain.invoke({"input": answer})
    sources_answer = parsed_result_answer['answer']['sources']

    # Combine sources from question and answer and make sure there are no duplicates
    # sources = sources_question + sources_answer
    sources = list(set(sources_question + sources_answer))
    return sources


# @st.cache_resource
def get_query_helper(user_input, chat_history, embedding_folder):
    # If we have "documents_summary" in the embedding folder, we can use it to speed up the search
    documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    if os.path.exists(documents_summary_path):
        with open(documents_summary_path, "r") as f:
            documents_summary = f.read()
    else:
        documents_summary = " "

    config = load_config()
    para = config['llm']
    llm = get_llm('basic', para)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = (
        """
        You are a educational professor helping a student reading a document {context}.
        The goals are:
        1. to ask questions in a better way.
        2. to identify the question is about local or global context of the document.

        Organize final response in the following JSON format:
        ```json
        {{
            "question": "<question rephrased in a better way>",
            "question_type": "<local/global>"
        }}
        ```
        """
    )
    human_prompt = (
        """
        Previous conversation history:
        ```{chat_history}```
        The student asked the following question:
        ```{input}```
        """
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    chain = prompt | llm | error_parser
    parsed_result = chain.invoke({"input": user_input,
                                  "context": documents_summary,
                                  "chat_history": truncate_chat_history(chat_history)})
    question = parsed_result['question']
    question_type = parsed_result['question_type']
    return question


def generate_document_summary(_documents, embedding_folder):
    """
    Generate a comprehensive markdown-formatted summary of the documents using multiple LLM calls.
    """
    config = load_config()
    para = config['llm']
    llm = get_llm('advance', para)  # Using advanced model for better quality

    # First generate the take-home message
    takehome_prompt = """
    Provide a single, impactful sentence that captures the most important takeaway from this document.
    
    Guidelines:
    - Be extremely concise and specific
    - Focus on the main contribution or finding
    - Use bold for key terms
    - Keep it to one sentence
    - Add a relevant emoji at the start of bullet points or the first sentence
    - For inline formulas use single $ marks: $E = mc^2$
    - For block formulas use double $$ marks:
      $$
      F = ma
      $$
    
    Document: {document}
    """
    takehome_prompt = ChatPromptTemplate.from_template(takehome_prompt)
    str_parser = StrOutputParser()
    takehome_chain = takehome_prompt | llm | str_parser
    takehome = takehome_chain.invoke({"document": truncate_document(_documents)})

    # Topics extraction
    topics_prompt = """
    Identify only the most essential topics/sections from this document.
    Be extremely selective and concise - only include major components.
    
    Return format:
    {{"topics": ["topic1", "topic2", ...]}}
    
    Guidelines:
    - Include maximum 4-5 topics
    - Focus only on critical sections
    - Use short, descriptive names
    
    Document: {document}
    """
    topics_prompt = ChatPromptTemplate.from_template(topics_prompt)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    topics_chain = topics_prompt | llm | error_parser
    topics_result = topics_chain.invoke({"document": truncate_document(_documents)})
    
    try:
        topics = topics_result.get("topics", [])
    except AttributeError:
        print("Warning: Failed to get topics. Using default topics.")
        topics = ["Overview", "Methods", "Results", "Discussion"]

    # Generate overview
    overview_prompt = """
    Provide a clear and engaging overview using bullet points.
    
    Guidelines:
    - Use 3-4 concise bullet points
    - **Bold** for key terms
    - Each bullet point should be one short sentence
    - For inline formulas use single $ marks: $E = mc^2$
    - For block formulas use double $$ marks:
      $$
      F = ma
      $$
    
    Document: {document}
    """
    overview_prompt = ChatPromptTemplate.from_template(overview_prompt)
    overview_chain = overview_prompt | llm | str_parser
    overview = overview_chain.invoke({"document": truncate_document(_documents)})

    # Generate summaries for each topic
    summaries = []
    for topic in topics:
        topic_prompt = """
        Provide an engaging summary for the topic "{topic}" using bullet points.
        
        Guidelines:
        - Use 2-3 bullet points
        - Each bullet point should be one short sentence
        - **Bold** for key terms
        - Use simple, clear language
        - Include specific metrics only if crucial
        - For inline formulas use single $ marks: $E = mc^2$
        - For block formulas use double $$ marks:
          $$
          F = ma
          $$
        
        Document: {document}
        """
        topic_prompt = ChatPromptTemplate.from_template(topic_prompt)
        topic_chain = topic_prompt | llm | str_parser
        topic_summary = topic_chain.invoke({
            "topic": topic,
            "document": truncate_document(_documents)
        })
        summaries.append((topic, topic_summary))

    # Combine everything into markdown format with welcome message and take-home message
    markdown_summary = f"""### 👋 Welcome to KnoWhiz Office Hour! 

I'm your AI tutor 🤖 ready to help you understand this document.

### 💡 Key Takeaway
{takehome}

### 📚 Document Overview
{overview}

"""
    
    # Add emojis for common topic titles
    topic_emojis = {
        "introduction": "📖",
        "overview": "🔎",
        "background": "📚",
        "methods": "🔬",
        "methodology": "🔬", 
        "results": "📊",
        "discussion": "💭",
        "conclusion": "🎯",
        "future work": "🔮",
        "implementation": "⚙️",
        "evaluation": "📈",
        "analysis": "🔍",
        "design": "✏️",
        "architecture": "🏗️",
        "experiments": "🧪",
        "related work": "🔗",
        "motivation": "💪",
        "approach": "🎯",
        "system": "🖥️",
        "framework": "🔧",
        "model": "🤖",
        "data": "📊",
        "algorithm": "⚡",
        "performance": "⚡",
        "limitations": "⚠️",
        "applications": "💡",
        "default": "📌" # Default emoji for topics not in the mapping
    }
    
    for topic, summary in summaries:
        # Get emoji based on topic, defaulting to 📌 if not found
        topic_lower = topic.lower()
        emoji = next((v for k, v in topic_emojis.items() if k in topic_lower), topic_emojis["default"])
        
        markdown_summary += f"""### {emoji} {topic}
{summary}

"""

    markdown_summary += """
---
### 💬 Ask Me Anything!
Feel free to ask me any questions about the document! I'm here to help! ✨
"""

    documents_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    with open(documents_summary_path, "w", encoding='utf-8') as f:
        f.write(markdown_summary)

    return markdown_summary
