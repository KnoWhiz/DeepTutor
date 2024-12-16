import os
import yaml
import fitz
import asyncio
import tiktoken
import pandas as pd

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


from streamlit_float import *

from pipeline.api_handler import ApiHandler
from pipeline.api_handler import create_env_file


@st.cache_resource
def count_tokens(text, model_name='gpt-4o'):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


@st.cache_resource
def truncate_chat_history(chat_history, max_tokens=2000, model_name='gpt-4o'):
    total_tokens = 0
    truncated_history = []
    for message in reversed(chat_history):
        message = str(message)
        message_tokens = count_tokens(message, model_name)
        if total_tokens + message_tokens > max_tokens:
            break
        truncated_history.insert(0, message)
        total_tokens += message_tokens
    return truncated_history


@st.cache_resource
def regen_with_graphrag():
    # Implement the logic to regenerate the response using GraphRAG here.
    st.warning("Regen with GraphRAG is not implemented yet.")


@st.cache_resource
def regen_with_longer_context():
    # Implement the logic to regenerate the response using a longer context here.
    st.warning("Regen with longer context is not implemented yet.")


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


@st.cache_resource
def generate_embedding(_documents, embedding_folder):
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
        texts = text_splitter.split_documents(_documents)
        print(f"length of document chunks generated for get_response_source:{len(texts)}")

        # Create the vector store to use as the index
        db = FAISS.from_documents(texts, embeddings)
        # Save the embeddings to the specified folder
        db.save_local(embedding_folder)

    return


@st.cache_resource
def generate_GraphRAG_embedding(_documents, embedding_folder):
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
    llm = get_llm('advance', para)
    embeddings = get_embedding_models('default', para)

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
        print("All necessary index files exist. Loading existing GraphRAG embeddings...")
    else:
        # Create the GraphRAG embedding
        print("Creating new GraphRAG embeddings...")

        # Initialize the project
        create_env_file(GraphRAG_embedding_folder)
        try:
            # Ensure an event loop is available
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            # Call the function with the desired path
            initialize_project_at(Path(GraphRAG_embedding_folder))
        except Exception as e:
            print("Initialization error:", e)
        settings = yaml.safe_load(open("./pipeline/graphrag_settings.yaml"))
        graphrag_config = create_graphrag_config(
            values=settings, root_dir=GraphRAG_embedding_folder
        )
        # Create output directory if it doesn't exist
        os.makedirs(GraphRAG_embedding_folder+'input', exist_ok=True)

        # Get all PDF files in the input_files directory
        pdf_files = [f for f in os.listdir('./input_files') if f.endswith('.pdf')]

        # Convert each PDF to a text file called 'texxt i.txt'
        for i, pdf_file in enumerate(pdf_files, start=1):
            pdf_path = os.path.join('./input_files', pdf_file)
            txt_path = os.path.join(GraphRAG_embedding_folder+'input', f'text {i}.txt')

            # Open the PDF file
            with fitz.open(pdf_path) as pdf_document:
                text = ""
                for page in pdf_document:
                    text += page.get_text()
            
            # Save the text to a file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

        # Create the GraphRAG embedding
        async def build_index_async(api, graphrag_config):
            index_result: list[PipelineRunResult] = await api.build_index(config=graphrag_config)
            return index_result

        # Assuming api and graphrag_config are already defined
        index_result = asyncio.run(build_index_async(api, graphrag_config))

        # index_result is a list of workflows that make up the indexing pipeline that was run
        for workflow_result in index_result:
            status = f"error\n{workflow_result.errors}" if workflow_result.errors else "success"
            print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")

    return


@st.cache_resource
def get_response(_documents, user_input, chat_history, embedding_folder):
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
    llm = get_llm('advance', para)
    parser = StrOutputParser()
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
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
    retriever = db.as_retriever(search_kwargs={"k":2})

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
        Since I am a student with no related knowledge backgroud, 
        please provide a concise answer and directly answer the question in easy to understand language.
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


@st.cache_resource
def get_response_source(_documents, user_input, chat_history, embedding_folder):
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
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
    retriever = db.as_retriever(search_kwargs={"k":2})

    # Create the RetrievalQA chain
    system_prompt = (
        """
        You are a honest professor helping a student reading a paper.
        For the given question about the context,
        find and provide sources that are helpful for answering the question.
        Use direct sentences or paragraphs from the context that are related to the question.
        ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. 
        DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING.
        Make as comprehensive a list of all the things that might be helpful

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
        My question is: {input}. I want to find the sources sentences that are helpful for answering the question.
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
    parsed_result = chain.invoke({"input": user_input})
    sources = parsed_result['answer']['sources']
    return sources