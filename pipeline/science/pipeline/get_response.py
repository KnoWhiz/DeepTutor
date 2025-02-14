import os
import json
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    tiktoken,
    truncate_chat_history,
    get_llm,
    get_embedding_models,
    translate_content,
    responses_refine,
    detect_language,
    generate_course_id,
    save_file_txt_locally,
    process_pdf_file,
)
from pipeline.science.pipeline.doc_processor import (
    generate_embedding,
)
from pipeline.science.pipeline.sources_retrieval import (
    get_response_source,
)
from pipeline.science.pipeline.inference import deepseek_inference
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.helper.index_files_saving import (
    vectorrag_index_files_decompress,
    vectorrag_index_files_compress,
    graphrag_index_files_decompress,
    graphrag_index_files_compress,
)

import logging
logger = logging.getLogger("tutorpipeline.science.get_response")

def tutor_agent(chat_session: ChatSession, file_path, user_input):
    """
    Taking the user input, document, and chat history, generate a response and sources.
    If user_input is None, generates the initial welcome message.
    """
    # Compute hashed ID and prepare embedding folder
    file_hash = generate_course_id(file_path)
    course_id = file_hash
    embedding_folder = os.path.join('embedded_content', course_id)
    print(f"Embedding folder: {embedding_folder}")
    if not os.path.exists('embedded_content'):
        os.makedirs('embedded_content')
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    # Save the file txt content locally
    file = open(file_path, 'rb')
    file_bytes = file.read()
    filename = os.path.basename(file_path)
    save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
    # Process file and create session states for document and PDF object
    _document, _doc = process_pdf_file(file_path)

    if chat_session.mode == ChatMode.BASIC:
        print("Basic (VectorRAG) mode")
        # Doc processing
        if(vectorrag_index_files_decompress(embedding_folder)):
            print("VectorRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder)
            if(vectorrag_index_files_compress(embedding_folder)):
                print("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder)
                if(vectorrag_index_files_compress(embedding_folder)):
                    print("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    print("Error compressing and uploading VectorRAG index files to Azure Blob Storage.")
    elif chat_session.mode == ChatMode.ADVANCED:
        print("Advanced (GraphRAG) mode")
        if(graphrag_index_files_decompress(embedding_folder)):
            print("GraphRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder)
            # asyncio.run(generate_GraphRAG_embedding(document, embedding_folder=embedding_folder))
            if(graphrag_index_files_compress(embedding_folder)):
                print("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder)
                # asyncio.run(generate_GraphRAG_embedding(document, embedding_folder=embedding_folder))
                if(graphrag_index_files_compress(embedding_folder)):
                    print("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    print("Error compressing and uploading GraphRAG index files to Azure Blob Storage.")
    else:
        print("Error: Invalid mode")

    chat_history = chat_session.chat_history

    # Use temporary chat history for follow-up questions if available
    if hasattr(chat_session, 'temp_chat_history') and chat_session.temp_chat_history:
        context_chat_history = chat_session.temp_chat_history
        # Clear the temporary chat history after using it
        chat_session.temp_chat_history = None
    else:
        context_chat_history = chat_history

    # Handle initial welcome message when chat history is empty
    # FIXME: uncomment this block after chat history is implemented
    if not chat_history:
        try:
            # Try to load existing document summary
            document_summary_path = os.path.join(embedding_folder, "document_summary.txt")
            with open(document_summary_path, "r") as f:
                initial_message = f.read()
        except FileNotFoundError:
            initial_message = "Hello! How can I assist you today?"

        answer = initial_message
        # Translate the initial message to the selected language
        answer = translate_content(
            content=answer,
            target_lang=chat_session.current_language
        )
        sources = {}  # Return empty dictionary for sources
        source_pages = {}
        return answer, sources, source_pages

    # Regular chat flow
    # Refine user input
    refined_user_input = get_query_helper(chat_session, user_input, context_chat_history, embedding_folder)
    logger.info(f"Refined user input: {refined_user_input}")
    # Get response
    answer = get_response(chat_session, _doc, _document, file_path, refined_user_input, context_chat_history, embedding_folder)
    # Get sources
    sources, source_pages = get_response_source(
        'Advanced' if chat_session.mode == ChatMode.ADVANCED else 'Basic',
        _doc, _document, file_path, refined_user_input, answer, context_chat_history, embedding_folder
    )

    images_sources = {}
    # If the sources have images, append the image URL (in image_urls.json mapping) to the end of the answer in markdown format
    if sources:
        image_url_path = os.path.join(embedding_folder, "markdown/image_urls.json")
        if os.path.exists(image_url_path):
            with open(image_url_path, 'r') as f:
                image_url_mapping = json.load(f)
        else:
            print("image_url_path does not exist")
            image_url_mapping = {}
            with open(image_url_path, 'w') as f:
                json.dump(image_url_mapping, f)

        # Process each source and check if it's an image
        sources_to_remove = []
        for source, score in sources.items():
            if any(source.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']):
                image_url = image_url_mapping.get(source, None)
                if image_url:
                    images_sources[source] = score
                    sources_to_remove.append(source)

        # Remove processed image sources from the main sources dict
        for source in sources_to_remove:
            del sources[source]

    answer = f"""Are you asking: **{refined_user_input}**
    """ + "\n" + answer

    # Translate the answer to the selected language
    answer = translate_content(
        content=answer,
        target_lang=chat_session.current_language
    )

    # Append images URL in markdown format to the end of the answer
    if images_sources:
        for source, _ in images_sources.items():
            image_url = image_url_mapping.get(source)
            if image_url:
                answer += "\n"
                answer += f"![]({image_url})"

    # Combine regular sources with image sources
    sources.update(images_sources)
    return answer, sources, source_pages


def get_response(chat_session: ChatSession, _doc, _document, file_path, user_input, chat_history, embedding_folder, deep_thinking = True):
    if chat_session.mode == ChatMode.ADVANCED:
        try:
            answer = get_GraphRAG_global_response(_doc, _document, user_input, chat_history, embedding_folder)
            return answer
        except Exception as e:
            print("Error getting response from GraphRAG:", e)

    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Check if all necessary files exist to load the embeddings
    generate_embedding(
        'Advanced' if chat_session.mode == ChatMode.ADVANCED else 'Basic',
        _document, _doc, file_path, embedding_folder
    )

    # Load existing embeddings
    print("Loading existing embeddings...")
    db = FAISS.load_local(
        embedding_folder, embeddings, allow_dangerous_deserialization=True
    )

    config = load_config()
    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

    if not deep_thinking:
        system_prompt = (
            """
            You are a patient and honest professor helping a student reading a paper.
            Use the given context to answer the question.
            If you don't know the answer, say you don't know.
            The previous conversation is: {chat_history} make sure your answer follow the previous conversation but not repetitive.
            Reference context from the paper: ```{context}```
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
            Student's query is: {input}
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
    else:
        chat_history_string = truncate_chat_history(chat_history)
        user_input_string = str(user_input)
        # Get relevant chunks for both question and answer with scores
        question_chunks_with_scores = db.similarity_search_with_score(user_input, k=config['retriever']['k'])
        # The total list of sources chunks
        sources_chunks = []
        for chunk in question_chunks_with_scores:
            sources_chunks.append(chunk[0])
        context = "\n\n".join([chunk.page_content for chunk in sources_chunks])

        prompt = f"\
        The previous conversation is: {chat_history_string}\
        Reference context from the paper: {context}\
        The user's query is: {user_input_string}\
        "
        answer = str(deepseek_inference(prompt))

        # Extract the content between <think> and </think> as answer_thinking, and the rest as answer_summary
        answer_thinking = answer.split("<think>")[1].split("</think>")[0]
        answer_summary = answer.split("<think>")[1].split("</think>")[1]
        answer_summary = responses_refine(answer_summary, "")
        answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary
    return answer


def get_GraphRAG_global_response(_doc, _document, user_input, chat_history, embedding_folder):
    # Chat history and user input
    chat_history_text = truncate_chat_history(chat_history)
    user_input_text = str(user_input)

    # Search for the document in the GraphRAG embedding
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

    search_engine_result = search_engine.search(
        f"""
        You are a patient and honest professor helping a student reading a paper.
        The student asked the following question:
        ```{user_input_text}```
        Use the given context to answer the question.
        Previous conversation history:
        ```{chat_history_text}```
        """,
    )
    context = search_engine_result.context_data["reports"]
    context = str(context)
    prompt = f"""
    The previous conversation is: {chat_history_text}
    Reference context from the paper: {context}
    The user's query is: {user_input_text}
    """
    answer = str(deepseek_inference(prompt))

    # Extract the content between <think> and </think> as answer_thinking, and the rest as answer_summary. but there is no <answer> tag in the answer, so after the answer_thinking extract the rest of the answer
    answer_thinking = answer.split("<think>")[1].split("</think>")[0]
    answer_summary = answer.split("<think>")[1].split("</think>")[1]
    answer_summary = responses_refine(search_engine_result.response, answer_summary)
    answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary

    return answer


def get_query_helper(chat_session: ChatSession, user_input, context_chat_history, embedding_folder):
    # If we have "document_summary" in the embedding folder, we can use it to speed up the search
    document_summary_path = os.path.join(embedding_folder, "document_summary.txt")
    if os.path.exists(document_summary_path):
        with open(document_summary_path, "r") as f:
            document_summary = f.read()
    else:
        document_summary = " "

    # Load languages from config
    config = load_config()
    llm = get_llm('basic', config['llm'])
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = (
        """
        You are a educational professor helping a student reading a document {context}.
        The goals are:
        1. to ask questions in a better way to make sure it's optimized to query a Vector Database for RAG (Retrieval Augmented Generation).
        2. to identify the question is about local or global context of the document.
        3. refer to the previous conversation history when generating the question.

        Previous conversation history:
        ```{chat_history}```

        Organize final response in the following JSON format:
        ```json
        {{
            "question": "<question try to understand what the user really mean by the question and rephrase it in a better way>",
            "question_type": "<local/global>",
        }}
        ```
        """
    )
    human_prompt = (
        """
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
                                  "context": document_summary,
                                  "chat_history": truncate_chat_history(context_chat_history)})
    question = parsed_result['question']
    question_type = parsed_result['question_type']
    language = detect_language(user_input)
    print("language detected:", language)

    chat_session.set_language(language)

    # # TEST
    # print("question rephrased:", question)
    return question


def generate_follow_up_questions(answer, chat_history):
    """
    Generate 3 relevant follow-up questions based on the assistant's response and chat history.
    """
    config = load_config()
    para = config['llm']
    llm = get_llm('basic', para)
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    system_prompt = """
    You are an expert at generating engaging follow-up questions based on a conversation between a tutor and a student.
    Given the tutor's response and chat history, generate 3 relevant follow-up questions that would help the student:
    1. Deepen their understanding of the topic
    2. Explore related concepts
    3. Apply the knowledge in practical ways

    The questions should be:
    - Clear and specific
    - Short and concise, no more than 10 words
    - Engaging and thought-provoking
    - Not repetitive with previous questions
    - Written in a way that encourages critical thinking

    Organize your response in the following JSON format:
    ```json
    {{
        "questions": [
            "<question 1>",
            "<question 2>",
            "<question 3>"
        ]
    }}
    ```
    """

    human_prompt = """
    Previous conversation history:
    ```{chat_history}```

    Tutor's last response:
    ```{answer}```
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    chain = prompt | llm | error_parser
    result = chain.invoke({
        "answer": answer,
        "chat_history": truncate_chat_history(chat_history)
    })

    return result["questions"]