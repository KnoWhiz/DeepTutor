import os
import json
import time
from typing import Dict

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
    get_embedding_models,
    translate_content,
    responses_refine,
    detect_language,
    generate_course_id,
    save_file_txt_locally,
    process_pdf_file,
    get_highlight_info,
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
from pipeline.science.pipeline.graphrag_get_response import get_GraphRAG_global_response
import logging
logger = logging.getLogger("tutorpipeline.science.get_response")


def format_time_tracking(time_tracking: Dict[str, float]) -> str:
    """
    Format time tracking dictionary into a readable string with appropriate units.
    
    Args:
        time_tracking: Dictionary of operation names and their durations in seconds
        
    Returns:
        Formatted string with times converted to appropriate units
    """
    formatted_times = []
    for operation, seconds in time_tracking.items():
        if seconds >= 60:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            formatted_times.append(f"{operation}: {minutes}m {remaining_seconds:.2f}s")
        else:
            formatted_times.append(f"{operation}: {seconds:.2f}s")
    
    return "\n".join(formatted_times)


async def tutor_agent(chat_session: ChatSession, file_path, user_input):
    """
    Taking the user input, document, and chat history, generate a response and sources.
    If user_input is None, generates the initial welcome message.
    """
    time_tracking: Dict[str, float] = {}

    # Compute hashed ID and prepare embedding folder
    hashing_start_time = time.time()
    file_hash = generate_course_id(file_path)
    course_id = file_hash
    embedding_folder = os.path.join('embedded_content', course_id)
    logger.info(f"Embedding folder: {embedding_folder}")
    if not os.path.exists('embedded_content'):
        os.makedirs('embedded_content')
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    time_tracking['file_hashing_setup_dirs'] = time.time() - hashing_start_time
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    file = open(file_path, 'rb')
    file_bytes = file.read()
    filename = os.path.basename(file_path)
    save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
    # Process file and create session states for document and PDF object
    _document, _doc = process_pdf_file(file_path)
    time_tracking['file_loading_save_text'] = time.time() - save_file_start_time
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    if chat_session.mode == ChatMode.BASIC:
        vectorrag_start_time = time.time()
        logger.info("Basic (VectorRAG) mode")
        # Doc processing
        if(vectorrag_index_files_decompress(embedding_folder)):
            logger.info("VectorRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            time_tracking = await generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            if(vectorrag_index_files_compress(embedding_folder)):
                logger.info("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                time_tracking = await generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                if(vectorrag_index_files_compress(embedding_folder)):
                    logger.info("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    logger.info("Error compressing and uploading VectorRAG index files to Azure Blob Storage.")
        time_tracking['vectorrag_generate_embedding'] = time.time() - vectorrag_start_time
        logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    elif chat_session.mode == ChatMode.ADVANCED:
        graphrag_start_time = time.time()
        logger.info("Advanced (GraphRAG) mode")
        if(graphrag_index_files_decompress(embedding_folder)):
            logger.info("GraphRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            time_tracking = await generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            # asyncio.run(generate_GraphRAG_embedding(embedding_folder=embedding_folder))
            if(graphrag_index_files_compress(embedding_folder)):
                logger.info("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                time_tracking = await generate_embedding(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                # asyncio.run(generate_GraphRAG_embedding(embedding_folder=embedding_folder))
                if(graphrag_index_files_compress(embedding_folder)):
                    logger.info("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    logger.info("Error compressing and uploading GraphRAG index files to Azure Blob Storage.")
        time_tracking['graphrag_generate_embedding'] = time.time() - graphrag_start_time
        logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    else:
        logger.info("Error: Invalid chat mode.")

    chat_history = chat_session.chat_history
    # Use temporary chat history for follow-up questions if available
    if hasattr(chat_session, 'temp_chat_history') and chat_session.temp_chat_history:
        context_chat_history = chat_session.temp_chat_history
        # Clear the temporary chat history after using it
        chat_session.temp_chat_history = None
    else:
        context_chat_history = chat_history

    # Handle initial welcome message when chat history is empty
    initial_message_start_time = time.time()
    if user_input == "Can you give me a summary of this document?" or not chat_history:
        try:
            # Try to load existing document summary
            document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
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
        source_react_annotations = []
        refined_source_pages = {}
        follow_up_questions = generate_follow_up_questions(answer, [])

        return answer, sources, source_pages, source_react_annotations, refined_source_pages, follow_up_questions
    
    time_tracking['summary_message'] = time.time() - initial_message_start_time
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Regular chat flow
    # Refine user input
    query_start = time.time()
    refined_user_input = get_query_helper(chat_session, user_input, context_chat_history, embedding_folder)
    logger.info(f"Refined user input: {refined_user_input}")
    time_tracking['query_refinement'] = time.time() - query_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get response
    response_start = time.time()
    answer = await get_response(chat_session, _doc, _document, file_path, refined_user_input, context_chat_history, embedding_folder)
    time_tracking['response_generation'] = time.time() - response_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get sources
    sources_start = time.time()
    sources, source_pages, refined_source_pages = get_response_source(
        chat_session.mode,
        _doc, _document, file_path, refined_user_input, answer, context_chat_history, embedding_folder
    )
    time_tracking['source_retrieval'] = time.time() - sources_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # If the sources have images, append the image URL (in image_urls.json mapping) to the end of the answer in markdown format
    # Process image sources
    images_processing_start = time.time()
    images_sources = {}
    if sources:
        image_url_path = os.path.join(embedding_folder, "markdown/image_urls.json")
        if os.path.exists(image_url_path):
            with open(image_url_path, 'r') as f:
                image_url_mapping = json.load(f)
        else:
            logger.info("Image URL mapping file not found. Creating a new one.")
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
    time_tracking['image_processing'] = time.time() - images_processing_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    
    # Refine and translate the answer to the selected language
    translation_start = time.time()
    answer = f"""**{refined_user_input}**
    """ + "\n" + answer
    answer = translate_content(
        content=answer,
        target_lang=chat_session.current_language
    )
    time_tracking['translation'] = time.time() - translation_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Append images URL in markdown format to the end of the answer
    annotations_start = time.time()
    if images_sources:
        for source, _ in images_sources.items():
            image_url = image_url_mapping.get(source)
            if image_url:
                answer += "\n"
                answer += f"![]({image_url})"

    source_react_annotations = []
    for source, _ in sources.items():
        react_annotations = get_highlight_info(_doc, [source])
        source_react_annotations.extend(react_annotations)
    time_tracking['annotations'] = time.time() - annotations_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Combine regular sources with image sources
    sources.update(images_sources)

    # Generate follow-up questions
    followup_start = time.time()
    follow_up_questions = generate_follow_up_questions(answer, chat_history)
    time_tracking['followup_questions'] = time.time() - followup_start
    logger.info(f"File id: {file_hash}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    return answer, sources, source_pages, source_react_annotations, refined_source_pages, follow_up_questions


async def get_response(chat_session: ChatSession, _doc, _document, file_path, user_input, chat_history, embedding_folder, deep_thinking = False):
    if chat_session.mode == ChatMode.ADVANCED:
        try:
            # Convert the synchronous call to await the response
            answer = await get_GraphRAG_global_response(_doc, _document, user_input, chat_history, embedding_folder)
            return answer
        except Exception as e:
            print("Error getting response from GraphRAG:", e)
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

    config = load_config()
    para = config['llm']
    llm = get_llm(para["level"], para)
    parser = StrOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    embeddings = get_embedding_models('default', para)

    # Check if all necessary files exist to load the embeddings
    await generate_embedding(
        chat_session.mode,
        _document, _doc, file_path, embedding_folder,
    )

    # Load existing embeddings
    # print("Loading existing embeddings...")
    logger.info("Loading existing embeddings...")
    try:
        # Load markdown embeddings
        print(f"Loading markdown embeddings from {os.path.join(embedding_folder, 'markdown')}")
        db = FAISS.load_local(
            os.path.join(embedding_folder, "markdown"), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        # If markdown embeddings are not found, load the default embeddings
        logger.info("Markdown embeddings not found. Loading default embeddings...")
        print(f"Error loading markdown embeddings: {e}")
        db = FAISS.load_local(
            embedding_folder, embeddings, allow_dangerous_deserialization=True
        )

    config = load_config()
    retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

    if not deep_thinking:
        logger.info("not deep thinking ...")
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
        logger.info("deep thinking ...")
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
        logger.info("before deepseek_inference ...")
        answer = str(deepseek_inference(prompt))
        logger.info("after deepseek_inference ...")

        # Extract the content between <think> and </think> as answer_thinking, and the rest as answer_summary
        answer_thinking = answer.split("<think>")[1].split("</think>")[0]
        answer_summary = answer.split("<think>")[1].split("</think>")[1]
        answer_summary = responses_refine(answer_summary, "")
        # answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary
        answer = answer_summary
    logger.info("get_response done ...")
    return answer


def get_query_helper(chat_session: ChatSession, user_input, context_chat_history, embedding_folder):
    # If we have "documents_summary" in the embedding folder, we can use it to speed up the search
    document_summary_path = os.path.join(embedding_folder, "documents_summary.txt")
    if os.path.exists(document_summary_path):
        with open(document_summary_path, "r") as f:
            documents_summary = f.read()
    else:
        documents_summary = " "

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
                                  "context": documents_summary,
                                  "chat_history": truncate_chat_history(context_chat_history)})
    question = parsed_result['question']
    question_type = parsed_result['question_type']
    try:
        language = detect_language(user_input)
        print("language detected:", language)
    except Exception as e:
        print("Error detecting language:", e)
        language = "English"

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