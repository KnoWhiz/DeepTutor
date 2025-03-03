import os
import json
import time
from typing import Dict

from pipeline.science.pipeline.utils import (
    translate_content,
    generate_file_id,
    format_time_tracking,
)
from pipeline.science.pipeline.doc_processor import (
    save_file_txt_locally,
    process_pdf_file,
    get_highlight_info,
)
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.helper.index_files_saving import (
    vectorrag_index_files_decompress,
    vectorrag_index_files_compress,
    graphrag_index_files_decompress,
    graphrag_index_files_compress,
    literag_index_files_decompress,
)
from pipeline.science.pipeline.embeddings_agent import embeddings_agent
from pipeline.science.pipeline.get_response import (
    get_query_helper,
    get_response,
    generate_follow_up_questions,
    Question,
)
from pipeline.science.pipeline.sources_retrieval import get_response_source
from pipeline.science.pipeline.config import load_config
import logging
logger = logging.getLogger("tutorpipeline.science.tutor_agent")


async def tutor_agent(chat_session: ChatSession, file_path, user_input, time_tracking=None, deep_thinking=True):
    """
    Taking the user input, document, and chat history, generate a response and sources.
    If user_input is None, generates the initial welcome message.
    """
    # If user_input starts with follow_up_questions_wording, remove it
    config = load_config()
    follow_up_questions_wording = config['follow_up_questions_wording']
    if user_input and user_input.startswith(follow_up_questions_wording):
        user_input = user_input[len(follow_up_questions_wording):]
        logger.info(f"User input after removing follow_up_questions_wording: {user_input}")

    stream = config['stream']
    if time_tracking is None:
        time_tracking: Dict = {}

    # Compute hashed ID and prepare embedding folder
    hashing_start_time = time.time()
    file_id = generate_file_id(file_path)
    file_id = file_id
    embedding_folder = os.path.join('embedded_content', file_id)
    logger.info(f"Embedding folder: {embedding_folder}")
    if not os.path.exists('embedded_content'):
        os.makedirs('embedded_content')
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    time_tracking['file_hashing_setup_dirs'] = time.time() - hashing_start_time
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Save the file txt content locally
    save_file_start_time = time.time()
    # with open(file_path, 'rb') as file:
    #     file_bytes = file.read()
    filename = os.path.basename(file_path)
    save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
    # Process file and create session states for document and PDF object
    _document, _doc = process_pdf_file(file_path)
    time_tracking['file_loading_save_text'] = time.time() - save_file_start_time
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    if chat_session.mode == ChatMode.LITE:
        logger.info("Lite mode - using raw text only")
        lite_embedding_start_time = time.time()
        if(literag_index_files_decompress(embedding_folder)):
            # Check if the LiteRAG index files are ready locally
            logger.info("LiteRAG embedding index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            logger.info("Lite embedding ...")
            await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder)
        time_tracking['lite_embedding_total'] = time.time() - lite_embedding_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
        logger.info("Lite embedding done ...")

    elif chat_session.mode == ChatMode.BASIC:
        vectorrag_start_time = time.time()
        logger.info("BASIC (VectorRAG) mode")
        # Doc processing
        if(vectorrag_index_files_decompress(embedding_folder)):
            logger.info("VectorRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            time_tracking = await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            if(vectorrag_index_files_compress(embedding_folder)):
                logger.info("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                time_tracking = await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                if(vectorrag_index_files_compress(embedding_folder)):
                    logger.info("VectorRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    logger.info("Error compressing and uploading VectorRAG index files to Azure Blob Storage.")
        time_tracking['vectorrag_generate_embedding_total'] = time.time() - vectorrag_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    elif chat_session.mode == ChatMode.ADVANCED:
        graphrag_start_time = time.time()
        logger.info("Advanced (GraphRAG) mode")
        if(graphrag_index_files_decompress(embedding_folder)):
            logger.info("GraphRAG index files are ready.")
        else:
            # Files are missing and have been cleaned up
            save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
            time_tracking = await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
            logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
            # asyncio.run(generate_GraphRAG_embedding(embedding_folder=embedding_folder))
            if(graphrag_index_files_compress(embedding_folder)):
                logger.info("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
            else:
                # Retry once if first attempt fails
                save_file_txt_locally(file_path, filename=filename, embedding_folder=embedding_folder)
                time_tracking = await embeddings_agent(chat_session.mode, _document, _doc, file_path, embedding_folder=embedding_folder, time_tracking=time_tracking)
                logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
                # asyncio.run(generate_GraphRAG_embedding(embedding_folder=embedding_folder))
                if(graphrag_index_files_compress(embedding_folder)):
                    logger.info("GraphRAG index files are ready and uploaded to Azure Blob Storage.")
                else:
                    logger.info("Error compressing and uploading GraphRAG index files to Azure Blob Storage.")
        time_tracking['graphrag_generate_embedding_total'] = time.time() - graphrag_start_time
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
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
        if chat_session.mode == ChatMode.LITE:
            initial_message = "Hello! How can I assist you today?"
        else:
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
        source_annotations = {}
        refined_source_pages = {}
        if chat_session.mode != ChatMode.LITE:
            follow_up_questions = generate_follow_up_questions(answer, [])
        else:
            follow_up_questions = []

        follow_up_questions_wording = config['follow_up_questions_wording']
        follow_up_questions = [follow_up_questions_wording + question for question in follow_up_questions]

        for i in range(len(follow_up_questions)):
            follow_up_questions[i] = translate_content(
                content=follow_up_questions[i],
                target_lang=chat_session.current_language
            )

        return answer, sources, source_pages, source_annotations, refined_source_pages, follow_up_questions

    time_tracking['summary_message'] = time.time() - initial_message_start_time
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Regular chat flow
    # Refine user input
    refined_user_input = user_input
    if chat_session.mode != ChatMode.LITE:
        query_start = time.time()
        question = get_query_helper(chat_session, user_input, context_chat_history, embedding_folder)
        refined_user_input = question.text
        logger.info(f"Refined user input: {refined_user_input}")
        time_tracking['query_refinement'] = time.time() - query_start
        logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")
    else:
        question = Question(text=user_input, language=chat_session.current_language, question_type="local")

    # Get response
    response_start = time.time()
    response = await get_response(chat_session, _doc, _document, file_path, question, context_chat_history, embedding_folder, deep_thinking=deep_thinking, stream=stream)
    answer = response[0] if isinstance(response, tuple) else response
    time_tracking['response_generation'] = time.time() - response_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Get sources
    sources = {}
    source_pages = {}
    refined_source_pages = {}
    sources_start = time.time()
    if chat_session.mode != ChatMode.LITE:
        sources, source_pages, refined_source_pages = get_response_source(
            chat_session.mode,
            _doc, _document, file_path, refined_user_input, answer, context_chat_history, embedding_folder
        )
    time_tracking['source_retrieval'] = time.time() - sources_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

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
            if any(source.lower().endswith(ext) for ext in config["image_extensions"]):
                image_url = image_url_mapping.get(source, None)
                if image_url:
                    images_sources[source] = score
                    sources_to_remove.append(source)
        # Remove processed image sources from the main sources dict
        for source in sources_to_remove:
            del sources[source]
    time_tracking['image_processing'] = time.time() - images_processing_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Refine and translate the answer to the selected language
    translation_start = time.time()
    # answer = f"""**{refined_user_input}**
    # """ + "\n" + answer
    answer = translate_content(
        content=answer,
        target_lang=chat_session.current_language
    )
    # print(f"translate_content Answer: {answer}")
    time_tracking['translation'] = time.time() - translation_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Append images URL in markdown format to the end of the answer
    annotations_start = time.time()
    if images_sources:
        for source, _ in images_sources.items():
            image_url = image_url_mapping.get(source)
            if image_url:
                answer += "\n"
                answer += f"![]({image_url})"

    source_annotations = {}
    for source, _ in sources.items():
        annotations, _ = get_highlight_info(_doc, [source])
        source_annotations[source] = annotations
    time_tracking['annotations'] = time.time() - annotations_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Combine regular sources with image sources
    sources.update(images_sources)

    # Generate follow-up questions
    followup_start = time.time()
    follow_up_questions = generate_follow_up_questions(answer, chat_history)
    for i in range(len(follow_up_questions)):
        follow_up_questions[i] = translate_content(
            content=follow_up_questions[i],
            target_lang=chat_session.current_language
        )
    time_tracking['followup_questions'] = time.time() - followup_start
    logger.info(f"File id: {file_id}\nTime tracking:\n{format_time_tracking(time_tracking)}")

    # Memory clean up 
    _document = None
    _doc = None

    logger.info(f"sources: {sources}")
    logger.info(f"source_pages: {source_pages}")
    logger.info(f"refined_source_pages: {refined_source_pages}")
    logger.info(f"source_annotations: {source_annotations}")

    return answer, sources, source_pages, source_annotations, refined_source_pages, follow_up_questions
