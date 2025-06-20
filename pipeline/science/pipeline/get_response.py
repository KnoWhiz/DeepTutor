import os
import re
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.utils import (
    truncate_chat_history,
    get_llm,
    responses_refine,
    count_tokens,
    replace_latex_formulas,
    generators_list_stream_response,
    Question,
    truncate_document
)
from pipeline.science.pipeline.doc_processor import (
    extract_document_from_file,
    process_pdf_file
)
from pipeline.science.pipeline.embeddings import (
    get_embedding_models,
    load_embeddings,
)
from pipeline.science.pipeline.content_translator import (
    detect_language,
    translate_content
)
from pipeline.science.pipeline.inference import deep_inference_agent
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.get_graphrag_response import get_GraphRAG_global_response
from pipeline.science.pipeline.get_rag_response import (
    get_embedding_folder_rag_response, 
    get_db_rag_response
)
from pipeline.science.pipeline.images_understanding import (
    aggregate_image_contexts_to_urls, 
    create_image_context_embeddings_db, 
    analyze_image
)
from pipeline.science.pipeline.rag_agent import get_rag_context

import logging
logger = logging.getLogger("tutorpipeline.science.get_response")


async def get_multiple_files_summary(file_path_list, embedding_folder_list, chat_session=None, stream=False):
    """
    Generate a summary for multiple files by combining previews of each file.
    
    Args:
        file_path_list: List of file paths to generate a summary for
        embedding_folder_list: List of embedding folders
        chat_session: The current chat session for tracking responses
        stream: Whether to stream the response
        
    Returns:
        A generator yielding the summary if stream=True, otherwise a string
    """
    config = load_config()
    llm = get_llm('advanced', config['llm'])
    
    # Log the list of files being processed
    logger.info(f"Processing multiple files for summary: {file_path_list}")
    logger.info(f"Using embedding folders: {embedding_folder_list}")
    
    # Extract first 3000 tokens from each file
    file_previews = []
    for i, file_path in enumerate(file_path_list):
        try:
            logger.info(f"Processing file {i+1}/{len(file_path_list)}: {file_path}")
            # Process the PDF file properly
            document, doc = process_pdf_file(file_path)
            logger.info(f"File {os.path.basename(file_path)} processed, document has {len(document)} pages")
            
            # Extract text from the document
            file_content = ""
            for page_doc in document:
                if hasattr(page_doc, 'page_content') and page_doc.page_content:
                    file_content += page_doc.page_content.strip() + "\n"
            
            # # Calculate token limit per file - maximum 3000 tokens per file but adjust for file count
            # token_limit = min(3000, 10000 // len(file_path_list))
            token_limit = int(config["basic_token_limit"] * 0.1 // len(file_path_list))
            
            # Get total tokens in the content
            try:
                total_tokens = count_tokens(file_content)
                logger.info(f"File {file_path} has {total_tokens} tokens, limiting to {token_limit}")
                
                # Log the first chunk of content (up to 200 chars)
                first_content_preview = file_content[:200].replace("\n", " ") + "..."
                logger.info(f"First content chunk for {os.path.basename(file_path)}: {first_content_preview}")
                
                # Truncate to token limit
                if total_tokens > token_limit:
                    # Take approximately the first X tokens (by characters, not exact)
                    char_limit = int(len(file_content) * (token_limit / total_tokens))
                    truncated_content = file_content[:char_limit]
                    truncated_content += "\n\n[Content truncated due to length...]"
                    logger.info(f"File {os.path.basename(file_path)} truncated from {len(file_content)} to {len(truncated_content)} characters")
                else:
                    truncated_content = file_content
                    logger.info(f"File {os.path.basename(file_path)} content used in full ({len(file_content)} characters)")
            except Exception as e:
                logger.exception(f"Error calculating tokens for {file_path}: {str(e)}")
                # Fallback to character-based approximation (roughly 4 chars per token)
                char_limit = token_limit * 4
                if len(file_content) > char_limit:
                    truncated_content = file_content[:char_limit]
                    truncated_content += "\n\n[Content truncated due to length...]"
                    logger.info(f"Using character-based fallback: truncated {os.path.basename(file_path)} to {char_limit} characters")
                else:
                    truncated_content = file_content
                    logger.info(f"Using character-based fallback: using full content of {os.path.basename(file_path)}")
            
            file_previews.append((os.path.basename(file_path), truncated_content))
            logger.info(f"Extracted preview from {os.path.basename(file_path)}: {len(truncated_content)} characters")
        except Exception as e:
            logger.exception(f"Error extracting content from {file_path}: {str(e)}")
            file_name = os.path.basename(file_path)
            file_previews.append((file_name, f"Error extracting content: {str(e)}"))
    
    # Format the prompt with file previews
    prompt_parts = []
    for i, (file_name, preview) in enumerate(file_previews):
        prompt_parts.append(f"\n\n### DOCUMENT {i+1}: {file_name}\n\nPreview Content:\n```\n{preview}\n```\n")
    
    formatted_previews = "\n".join(prompt_parts)
    logger.info(f"Created formatted previews for {len(file_previews)} files, total length: {len(formatted_previews)} characters")
    
    prompt = f"""
    You are an expert academic tutor helping a student understand multiple documents. 
    The student has loaded multiple PDF files and needs a comprehensive summary that explains what each document is about.
    
    Here are the files with previews of their content:
    {formatted_previews}
    
    Please provide a comprehensive summary that:
    1. Introduces each document with its title (derived from content if possible) and main topic
    2. Summarizes the key content and main findings of each document
    3. Identifies relationships or connections between the documents (they appear to be related scientific papers)
    4. Highlights the most important concepts across all documents
    5. Uses markdown formatting for clear organization with sections and subsections
    6. Makes appropriate use of bold, bullet points, and other formatting to improve readability
    7. Highest title level is 3, and the title should be concise and informative.
    
    Format your summary with a friendly welcome message at the beginning and a closing "Ask me anything" message at the end.
    """
    
    logger.info(f"Generated summary prompt with length: {len(prompt)} characters")
    logger.info(f"Generating summary for multiple files: {[os.path.basename(fp) for fp in file_path_list]}")
    
    if stream:
        # Stream response for real-time feedback - remove thinking part
        logger.info("Using streaming mode for summary generation")
        answer = llm.stream(prompt)
        
        def process_stream():
            yield "<response>\n\n"
            for chunk in answer:
                # Convert AIMessageChunk to string
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
            yield "\n\n</response>"
            logger.info("Completed streaming summary generation")
        
        return process_stream()
    else:
        # Return complete response at once - remove thinking part
        logger.info("Using non-streaming mode for summary generation")
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Generated summary with length: {len(response_text)} characters")
        return f"<response>\n\n{response_text}\n\n</response>"


async def get_response(chat_session: ChatSession, file_path_list, question: Question, chat_history, embedding_folder_list, deep_thinking = True, stream=False):
    generators_list = []
    config = load_config()
    user_input = question.text
    user_input_string = str(user_input + "\n\n" + question.special_context)
    
    # Check if this is a summary request for multiple files. If so, return a generator from get_multiple_files_summary
    if len(file_path_list) > 1 and user_input == config["summary_wording"]:
        logger.info("Handling multiple files summary request.")
        return await get_multiple_files_summary(file_path_list, embedding_folder_list, chat_session, stream)
    
    # Handle Lite mode first
    if chat_session.mode == ChatMode.LITE:
        # lite_prompt = """
        # You are an expert tutor specializing in precise, professional explanations of complex document content.
        # CONTEXT INFORMATION:
        # - Previous conversation: {chat_history}
        # - Reference content from document: {context}
        # USER QUESTION:
        # {input}
        # RESPONSE GUIDELINES:
        # 1. Provide concise, accurate answers directly addressing the question
        # 2. Use clear, precise language with appropriate technical terminology
        # 3. Format key concepts and important points in **bold**
        # 4. Maintain a professional, academic tone throughout the response
        # 5. Break down complex information into structured, logical segments
        # 6. When explaining technical concepts, include relevant examples or applications
        # 7. Clearly state limitations of explanations when uncertainty exists
        # 8. Use bullet points or numbered lists for sequential explanations
        # 9. Cite specific parts of the document when directly referencing content
        # Your goal is to deliver accurate, clear, and professionally structured responses that enhance comprehension of complex topics.
        # """
        # actual_embedding_folder_list = [os.path.join(embedding_folder, 'lite_embedding') for embedding_folder in embedding_folder_list]
        # db = load_embeddings(actual_embedding_folder_list, 'lite')
        # logger.info(f"Type of db: {type(db)}")
        # answer = await get_db_rag_response(
        #     prompt_string=lite_prompt,
        #     user_input=user_input + "\n\n" + question.special_context,
        #     chat_history=chat_history,
        #     chat_session=chat_session,
        #     db=db,
        #     stream=stream
        # )   # If stream is True, the answer is a generator; otherwise, it's a string
        # if stream is True:
        #     return answer
        # else:
        #     return answer

        config = load_config()
        token_limit = config["inference_token_limit"]
        map_symbol_to_index = config["map_symbol_to_index"]
        # Get the first 3 keys from map_symbol_to_index for examples in the prompt
        first_keys = list(map_symbol_to_index.keys())[:3]
        example_keys = ", or ".join(first_keys)
        logger.info(f"embedding_folder_list in get_response: {embedding_folder_list}")
        formatted_context = await get_rag_context(chat_session=chat_session,
                                            file_path_list=file_path_list,
                                            question=question,
                                            chat_history=chat_history,
                                            embedding_folder_list=embedding_folder_list,
                                            deep_thinking=deep_thinking,
                                            stream=stream,
                                            context="")
        # formatted_context_string = str(formatted_context)
        formatted_context_string = chat_session.formatted_context
        prompt = f"""
        You are a deep thinking tutor helping a student reading a paper.
        Reference context chunks with relevance scores from the paper: 
        {formatted_context_string}
        The student's query is: {user_input_string}
        For formulas, use LaTeX format with $...$ or 
        $$
        ...
        $$
        and make sure latex syntax can be properly rendered in the response.

        RESPONSE GUIDELINES:
        1. Provide concise, accurate answers directly addressing the question
        2. Use clear, precise language with appropriate technical terminology
        3. Format key concepts and important points in **bold**
        4. Maintain a professional, academic tone throughout the response
        5. Break down complex information into structured, logical segments
        6. When explaining technical concepts, include relevant examples or applications
        7. Clearly state limitations of explanations when uncertainty exists
        8. Use bullet points or numbered lists for sequential explanations
        9. Cite specific parts of the document when directly referencing content
        Your goal is to deliver accurate, clear, and professionally structured responses that enhance comprehension of complex topics.

        Requirement:
        Only use the information from the context chunks to answer the question. Give the response in a scientific and academic tone. Do not make up or assume anything or guess without any evidence. If you answer some questions based on your own knowledge, clearly state that you are using your own knowledge.

        Format requirement:
        1. Make sure each sentence in the response there is a corresponding context chunk to support the sentence, and cite the most relevant context chunk keys in the format "[<chunk_key, like {example_keys}, etc>]" at the end of the sentence after the period mark. If there are more than one context chunk keys, use the format "[<chunk_key_1>][<chunk_key_2>] ..." to cite all the context chunk keys.
        2. Use markdown syntax for formatting the response to make it more clear and readable.
        """
        # prompt = ChatPromptTemplate.from_template(prompt)
        llm = get_llm('advanced', config['llm'])
        # chain = prompt | llm | StrOutputParser()
        answer = llm.stream(prompt)
        def process_stream():
            yield "<response>\n\n"
            for chunk in answer:
                # Convert AIMessageChunk to string
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
            yield "\n\n</response>"
        return process_stream()

    # Handle Advanced mode
    if chat_session.mode == ChatMode.ADVANCED:
        try:
            answer = await get_GraphRAG_global_response(question=question, 
                                                        chat_history=chat_history, 
                                                        file_path_list=file_path_list,
                                                        embedding_folder_list=embedding_folder_list, 
                                                        deep_thinking=deep_thinking, 
                                                        chat_session=chat_session, 
                                                        stream=stream)
            logger.info(f"TEST: chat_session.formatted_context after get_GraphRAG_global_response: {chat_session.formatted_context}")
            return answer
        except Exception as e:
            logger.exception("Error getting response from GraphRAG:", e)
            import traceback
            traceback.print_exc()

    # Handle Basic mode
    if not deep_thinking:
        logger.info("not deep thinking ...")
        basic_prompt = """
            You are a patient and honest professor helping a student reading a paper.
            Use the given context to answer the question.
            If you don't know the answer, say you don't know.
            Reference context from the paper: ```{context}```
            If the concept can be better explained by formulas, use LaTeX syntax in markdown
            For inline formulas, use single dollar sign: $a/b = c/d$
            For block formulas, use double dollar sign:
            \n$$
            \frac{{a}}{{b}} = \frac{{c}}{{d}}
            \n$$
            """ + "\n\nThis is a detailed plan for constructing the answer: " + str(question.answer_planning)

        # Load embeddings for Non-deep thinking mode
        try:
            logger.info(f"Loading markdown embeddings from {[os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]}")
            markdown_embedding_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
            db = load_embeddings(markdown_embedding_folder_list, 'default')
        except Exception as e:
            logger.exception(f"Failed to load markdown embeddings for Non-deep thinking mode: {str(e)}")
            db = load_embeddings(embedding_folder_list, 'default')

        answer = await get_db_rag_response(
            prompt_string=basic_prompt,
            user_input=user_input + "\n\n" + question.special_context,
            chat_history=chat_history,
            chat_session=chat_session,
            db=db,
            stream=stream
        )
        if stream is True:
            # If stream is True, the answer is a generator; otherwise, it's a string
            # FIXME: Later we can add response_refine to the generator
            return answer
        else:   # If stream is False, we need to refine the answer
            answer = responses_refine(answer, stream=stream)
            return answer
    else:
        logger.info("deep thinking ...")

        # Load config for deep thinking mode
        config = load_config()
        token_limit = config["inference_token_limit"]
        map_symbol_to_index = config["map_symbol_to_index"]
        # Get the first 3 keys from map_symbol_to_index for examples in the prompt
        first_keys = list(map_symbol_to_index.keys())[:3]
        example_keys = ", or ".join(first_keys)
        chat_history_string = truncate_chat_history(chat_history, token_limit=token_limit)
        # user_input_string = str(user_input + "\n\n" + question.special_context)
        
        formatted_context = await get_rag_context(chat_session=chat_session,
                                            file_path_list=file_path_list,
                                            question=question,
                                            chat_history=chat_history,
                                            embedding_folder_list=embedding_folder_list,
                                            deep_thinking=deep_thinking,
                                            stream=stream,
                                            context="")
        # formatted_context_string = str(formatted_context)
        formatted_context_string = chat_session.formatted_context

        prompt = f"""
        You are a deep thinking tutor helping a student reading a paper.
        Reference context chunks with relevance scores from the paper: 
        {formatted_context_string}
        This is a detailed plan for constructing the answer: {str(question.answer_planning)}
        The student's query is: {user_input_string}
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
        logger.info(f"Size of whole prompt: {len(prompt)}")
        if stream is False:
            try:
                answer = str(deep_inference_agent(user_prompt=prompt, stream=stream, chat_session=chat_session))
            except Exception as e:
                logger.exception(f"Error in deep_inference_agent with chat history, retry with no chat history: {e}")
                prompt = f"""
                You are a deep thinking tutor helping a student reading a paper.
                Reference context from the paper: {formatted_context_string}
                This is a detailed plan for constructing the answer: {str(question.answer_planning)}
                The student's query is: {user_input_string}
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
                answer = str(deep_inference_agent(user_prompt=prompt, stream=stream, chat_session=chat_session))

            if "<think>" in answer:
                answer_thinking = answer.split("<think>")[1].split("</think>")[0]
                answer_summary = answer.split("<think>")[1].split("</think>")[1]
                answer_summary_refined = responses_refine(answer_summary, "", stream=stream)
                answer = answer_summary_refined
            else:
                answer = responses_refine(answer, stream=stream)

            # Replace LaTeX formulas in the final answer
            answer = replace_latex_formulas(answer)

            logger.info("get_response done ...")
            return answer
        else:
            # If stream is True, the answer is a generator; otherwise, it's a string
            # FIXME: Later we can add response_refine and replace_latex_formulas to the generator
            try:
                logger.info("triggering deep_inference_agent ...")
                answer = deep_inference_agent(user_prompt=prompt, stream=stream, chat_session=chat_session)
            except Exception as e:
                logger.exception(f"Error in deep_inference_agent with chat history, retry with no chat history: {e}")
                prompt = f"""
                You are a deep thinking tutor helping a student reading a paper.
                Reference context from the paper: {formatted_context_string}
                This is a detailed plan for constructing the answer: {str(question.answer_planning)}
                The student's query is: {user_input_string}
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
            return answer


async def get_query_helper(chat_session: ChatSession, user_input, context_chat_history, embedding_folder_list):
    # Replace LaTeX formulas in the format \( formula \) with $ formula $
    user_input = replace_latex_formulas(user_input)

    logger.info(f"TEST: user_input: {user_input}")
    # yield f"\n\n**💬 User input: {user_input}**"
    # If we have "documents_summary" in the embedding folder, we can use it to speed up the search
    document_summary_path_list = [os.path.join(embedding_folder, "documents_summary.txt") for embedding_folder in embedding_folder_list]
    documents_summary_list = []
    for document_summary_path in document_summary_path_list:
        if os.path.exists(document_summary_path):
            with open(document_summary_path, "r") as f:
                documents_summary_list.append(f.read())
        else:
            documents_summary_list.append(" ")

    # Join all the documents summaries into one string
    # FIXME: Add a function to combine the initial messages into a single summary message
    documents_summary = "\n".join(documents_summary_list)

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
            "question_type": "local" or "global" or "image", (if the question is like "what is fig. 1 mainly about?", the question_type should be "image")
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
    try:
        parsed_result = chain.invoke({"input": user_input,
                                    "context": documents_summary,
                                    "chat_history": truncate_chat_history(context_chat_history)})
        question = parsed_result['question']
        question_type = parsed_result['question_type']
    except Exception as e:
        try:
            logger.exception(f"Error in get_query_helper: {e}")
            parsed_result = chain.invoke({"input": user_input,
                                        "context": documents_summary,
                                        "chat_history": truncate_chat_history(context_chat_history)})
            question = parsed_result['question']
            question_type = parsed_result['question_type']
        except Exception as e:
            logger.exception(f"Error again in get_query_helper: {e}")
            question = user_input
            question_type = "local"
    try:
        language = detect_language(user_input)
        logger.info(f"language detected: {language}")
    except Exception as e:
        logger.info(f"Error detecting language: {e}")
        language = "English"

    chat_session.set_language(language)

    # Create the answer planning using an additional LLM call
    planning_system_prompt = (
        """
        You are an educational AI assistant tasked with deeply analyzing a student's question and planning a comprehensive answer.

        Your goal is to:
        1. Understand what the student truly wants to know based on the conversation history and current question
        2. Create a detailed plan for constructing the answer
        3. Identify what information should be included in the answer
        4. Identify what information should NOT be included (e.g., repeated information, information the student already knows)
        5. Do not make up or assume anything or guess without any evidence, only use the information provided in the previous conversation history and current question to analyze the user's intent and what to include and exclude in the answer.
        6. If the query is about a specific figure, please include the figure number in the answer.

        Document summary:
        ```{context}```

        Previous conversation history:
        ```{chat_history}```

        Organize your analysis in the following format:
        ```json
        {{
            "user_intent": "<detailed analysis of what the user truly wants to know. based on the previous conversation history and the current question analyse what user already knows and what user doesn't know>",
            "things_explained_already": ["<list of things that has been explained in the previous conversation and should not be repeated in detail in the answer>"],
            "key_focus_areas": ["<list of specific topics/concepts that should be explained>"],
            "information_to_include": ["<list of specific information points that should be included>"],
            "information_to_exclude": ["<list of information that should be excluded - already known/redundant>"],
            "answer_structure": ["<outline of how the answer should be structured>"],
            "explanation_depth": "<assessment of how detailed the explanation should be (basic/intermediate/advanced)>",
            "misconceptions_to_address": ["<potential misconceptions that should be corrected>"]
        }}
        ```
        """
    )

    planning_human_prompt = (
        """
        The student's current question:
        ```{input}```

        The rephrased question for RAG context:
        ```{rephrased_question}```

        Question type: {question_type}

        Based on the conversation history, document summary, and the current question, create a detailed plan for constructing the answer.
        """
    )

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", planning_system_prompt),
            ("human", planning_human_prompt),
        ]
    )
    parser_string = StrOutputParser()
    error_parser_string = OutputFixingParser.from_llm(parser=parser_string, llm=llm)
    planning_chain = planning_prompt | llm | error_parser_string
    answer_planning = planning_chain.invoke({
        "input": user_input,
        "rephrased_question": question,
        "question_type": question_type,
        "context": documents_summary,
        "chat_history": truncate_chat_history(context_chat_history)
    })
    logger.info(f"TEST: answer_planning: {answer_planning}")

    question = Question(
        text=question,
        language=language,
        question_type=question_type,
        answer_planning=answer_planning,
        image_url=None,
    )
    logger.info(f"TEST: question.question_type: {question.question_type}")
    # yield f"\n\n**Question: {question}**"
    # yield f"\n\n**Question type: {question_type}**"
    # yield f"\n\n**Answer planning: {answer_planning}**"
    # yield f"\n\n**Language: {language}**"
    yield "\n\n**🧠 Answer planning...**"

    if question_type == "image":
        logger.info(f"question_type for input: {user_input} is --image-- ...")
        markdown_folder_list = [os.path.join(embedding_folder, 'markdown') for embedding_folder in embedding_folder_list]
        db, truncated_db = create_image_context_embeddings_db(markdown_folder_list)
        # Replace variations of "fig" or "figure" with "Image" for better matching
        processed_input = re.sub(r"\b(?:[Ff][Ii][Gg](?:ure)?\.?|[Ff]igure)\b", "Image", user_input)

        # Get the image chunks from the truncated database
        truncated_image_chunks = truncated_db.similarity_search_with_score(processed_input, k=1)
        logger.info(f"TEST: truncated_image_chunks for image loading: {truncated_image_chunks}")

        # Map the image chunks to the original database based on the index number of the chunk
        # Find the index of the truncated image chunk in the original database
        image_chunks = db.similarity_search_with_score(truncated_image_chunks[0][0].page_content, k=1)

        image_url_mapping = aggregate_image_contexts_to_urls(markdown_folder_list)
        if image_chunks:
            question.special_context = """
            Here is the context and visual understanding of the corresponding image:
            """ + image_chunks[0][0].page_content # + "\n\n" + image_chunks[1][0].page_content

            # Get the image url from the image chunks
            highest_score_url = None
            highest_score = float('-inf')

            for chunk, score in image_chunks:
                chunk_content = chunk.page_content
                # Check if any key from image_url_mapping exists in the chunk content
                for context_key, url in image_url_mapping.items():
                    if context_key in chunk_content and score > highest_score:
                        highest_score = score
                        highest_score_url = url
                        logger.info(f"Found matching image URL: {url} with score: {score}")

            # Set the image URL with the highest score in the question object
            if highest_score_url:
                question.image_url = highest_score_url
                logger.info(f"Setting image URL in question: {highest_score_url}")

        if question.image_url:
            # Get the images understanding from the image url about the question
            question.special_context = """
            Here is the context and visual understanding of the corresponding image:
            """ + analyze_image(question.image_url, f"The user's question is: {question.text}", f"The user's question is: {question.text}")

        logger.info(f"TEST: question.special_context: {question.special_context}")
    elif question_type == "local":
        logger.info(f"question_type for input: {user_input} is --local-- ...")
    elif question_type == "global":
        logger.info(f"question_type for input: {user_input} is --global-- ...")
    else:
        logger.info(f"question_type for input: {user_input} is unknown ...")

    # TEST: print the question object
    logger.info(f"TEST: question: {question}")
    chat_session.question = question
    yield (question)


def generate_follow_up_questions(answer, chat_history):
    """
    Generate 3 relevant follow-up questions based on the assistant's response and chat history.
    """
    logger.info("Generating follow-up questions ...")
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

    logger.info(f"Generated follow-up questions: {result['questions']}")
    return result["questions"]