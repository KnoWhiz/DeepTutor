import os
import logging
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
    responses_refine,
    detect_language,
    count_tokens
)
from pipeline.science.pipeline.embeddings import (
    get_embedding_models,
    load_embeddings,
)
from pipeline.science.pipeline.inference import deep_inference_agent
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.get_graphrag_response import get_GraphRAG_global_response
from pipeline.science.pipeline.get_rag_response import get_standard_rag_response

logger = logging.getLogger("tutorpipeline.science.get_response")


class Question:
    """
    Represents a question with its text, language, and type information.
    
    Attributes:
        text (str): The text content of the question
        language (str): The detected language of the question (e.g., "English")
        question_type (str): The type of question (e.g., "local" or "global" or "image")
    """
    
    def __init__(self, text="", language="English", question_type="global", special_context=""):
        """
        Initialize a Question object.
        
        Args:
            text (str): The text content of the question
            language (str): The language of the question
            question_type (str): The type of the question (local or global or image)
        """
        self.text = text
        self.language = language
        if question_type not in ["local", "global", "image"]:
            self.question_type = "global"
        else:
            self.question_type = question_type

        self.special_context =  special_context
    
    def __str__(self):
        """Return string representation of the Question."""
        return f"Question(text='{self.text}', language='{self.language}', type='{self.question_type}')"

    def to_dict(self):
        """Convert Question object to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "question_type": self.question_type
        }


async def get_response(chat_session: ChatSession, _doc, _document, file_path, question: Question, chat_history, embedding_folder, deep_thinking = False, stream=False):
    user_input = question.text
    # Handle Lite mode first
    if chat_session.mode == ChatMode.LITE:
        lite_prompt = """You are a helpful tutor assisting with document understanding.
            Use the given context to answer the question.
            If you don't know the answer, say so.
            Previous conversation: {chat_history}
            Reference context: {context}
            
            Please provide a clear and concise answer that:
            1. Directly addresses the question
            2. Uses simple language
            3. Highlights key points in bold
            4. Uses emojis when appropriate to make the response engaging"""

        answer = await get_standard_rag_response(
            prompt_string=lite_prompt,
            user_input=user_input + "\n\n" + question.special_context,
            chat_history=chat_history,
            embedding_folder=embedding_folder,
            embedding_type='lite',
            chat_session=chat_session,
            doc=_doc,
            document=_document,
            file_path=file_path,
            stream=stream
        )
        
        # # For Lite mode, we return empty containers for sources and follow-up questions
        # sources = {}
        # source_pages = {}
        # source_annotations = {}
        # refined_source_pages = {}
        # follow_up_questions = []
        
        return answer

    # Handle Advanced mode
    if chat_session.mode == ChatMode.ADVANCED:
        try:
            answer = await get_GraphRAG_global_response(_doc, _document, user_input, chat_history, embedding_folder, deep_thinking)
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
            The previous conversation is: {chat_history} make sure your answer follow the previous conversation but not repetitive.
            Reference context from the paper: ```{context}```
            If the concept can be better explained by formulas, use LaTeX syntax in markdown
            For inline formulas, use single dollar sign: $a/b = c/d$
            FOr block formulas, use double dollar sign:
            $$
            \frac{{a}}{{b}} = \frac{{c}}{{d}}
            $$
            """

        answer = await get_standard_rag_response(
            prompt_string=basic_prompt,
            user_input=user_input + "\n\n" + question.special_context,
            chat_history=chat_history,
            embedding_folder=embedding_folder,
            stream=stream
        )
        return answer
    else:
        logger.info("deep thinking ...")
        # Load config and embeddings for deep thinking mode
        config = load_config()
        token_limit = config["inference_token_limit"]
        
        try:
            logger.info(f"Loading markdown embeddings from {os.path.join(embedding_folder, 'markdown')}")
            db = load_embeddings(os.path.join(embedding_folder, 'markdown'), 'default')
        except Exception as e:
            logger.exception(f"Failed to load markdown embeddings for deep thinking: {str(e)}")
            db = load_embeddings(embedding_folder, 'default')

        chat_history_string = truncate_chat_history(chat_history, token_limit=token_limit)
        user_input_string = str(user_input + "\n\n" + question.special_context)
        # Get relevant chunks for both question and answer with scores
        question_chunks_with_scores = db.similarity_search_with_score(user_input_string, k=config['retriever']['k'])
        # The total list of sources chunks
        sources_chunks = []
        # From the highest score to the lowest score, until the total tokens exceed 3000
        total_tokens = 0
        for chunk in question_chunks_with_scores:
            if total_tokens + count_tokens(chunk[0].page_content) > token_limit:
                break
            sources_chunks.append(chunk[0])
            total_tokens += count_tokens(chunk[0].page_content)
        context = "\n\n".join([chunk.page_content for chunk in sources_chunks])

        prompt = f"""
        You are a deep thinking tutor helping a student reading a paper.
        The previous conversation is: {chat_history_string}
        Reference context from the paper: {context}
        The student's query is: {user_input_string}
        """
        logger.info(f"user_input_string tokens: {count_tokens(user_input_string)}")
        logger.info(f"chat_history_string tokens: {count_tokens(chat_history_string)}")
        logger.info(f"context tokens: {count_tokens(context)}")
        logger.info("before deep_inference_agent ...")
        try:
            answer = str(deep_inference_agent(prompt))
        except Exception as e:
            logger.exception(f"Error in deep_inference_agent with chat history, retry with no chat history: {e}")
            prompt = f"""
            You are a deep thinking tutor helping a student reading a paper.
            Reference context from the paper: {context}
            The student's query is: {user_input_string}
            """
            answer = str(deep_inference_agent(prompt))

        if "<think>" in answer:
            answer_thinking = answer.split("<think>")[1].split("</think>")[0]
            answer_summary = answer.split("<think>")[1].split("</think>")[1]
            answer_summary = responses_refine(answer_summary, "")
            answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary
        else:
            answer = responses_refine(answer)
        
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
    question = Question(text=question, language=language, question_type=question_type)
    logger.info(f"TEST: question.question_type: {question.question_type}")

    if question_type == "image":
        # Find a single chunk in the embedding folder
        db = load_embeddings(embedding_folder, 'default')
        image_chunks = db.similarity_search_with_score(user_input + "\n\n" + question.special_context, k=1)
        if image_chunks:
            question.special_context = """
            Here is the context and visual understanding of the image:
            """ + image_chunks[0][0].page_content
        logger.info(f"TEST: question.special_context: {question.special_context}")
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