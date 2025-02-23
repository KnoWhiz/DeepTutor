import os

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
    responses_refine,
    detect_language,
)
from pipeline.science.pipeline.doc_processor import (
    generate_embedding,
)
from pipeline.science.pipeline.inference import deepseek_inference
from pipeline.science.pipeline.session_manager import ChatSession, ChatMode
from pipeline.science.pipeline.graphrag_get_response import get_GraphRAG_global_response
import logging
logger = logging.getLogger("tutorpipeline.science.get_response")


async def get_response(chat_session: ChatSession, _doc, _document, file_path, user_input, chat_history, embedding_folder, deep_thinking = False):
    # Handle lite mode first
    if chat_session.mode == ChatMode.LITE:
        config = load_config()
        para = config['llm']
        llm = get_llm('basic', para)
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        embeddings = get_embedding_models('lite', para)
        lite_embedding_folder = os.path.join(embedding_folder, 'lite_embedding')

        # Create or load text chunks for RAG
        try:
            db = FAISS.load_local(lite_embedding_folder, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            await generate_embedding(chat_session.mode, _doc, _document, file_path, embedding_folder=embedding_folder)
            db = FAISS.load_local(lite_embedding_folder, embeddings, allow_dangerous_deserialization=True)

        # Set up RAG retriever
        retriever = db.as_retriever(search_kwargs={"k": config['retriever']['k']})

        # Create prompt for lite mode
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful tutor assisting with document understanding.
                Use the given context to answer the question.
                If you don't know the answer, say so.
                Previous conversation: {chat_history}
                Reference context: {context}
                
                Please provide a clear and concise answer that:
                1. Directly addresses the question
                2. Uses simple language
                3. Highlights key points in bold
                4. Uses emojis when appropriate to make the response engaging"""),
            ("human", "{input}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": lambda x: format_docs(x["context"]),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | llm
            | error_parser
        )

        # Pass input query to retriever
        retrieve_docs = (lambda x: x["input"]) | retriever
        chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain
        )

        # Get response
        parsed_result = chain.invoke({
            "input": user_input,
            "chat_history": truncate_chat_history(chat_history)
        })
        answer = parsed_result['answer']
        
        # For lite mode, we return empty containers for sources and follow-up questions
        sources = {}
        source_pages = {}
        source_react_annotations = []
        refined_source_pages = {}
        follow_up_questions = []
        
        return answer, sources, source_pages, source_react_annotations, refined_source_pages, follow_up_questions

    # Handle advanced mode
    if chat_session.mode == ChatMode.ADVANCED:
        try:
            answer = await get_GraphRAG_global_response(_doc, _document, user_input, chat_history, embedding_folder)
            return answer
        except Exception as e:
            print("Error getting response from GraphRAG:", e)
            import traceback
            traceback.print_exc()

    # Handle basic mode
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
        answer = "### Here is my thinking process\n\n" + answer_thinking + "\n\n### Here is my summarized answer\n\n" + answer_summary
        # answer = answer_summary
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