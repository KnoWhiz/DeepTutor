import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from streamlit_float import *

from pipeline.config import load_config
from pipeline.utils import (
    tiktoken,
    truncate_chat_history,
    get_llm,
    get_embedding_models
)


def get_response_source(_doc, _documents, user_input, answer, chat_history, embedding_folder):
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
    sources = refine_sources(_doc, _documents, sources)
    return sources


def refine_sources(_doc, _documents, sources):
    """
    Refine sources by checking if they can be found in the document
    Only get first 10 sources
    Show then in the order they are found in the document
    """
    refined_sources = []
    for page in _doc:
        for source in sources:
            text_instances = page.search_for(source)
            if text_instances:
                refined_sources.append(source)

    print(f"refined_sources: {refined_sources}")
    print(f"length of refined_sources: {len(refined_sources)}")
    return refined_sources[:10]