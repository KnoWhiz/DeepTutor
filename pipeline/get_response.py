import os
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
from streamlit_float import *

from pipeline.api_handler import ApiHandler


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


def get_response(_documents, embedding_folder):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
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
        My question is: {input}
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
    return chain


def get_response_source(_documents, embedding_folder):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
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
    return chain