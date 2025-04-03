import os
import sys
import logging
import pprint
from pathlib import Path

# Add the project root to the Python path so the pipeline module can be found
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent
sys.path.append(str(project_root))
print(f"Added {project_root} to Python path")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.embeddings import get_embedding_models
from pipeline.science.pipeline.config import load_config
from pipeline.science.features_lab.visualize_graph_test import visualize_graph
from langchain_community.document_loaders import PyPDFLoader

def agentic_rag_test(input: str, urls: list[str] = None, file_path_list: list[str] = None):
    if urls:
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
    elif file_path_list:
        docs = [PyPDFLoader(file_path).load() for file_path in file_path_list]
        docs_list = [item for sublist in docs for item in sublist]
    else:
        raise ValueError("Either urls or file_path_list must be provided")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    config = load_config()

    ### Add to vectorDB

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=get_embedding_models('default', config['llm']),
    )
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    tools = [retriever_tool]

    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]

    ### Edges

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        logging.info("---CHECK RELEVANCE---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = get_llm('advanced', config['llm'], stream=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(grade, method="function_calling")

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            logging.info("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            logging.info("---DECISION: DOCS NOT RELEVANT---")
            logging.info(score)
            return "rewrite"

    ### Nodes

    def agent(state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        logging.info("---CALL AGENT---")
        messages = state["messages"]
        model = get_llm('advanced', config['llm'], stream=True)
        model = model.bind_tools(tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    def rewrite(state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        logging.info("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        model = get_llm('advanced', config['llm'], stream=True)
        response = model.invoke(msg)
        return {"messages": [response]}


    def generate(state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        logging.info("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
        # Customized prompt
        prompt = """
        You are a helpful assistant that can answer questions about the provided context.
        Here is the context:
        {context}
        Here is the question:
        {question}
        """
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["context", "question"],
        )
        logging.info("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
        logging.info(prompt)

        # LLM
        llm = get_llm('advanced', config['llm'], stream=True)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    # logging.info("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
    # prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like

    ### Define a new graph

    workflow = StateGraph(AgentState)
    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    ### Decide whether to retrieve

    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    ### Edges taken after the `action` node is called.

    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    ### Compile
    
    graph = workflow.compile()
    visualize_graph(graph)

    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    inputs = {
        "messages": [
            ("user", input),
        ]
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    return output


if __name__ == "__main__":
    input = "How is multiplexing implemented in the paper?"
    file_path_list = [
        "/Users/bingranyou/Library/Mobile Documents/com~apple~CloudDocs/Downloads/temp/Multiplexed_single_photon_source_arXiv__resubmit_.pdf",
    ]
    output = agentic_rag_test(input=input, file_path_list=file_path_list)
    pprint.pprint(output)

    # input = "What does Lilian Weng say about the types of agent memory?"
    # urls = [
    #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
    #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    # ]
    # output = agentic_rag_test(input=input, urls=urls)
    # pprint.pprint(output)