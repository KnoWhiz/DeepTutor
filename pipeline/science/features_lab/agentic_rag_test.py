from typing import Literal
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path so the pipeline module can be found
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent.parent
sys.path.append(str(project_root))
print(f"Added {project_root} to Python path")

# Now import the modules
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
# Import util functions from pipeline to use ApiHandler and get_llm
from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.embeddings import get_embedding_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag_test")

# Define the tool for context retrieval
@tool
def retrieve_context(query: str):
    """Search for relevant documents."""
    try:
        # Example URL configuration
        urls = [
            "https://docs.python.org/3/tutorial/index.html",
            "https://realpython.com/python-basics/",
            "https://www.learnpython.org/"
        ]
        # Load documents
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(docs)

        # Use embeddings from the pipeline
        config = load_config()
        # The get_embedding_models function returns the model directly, not a dictionary
        embedding_model = get_embedding_models('default', config['llm'])

        # Create VectorStore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="python_docs",
            embedding=embedding_model,
        )
        retriever = vectorstore.as_retriever()
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {e}"

def agentic_rag():
    try:
        # Configure tools
        tools = [retrieve_context]
        tool_node = ToolNode(tools)

        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the advanced model (gpt-4o), with stream=False for tool calling
        model = get_llm('advanced', llm_params, stream=False)
        
        # Bind tools to the model
        model = model.bind_tools(tools)

        # Function to decide whether to continue or stop the workflow
        def should_continue(state: MessagesState) -> Literal["tools", END]:
            messages = state['messages']
            last_message = messages[-1]
            # If the LLM makes a tool call, go to the "tools" node
            if last_message.tool_calls:
                return "tools"
            # Otherwise, finish the workflow
            return END

        # Function that invokes the model
        def call_model(state: MessagesState):
            messages = state['messages']
            response = model.invoke(messages)
            return {"messages": [response]}  # Returns as a list to add to the state

        # Define the workflow with LangGraph
        workflow = StateGraph(MessagesState)

        # Add nodes to the graph
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # Connect nodes
        workflow.add_edge(START, "agent")  # Initial entry
        workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
        workflow.add_edge("tools", "agent")  # Cycle between tools and agent

        # Configure memory to persist the state
        checkpointer = MemorySaver()

        # Compile the graph into a LangChain Runnable application
        app = workflow.compile(checkpointer=checkpointer)

        # Execute the workflow
        logger.info("Starting workflow execution...")
        final_state = app.invoke(
            {"messages": [HumanMessage(content="Explain what a list is in Python")]},
            config={"configurable": {"thread_id": 42}}
        )

        # Show the final response
        print("Final response:\n\n")
        print(final_state["messages"][-1].content)
        
        return final_state["messages"][-1].content
    except Exception as e:
        logger.error(f"Error in agentic_rag: {e}")
        raise

if __name__ == "__main__":
    try:
        # Check if API keys are set
        from dotenv import load_dotenv
        load_dotenv()
        
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Run the test
        result = agentic_rag()
    except Exception as e:
        logger.error(f"Test failed: {e}")
