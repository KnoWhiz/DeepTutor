from typing import Literal, Dict, List, Any
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.runnables.graph import MermaidDrawMethod
import matplotlib.pyplot as plt
from PIL import Image
import io
# Import util functions from pipeline to use ApiHandler and get_llm
from pipeline.science.pipeline.utils import get_llm
from pipeline.science.pipeline.config import load_config
from pipeline.science.pipeline.api_handler import ApiHandler
from pipeline.science.pipeline.embeddings import get_embedding_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag_test")

def visualize_graph(graph, save_path="visualizations"):
    """
    Visualize a LangGraph and save it as a PNG file.
    
    Args:
        graph: The compiled LangGraph object
        save_path (str): Directory path to save the visualization
        
    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create the directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Generate a filename based on timestamp
        import time
        timestamp = int(time.time())
        filename = f"graph_visualization_{timestamp}.png"
        full_path = os.path.join(save_path, filename)
        
        # Generate the visualization
        logger.info(f"Generating graph visualization...")
        png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        
        # Save the PNG data to a file
        with open(full_path, "wb") as f:
            f.write(png_data)
        
        logger.info(f"Graph visualization saved to {full_path}")
        return full_path
    
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        return None

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

def handle_tools(state: MessagesState) -> MessagesState:
    """Execute tool calls from the latest AI message."""
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # No tool calls to execute
        return state
    
    new_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        
        # Find the correct tool to call
        tool_to_use = None
        if tool_name == "retrieve_context":
            tool_to_use = retrieve_context
        
        if tool_to_use:
            # Call the tool with the provided arguments
            tool_result = tool_to_use(**arguments)
            # Create a tool message with the result
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
            new_messages.append(tool_message)
    
    return {"messages": messages + new_messages}

def agentic_rag():
    try:
        # Configure tools
        tools = [retrieve_context]
        
        # Load config to get LLM parameters
        config = load_config()
        llm_params = config['llm']
        
        # Use get_llm to get the advanced model (gpt-4o), with stream=False for tool calling
        model = get_llm('advanced', llm_params, stream=False)
        
        # Bind tools to the model
        model = model.bind_tools(tools)

        # Function to decide whether to continue or stop the workflow
        def should_continue(state: MessagesState) -> Literal["tools", "agent", END]:
            messages = state['messages']
            last_message = messages[-1]
            
            # If the last message is from the AI and has tool calls, route to tools node
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            # If the last message is a tool message, route back to the agent
            elif isinstance(last_message, ToolMessage):
                return "agent"
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
        workflow.add_node("tools", handle_tools)

        # Connect nodes
        workflow.add_edge(START, "agent")  # Initial entry
        workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
        workflow.add_conditional_edges("tools", should_continue)  # Decision after the "tools" node

        # Configure memory to persist the state
        checkpointer = MemorySaver()

        # Compile the graph into a LangChain Runnable application
        app = workflow.compile(checkpointer=checkpointer)
        
        # Visualize and save the graph
        visualization_path = visualize_graph(app)
        if visualization_path:
            logger.info(f"Graph visualization saved to: {visualization_path}")

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
